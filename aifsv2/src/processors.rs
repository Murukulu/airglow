use core::{error, fmt};

use burn::{prelude::*, tensor::Bool};
use burn_store::SafetensorsStore;

use crate::{graph::snapshot, metadata::Metadata};

// The anemoi pre/post-processing stage -- everything under `config.data.processors`.
//   pre :  conditional_nan (identity) -> const_imputer -> normalizer
//   post:  normalizer -> const_imputer -> conditional_nan
#[derive(Debug)]
pub struct Processors<B: Backend> {
    normalizer: Normalizer<B>,
    imputer: Imputer<B>,
    conditional_nan: ConditionalNan<B>,
}

/// What `pre` hands to `post`.
pub struct PreProcessed<B: Backend> {
    /// `[batch, time, grid, vars_in]`, imputed and normalised: what the model consumes.
    pub x: Tensor<B, 4>,
    /// Where the imputer filled, laid out over the *output* channels that inherit those NaN:
    /// `[batch * grid, vars_out]`. Already in the frame `post` needs, so it can mask back directly.
    imputed: Tensor<B, 2, Bool>,
}

impl<B: Backend> Processors<B> {
    pub fn load(
        store: &mut SafetensorsStore,
        metadata: &Metadata,
        device: &B::Device,
    ) -> Result<Processors<B>, Error> {
        Ok(Processors {
            normalizer: Normalizer::load(store, device)?,
            imputer: Imputer::new(metadata, device),
            conditional_nan: ConditionalNan::new(metadata, device)?,
        })
    }

    /// `x` is `[batch, time, grid, vars_in]` in physical units, NaN where the source had no value.
    pub fn pre(&self, x: Tensor<B, 4>) -> PreProcessed<B> {
        let (x, imputed) = self.imputer.forward(x);
        PreProcessed {
            x: self.normalizer.forward(x),
            imputed,
        }
    }

    /// `y` is the model's `[batch * grid, vars_out]`, still normalised. Returns physical units.
    pub fn post(&self, y: Tensor<B, 2>, pre: &PreProcessed<B>) -> Tensor<B, 2> {
        let y = self.normalizer.inverse(y);
        let y = y.mask_fill(pre.imputed.clone(), f32::NAN);
        self.conditional_nan.inverse(y)
    }
}

// InputNormalizer -- a pure affine map. The method (mean-std, std, max, none) is resolved at
// training time into two coefficient vectors over the 134-variable dataset space
// (anemoi/models/preprocessing/normalizer.py:71-95), so there is no dispatch to port: only a
// gather through `_input_idx` / `_output_idx` onto the tensor's own channels, done once here.
#[derive(Debug)]
struct Normalizer<B: Backend> {
    input_mul: Tensor<B, 1>,  // [vars_in]
    input_add: Tensor<B, 1>,  // [vars_in]
    output_mul: Tensor<B, 1>, // [vars_out]
    output_add: Tensor<B, 1>, // [vars_out]
}

impl<B: Backend> Normalizer<B> {
    // The pre_processors and post_processors copies of _norm_mul/_norm_add are identical in this
    // checkpoint, but each direction is read from its own namespace rather than assuming that.
    fn load(store: &mut SafetensorsStore, device: &B::Device) -> Result<Normalizer<B>, Error> {
        const PRE: &str = "pre_processors.processors.normalizer";
        const POST: &str = "post_processors.processors.normalizer";

        // These come from the registered buffers in InputNormalizer, which will be stored in the
        // AnemoiModelEncProcDec.
        let input_idx = ints(store, &format!("{PRE}._input_idx"))?;
        let output_idx = ints(store, &format!("{POST}._output_idx"))?;

        // The coefficients are in dataset space; the tensors are in channel space.
        let gather = |values: Vec<f32>, index: &[i32]| {
            let values: Vec<f32> = index.iter().map(|&i| values[i as usize]).collect();
            let len = values.len();
            Tensor::<B, 1>::from_data(TensorData::new(values, [len]), device)
        };

        Ok(Normalizer {
            input_mul: gather(floats(store, &format!("{PRE}._norm_mul"))?, &input_idx),
            input_add: gather(floats(store, &format!("{PRE}._norm_add"))?, &input_idx),
            output_mul: gather(floats(store, &format!("{POST}._norm_mul"))?, &output_idx),
            output_add: gather(floats(store, &format!("{POST}._norm_add"))?, &output_idx),
        })
    }

    // `x.mul_(_norm_mul).add_(_norm_add)` (normalizer.py:162-166), broadcast down the channel axis.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let vars = x.shape().dims::<4>()[3];
        x.mul(self.input_mul.clone().reshape([1, 1, 1, vars]))
            .add(self.input_add.clone().reshape([1, 1, 1, vars]))
    }

    fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let vars = y.shape().dims::<2>()[1];
        y.sub(self.output_add.clone().reshape([1, vars]))
            .div(self.output_mul.clone().reshape([1, vars]))
    }
}

// ConstantImputer, fill value 0 over the variables the checkpoint names.
//
// A named variable joins `fill` only if it has an input channel, and joins the restore pairing
// only if it has both an input and an output channel.
// Ex. For the original checkpoint that is 15 `fill`ed and 14 `restores`. `ro` and `snowc` are
// diagnostic variables with no input channel, `wmb` is a forcing with no output channel. Those two
// diagnostics are what ConditionalNaN picks up instead.
#[derive(Debug)]
struct Imputer<B: Backend> {
    // [vars_in]; true on the channels the fill covers.
    fill: Tensor<B, 1, Bool>,

    // [vars_out]; for each output channel, the input channel whose NaN it inherits. Channels with
    // no counterpart hold an arbitrary index and are masked out by `restores`.
    restore_from: Tensor<B, 1, Int>,

    // [vars_out]; true where input/output pairing exists. Acts to mask over restore_from.
    restores: Tensor<B, 1, Bool>,
}

impl<B: Backend> Imputer<B> {
    fn new(metadata: &Metadata, device: &B::Device) -> Imputer<B> {
        let mut fill = vec![false; metadata.model_input.full.len()];
        let mut restore_from = vec![0i64; metadata.model_output.full.len()];
        let mut restores = vec![false; metadata.model_output.full.len()];

        for name in &metadata.imputer_zero {
            let Some(input) = metadata.input_channel(name) else {
                continue;
            };
            fill[input] = true;
            if let Some(output) = metadata.output_channel(name) {
                restore_from[output] = input as i64;
                restores[output] = true;
            }
        }

        Imputer {
            fill: bools(fill, device),
            restore_from: indices(restore_from, device),
            restores: bools(restores, device),
        }
    }

    // Fill NaN with 0, and record where. anemoi keeps the first timestep's locations only
    // (`nan_locations[:, 0]`, imputer.py:227) and reuses them for the inverse, so a NaN that
    // appears at a later timestep is filled but never restored.
    fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2, Bool>) {
        let shape = x.shape();
        let [batch, _time, grid, vars_in] = shape.dims();
        let vars_out = self.restores.shape().dims::<1>()[0];

        let nan = x.clone().is_nan();

        // Taken before the fill, and rearranged from input to output channels here rather than in
        // `post`: `select` is a gather, so every output channel picks up its own source column and
        // `restores` blanks the ones with no source.
        let imputed = nan
            .clone()
            .slice_dim(1, 0..1) // On time dim, take one step, earliest t0.
            .reshape([batch * grid, vars_in]) // Squash
            .select(1, self.restore_from.clone()) // Select all variables in output.
            .bool_and(
                // bool_and broadcasts lhs and rhs such that the shapes match.
                self.restores // Mask over to get only output vars that have a matching input.
                    .clone()
                    .reshape([1, vars_out])
                    .expand([batch * grid, vars_out]),
            );

        let mask = self.fill.clone().reshape([1, 1, 1, vars_in]).expand(shape);

        // Return two things:
        // 1. Only get the nan-values that matter based on self.fill, which holds the input variables
        //    that ConstantImputer applies to. Fill these to 0 over x.
        // 2. Return the inversion helper, which tracks the nan-values and maps it to the correct
        //    output variables.
        (x.mask_fill(nan.bool_and(mask), 0.0), imputed)
    }
}

// ConditionalNaNPostprocessor sets NaN from some reference variable's NaN values to some target
// variables. This only applies on post-processing.
//
// It runs last because the imputer's inverse is what puts NaN into the reference in the first
// place.
#[derive(Debug)]
struct ConditionalNan<B: Backend> {
    // [vars_out]; the reference channel repeated, so one gather widens it to every column.
    reference: Tensor<B, 1, Int>,
    // [vars_out]; true on the channels it blanks.
    targets: Tensor<B, 1, Bool>,
}

impl<B: Backend> ConditionalNan<B> {
    fn new(metadata: &Metadata, device: &B::Device) -> Result<ConditionalNan<B>, Error> {
        let vars_out = metadata.model_output.full.len();

        // The reference MUST exist if we are using ConditionalNaN. If not, there is an issue with
        // the metadata.
        let reference = metadata
            .output_channel(&metadata.nan_postprocessor_reference)
            .ok_or(Error::Metadata(format!(
                "nan postprocessor remaps on {:?}, which has no output channel",
                metadata.nan_postprocessor_reference
            )))?;

        // Setup bool mask on which targets should have ConditionalNaN from reference.
        let mut targets = vec![false; vars_out];
        for name in &metadata.nan_postprocessor_vars {
            if let Some(channel) = metadata.output_channel(name) {
                targets[channel] = true;
            }
        }

        Ok(ConditionalNan {
            reference: indices(vec![reference as i64; vars_out], device),
            targets: bools(targets, device),
        })
    }

    fn inverse(&self, y: Tensor<B, 2>) -> Tensor<B, 2> {
        let [rows, vars_out] = y.shape().dims();

        let mask = y
            .clone()
            .select(1, self.reference.clone()) // Get all rows for self.reference variable.
            .is_nan() // Get nan.
            .bool_and(
                // Mask over nan for variables we care about.
                // Reshape to 2D by dup'ing across all rows.
                self.targets
                    .clone()
                    .reshape([1, vars_out])
                    .expand([rows, vars_out]),
            );

        y.mask_fill(mask, f32::NAN)
    }
}

fn indices<B: Backend>(values: Vec<i64>, device: &B::Device) -> Tensor<B, 1, Int> {
    let len = values.len();
    Tensor::from_data(TensorData::new(values, [len]), device)
}

fn bools<B: Backend>(values: Vec<bool>, device: &B::Device) -> Tensor<B, 1, Bool> {
    let len = values.len();
    Tensor::from_data(TensorData::new(values, [len]), device)
}

fn floats(store: &mut SafetensorsStore, name: &str) -> Result<Vec<f32>, Error> {
    snapshot(store, name)
        .map_err(|e| Error::Graph(e))?
        .to_vec::<f32>()
        .map_err(|e| Error::TensorType(format!("{name} is not f32: {e:?}")))
}

fn ints(store: &mut SafetensorsStore, name: &str) -> Result<Vec<i32>, Error> {
    snapshot(store, name)
        .map_err(|e| Error::Graph(e))?
        .to_vec::<i32>()
        .map_err(|e| Error::TensorType(format!("{name} is not i32: {e:?}")))
}

#[derive(Debug)]
pub enum Error {
    Metadata(String),
    TensorType(String),
    Graph(crate::graph::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Metadata(reason) => write!(f, "metadata issue: {}", reason),
            Error::TensorType(reason) => write!(f, "tensor type issue: {}", reason),
            Error::Graph(reason) => write!(f, "graph issue: {}", reason),
        }
    }
}

impl error::Error for Error {}

#[cfg(test)]
#[path = "processors_test.rs"]
mod tests;
