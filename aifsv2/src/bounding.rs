//! The output boundings -- anemoi's `config.model.bounding`, applied in list order at the end of
//! `assemble_output`.
//!
//! These compute the bounded value over every channel and use a `[vars]` mask to choose which
//! columns keep it.
//!
//! All of this runs in *normalised* space, because anemoi applies the boundings inside `forward`
//! and only then hands off to `post_processors`.

use crate::metadata::{self, BoundingConfig, ChannelKind, Metadata};
use burn::{
    prelude::*,
    tensor::{Bool, activation::relu},
};

// Broadcast a [vars] channel mask over the rows of a [rows, vars] tensor.
fn over_rows<B: Backend>(mask: Tensor<B, 1, Bool>, rows: usize) -> Tensor<B, 2, Bool> {
    let vars = mask.shape().dims::<1>()[0];

    // Expand mask via duplication.
    mask.reshape([1, vars]).expand([rows, vars])
}

#[derive(Module, Debug)]
pub struct ReluBounding<B: Backend> {
    mask: Tensor<B, 1, Bool>, // [vars]; true on the channels this bounds.
}

impl<B: Backend> ReluBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        vars: &[String],
        kind: &ChannelKind,
        device: &B::Device,
    ) -> Result<ReluBounding<B>, metadata::Error> {
        Ok(ReluBounding {
            mask: metadata.mask_channels_of_vec(vars, kind, device)?,
        })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let rows = x.shape().dims::<2>()[0];
        let mask = over_rows(self.mask.clone(), rows);
        x.clone().mask_where(mask, relu(x))
    }
}

#[derive(Module, Debug)]
pub struct HardtanhBounding<B: Backend> {
    mask: Tensor<B, 1, Bool>, // [vars]; true on the channels this bounds.
    min: f64,
    max: f64,
}

impl<B: Backend> HardtanhBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        min: f64,
        max: f64,
        vars: &[String],
        kind: &ChannelKind,
        device: &B::Device,
    ) -> Result<HardtanhBounding<B>, metadata::Error> {
        Ok(HardtanhBounding {
            mask: metadata.mask_channels_of_vec(vars, kind, device)?,
            min,
            max,
        })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let rows = x.shape().dims::<2>()[0];
        let mask = over_rows(self.mask.clone(), rows);
        // Hardtanh is the piecewise function
        // (----------------------------)
        // |   max_val if x > max_val   |
        // |   min_val if x < min_val   |
        // |   x       else             |
        // (----------------------------)
        let h = x.clone().clamp_min(self.min).clamp_max(self.max);
        x.mask_where(mask, h)
    }
}

#[derive(Module, Debug)]
pub struct FractionBounding<B: Backend> {
    // The channel the fraction is taken of, as a plain index. A one-element `Tensor<B, 1, Int>`
    // fed to `select` is the obvious spelling and it is wrong here: held as a field and used
    // across several forwards, it makes `select` return channel 0 instead of the one it names, on
    // wgpu, silently. A freshly built index tensor on the same input returns the right column, so
    // it is the stored one that goes bad. `slice_dim` needs no index tensor at all and was
    // verified against a host-side read of the same column.
    total_var: usize,
    mask: Tensor<B, 1, Bool>, // [vars]; true on the channels this bounds.
    min: f64,
    max: f64,
}

impl<B: Backend> FractionBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        min: f64,
        max: f64,
        total_var: &str,
        vars: &[String],
        kind: &ChannelKind,
        device: &B::Device,
    ) -> Result<FractionBounding<B>, metadata::Error> {
        let total_var = match kind {
            ChannelKind::Input => metadata.input_channel(total_var),
            ChannelKind::Output => metadata.output_channel(total_var),
        }
        .unwrap_or_else(|var| {
            panic!("FractionBounding total_var ({total_var:?}, {var:?}) has no channel")
        });

        Ok(FractionBounding {
            total_var,
            mask: metadata.mask_channels_of_vec(vars, kind, device)?,
            min,
            max,
        })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let rows = x.shape().dims::<2>()[0];
        let mask = over_rows(self.mask.clone(), rows);

        // Get the total_var
        let total = x.clone().slice_dim(1, self.total_var..self.total_var + 1);
        // Scale total variable by the normalised values.
        let f = x.clone().clamp_min(self.min).clamp_max(self.max) * total;

        x.mask_where(mask, f)
    }
}

#[derive(Module, Debug)]
pub enum BoundingType<B: Backend> {
    Relu(ReluBounding<B>),
    Hardtanh(HardtanhBounding<B>),
    Fraction(FractionBounding<B>),
}

impl<B: Backend> BoundingType<B> {
    pub fn from_bounding_config(
        metadata: &Metadata,
        conf: &BoundingConfig,
        kind: &ChannelKind,
        device: &B::Device,
    ) -> Result<BoundingType<B>, metadata::Error> {
        match conf {
            BoundingConfig::Relu { variables } => Ok(BoundingType::Relu(ReluBounding::new_init(
                metadata, variables, kind, device,
            )?)),
            BoundingConfig::Hardtanh {
                variables,
                min_val,
                max_val,
            } => Ok(BoundingType::Hardtanh(HardtanhBounding::new_init(
                metadata, *min_val, *max_val, variables, kind, device,
            )?)),
            BoundingConfig::Fraction {
                variables,
                min_val,
                max_val,
                total_var,
            } => {
                // The forward reads the total column from the tensor it is about to write. If the
                // total were one of the bounded variables the two would race and the port would
                // silently disagree with anemoi, which writes first and reads after.
                assert!(
                    !variables.contains(total_var),
                    "FractionBounding total_var {total_var:?} is also one of its own variables",
                );
                Ok(BoundingType::Fraction(FractionBounding::new_init(
                    metadata, *min_val, *max_val, total_var, variables, kind, device,
                )?))
            }
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            BoundingType::Relu(r) => r.forward(x),
            BoundingType::Hardtanh(h) => h.forward(x),
            BoundingType::Fraction(f) => f.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct Bounding<B: Backend> {
    boundings: Vec<BoundingType<B>>,
}

impl<B: Backend> Bounding<B> {
    pub fn new(boundings: Vec<BoundingType<B>>) -> Bounding<B> {
        Bounding { boundings }
    }

    /// Applies each bounding in turn. The order is the order of `config.model.bounding` and is
    /// load-bearing: `FractionBounding` reads a variable the earlier entries have already clamped.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut a = x;
        for b in self.boundings.iter() {
            a = b.forward(a);
        }
        a
    }
}

#[cfg(test)]
#[path = "bounding_test.rs"]
mod tests;
