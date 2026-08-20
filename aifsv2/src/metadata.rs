use std::collections::HashMap;
use std::error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use serde::Serialize;

#[derive(Debug)]
pub enum Error {
    Io(PathBuf, io::Error),
    Json(PathBuf, serde_json::Error),
    // The checkpoint parsed but does not describe a usable model.
    Malformed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(path, e) => write!(f, "reading {}: {e}", path.display()),
            Error::Json(path, e) => write!(f, "parsing {}: {e}", path.display()),
            Error::Malformed(message) => f.write_str(message),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::Io(_, e) => Some(e),
            Error::Json(_, e) => Some(e),
            Error::Malformed(_) => None,
        }
    }
}

// Role partitions for one tensor. `full` is not always the union of the other three:
// `Metadata::data_input` lists diagnostics that the input tensor does not carry.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct IndexSet {
    pub full: Vec<usize>,
    // Channel indices for the prognostic residual. input_prognostic indexes the
    // num_input_channels space and output_prognostic the num_output_channels space; these are
    // different index spaces (input starts [0, 1, 2, ..], output starts [2, 3, 4, ..]) naming the
    // same variables, so they must have the same length.
    pub prognostic: Vec<usize>,
    pub diagnostic: Vec<usize>,
    pub forcing: Vec<usize>,
}

// Output clamps, applied in order after the residual. Fraction divides by a variable the
// earlier entries have already clamped, so reordering these is silently wrong.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "_target_")]
pub enum Bounding {
    #[serde(rename = "anemoi.models.layers.bounding.ReluBounding")]
    Relu { variables: Vec<String> },
    #[serde(rename = "anemoi.models.layers.bounding.HardtanhBounding")]
    Hardtanh {
        variables: Vec<String>,
        min_val: f64,
        max_val: f64,
    },
    // Bounds `variables` to [min_val, max_val] * total_var.
    #[serde(rename = "anemoi.models.layers.bounding.FractionBounding")]
    Fraction {
        variables: Vec<String>,
        min_val: f64,
        max_val: f64,
        total_var: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Metadata {
    pub variables: Vec<String>, // The total possible variable space. The actual variables on inference and once infered vary.
    pub multistep: usize,       // Timesteps stacked into one input tensor.
    pub timestep: Duration,     // Spacing between them, and the forecast step.

    // The same two tensors described twice over: `data_*` numbers each variable by its index
    // in `variables`, `model_*` by its channel in the tensor itself. The residual maps
    // `model_input.prognostic` onto `model_output.prognostic`, which is why both exist.
    //
    // These are different indices as the input variables are stored lexicographically in the dataset,
    // and then some are removed. For example the input tensor does not need diagnostics as they are
    // never passed in. The output tensor does not need forcings are they are not inferred.
    //
    // Example
    //   dataset      0=100u  1=100v  2=10u  3=10v  4=2d  5=2t  ...
    //   input ch       —       —       0      1     2     3    ...   (100u/100v deleted)
    //   output ch      0       1       2      3     4     5    ...   (both kept)
    //
    pub data_input: IndexSet, // The index into the dataset on input, location.
    pub data_output: IndexSet, // The index into the dataset on output, location.
    pub model_input: IndexSet, // The index into the model's tensor's channels at input, location.
    pub model_output: IndexSet, // The index into the model's tensor's channels at output, location.

    pub var_to_input_channel: HashMap<String, usize>, // Variable name -> input channel.
    pub var_to_output_channel: HashMap<String, usize>, // Variable name -> output channel.
    pub output_channel_to_var: Vec<String>,           // Output channel -> variable name.

    pub computed_forcing: Vec<String>, // Derived from date and grid position, not retrieved.
    pub constant_in_time: Vec<String>, // Never recomputed; overlaps `computed_forcing`.
    pub imputer_zero: Vec<String>,     // Filled with 0 wherever the source data is NaN.
    pub boundings: Vec<Bounding>,      // Output clamps, in application order.

    pub nan_postprocessor_reference: String, // Reference variable for NaN masking
    pub nan_postprocessor_vars: Vec<String>, // Variables to be masked

    // Not for graph construction: the edges were built from the f32 sin/cos coordinates.
    pub latitudes: Vec<f64>,  // Degrees, one per grid point.
    pub longitudes: Vec<f64>, // Degrees, 0..360.
}

pub enum ChannelKind {
    Input,
    Output,
}

impl Metadata {
    // Get the index of a variable name in the list of variables in metadata.
    // You can use this to index into the input tensor.
    pub fn input_channel(&self, name: &str) -> Option<usize> {
        self.var_to_input_channel.get(name).copied()
    }

    pub fn output_channel(&self, name: &str) -> Option<usize> {
        self.var_to_output_channel.get(name).copied()
    }

    // Filters names that do not work.
    pub fn channels_of_vec(&self, channel_names: &Vec<String>, kind: ChannelKind) -> Vec<i64> {
        channel_names
            .iter()
            .filter_map(|var_name| match kind {
                ChannelKind::Input => self.input_channel(var_name),
                ChannelKind::Output => self.output_channel(var_name),
            })
            .map(|x| x as i64)
            .collect()
    }

    pub fn load(anemoi_metadata_dir: &Path) -> Result<Metadata, Error> {
        let path = anemoi_metadata_dir.join("ai-models.json");
        let file = fs::File::open(&path).map_err(|e| Error::Io(path.clone(), e))?;
        let raw: Raw = serde_json::from_reader(file).map_err(|e| Error::Json(path, e))?;

        let variables = raw.dataset.variables;
        let flagged = |flag: fn(&VarMeta) -> bool| -> Vec<String> {
            variables
                .iter()
                .filter(|name| raw.dataset.variables_metadata.get(*name).is_some_and(flag))
                .cloned()
                .collect()
        };

        let var_to_output_channel = channel_map(
            &variables,
            &raw.data_indices.data.output.full,
            &raw.data_indices.model.output.full,
        )?;
        let mut output_channel_to_var = vec![String::new(); var_to_output_channel.len()];
        for (name, &channel) in &var_to_output_channel {
            let slot = output_channel_to_var.get_mut(channel).ok_or_else(|| {
                Error::Malformed(format!(
                    "output channel {channel} for {name:?} is out of range"
                ))
            })?;
            *slot = name.clone();
        }

        // Parse the numpy files in the checkpoint.
        let latitudes = read_f64_array(&anemoi_metadata_dir.join("latitudes.numpy"))?;
        let longitudes = read_f64_array(&anemoi_metadata_dir.join("longitudes.numpy"))?;
        if latitudes.len() != longitudes.len() {
            return Err(Error::Malformed(format!(
                "{} latitudes against {} longitudes",
                latitudes.len(),
                longitudes.len()
            )));
        }

        let conditional_nan_postprocessor_config = raw
            .config
            .data
            .processors
            .conditional_nan_postprocessor
            .config;
        let nan_postprocessor_reference = conditional_nan_postprocessor_config.remap;
        let nan_postprocessor_vars = conditional_nan_postprocessor_config.nan;

        Ok(Metadata {
            var_to_input_channel: channel_map(
                &variables,
                &raw.data_indices.data.input.full,
                &raw.data_indices.model.input.full,
            )?,
            var_to_output_channel,
            output_channel_to_var,
            computed_forcing: flagged(|v| v.computed_forcing),
            constant_in_time: flagged(|v| v.constant_in_time),
            variables,
            multistep: raw.config.training.multistep_input,
            timestep: parse_timestep(&raw.config.data.timestep)?,
            data_input: raw.data_indices.data.input,
            data_output: raw.data_indices.data.output,
            model_input: raw.data_indices.model.input,
            model_output: raw.data_indices.model.output,
            imputer_zero: raw.config.data.processors.const_imputer.config.zero,
            boundings: raw.config.model.bounding,
            latitudes,
            longitudes,
            nan_postprocessor_reference,
            nan_postprocessor_vars,
        })
    }
}

// Only the keys we read; serde ignores the rest.
#[derive(Deserialize)]
struct Raw {
    dataset: RawDataset,
    data_indices: RawIndices,
    config: RawConfig,
}

#[derive(Deserialize)]
struct RawDataset {
    variables: Vec<String>,
    variables_metadata: HashMap<String, VarMeta>,
}

#[derive(Deserialize)]
struct VarMeta {
    #[serde(default)]
    computed_forcing: bool,
    #[serde(default)]
    constant_in_time: bool,
}

#[derive(Deserialize)]
struct RawIndices {
    data: RawSpace,
    model: RawSpace,
}

#[derive(Deserialize)]
struct RawSpace {
    input: IndexSet,
    output: IndexSet,
}

#[derive(Deserialize)]
struct RawConfig {
    data: RawData,
    model: RawModel,
    training: RawTraining,
}

#[derive(Deserialize)]
struct RawData {
    timestep: String,
    processors: RawProcessors,
}

#[derive(Deserialize)]
struct RawProcessors {
    const_imputer: RawImputer,
    conditional_nan_postprocessor: RawConditionalNanPostProcessor,
}

#[derive(Deserialize)]
struct RawConditionalNanPostProcessor {
    config: RawConditionalNanPostProcessorConfig,
}

#[derive(Deserialize)]
struct RawImputer {
    config: RawImputerConfig,
}

// Keys are fill values. Only 0 is implemented, so reject the rest rather than drop them and
// leave those variables unimputed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImputerConfig {
    #[serde(rename = "0")]
    zero: Vec<String>,
    // A fallback policy rather than a variable list. Named only to keep it off the reject list.
    #[allow(dead_code)]
    default: String,
}
//
// Keys are fill values. Only 0 is implemented, so reject the rest rather than drop them and
// leave those variables unimputed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConditionalNanPostProcessorConfig {
    nan: Vec<String>,
    remap: String,
}

#[derive(Deserialize)]
struct RawModel {
    bounding: Vec<Bounding>,
}

#[derive(Deserialize)]
struct RawTraining {
    multistep_input: usize,
}

// `data_full` and `model_full` are parallel: entry i describes one variable, as its index in
// `variables` and as its channel in the model tensor. Nothing else states that pairing.
//
// See not above in [[struct Metadata]] for more information.
fn channel_map(
    variables: &[String],
    data_full: &[usize],
    model_full: &[usize],
) -> Result<HashMap<String, usize>, Error> {
    if data_full.len() != model_full.len() {
        return Err(Error::Malformed(format!(
            "index spaces disagree on width: {} data against {} model",
            data_full.len(),
            model_full.len()
        )));
    }
    data_full
        .iter()
        .zip(model_full)
        .map(|(&data, &model)| {
            let name = variables.get(data).ok_or_else(|| {
                Error::Malformed(format!("variable index {data} is out of range"))
            })?;
            Ok((name.clone(), model))
        })
        .collect()
}

// Frequency strings, e.g. "6h".
fn parse_timestep(text: &str) -> Result<Duration, Error> {
    let bad = || Error::Malformed(format!("unparsable timestep {text:?}"));
    let split = text.find(|c: char| !c.is_ascii_digit()).ok_or_else(bad)?;
    let (count, unit) = text.split_at(split);
    let seconds = match unit {
        "s" => 1,
        "m" => 60,
        "h" => 60 * 60,
        "d" => 24 * 60 * 60,
        _ => return Err(bad()),
    };
    let count = count.parse::<u64>().map_err(|_| bad())?;
    Ok(Duration::from_secs(count * seconds))
}

// The .numpy extension is a misnomer: raw little-endian f64, no header.
fn read_f64_array(path: &Path) -> Result<Vec<f64>, Error> {
    let bytes = fs::read(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;
    if bytes.len() % size_of::<f64>() != 0 {
        return Err(Error::Malformed(format!(
            "{} is not a whole number of f64",
            path.display()
        )));
    }

    let mut values = Vec::with_capacity(bytes.len() / size_of::<f64>());
    for chunk in bytes.chunks_exact(size_of::<f64>()) {
        let mut word = [0u8; size_of::<f64>()];
        word.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(word));
    }
    Ok(values)
}
