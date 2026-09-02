//! The forecast as GRIB2, one message per output variable.
//!
//! This is anemoi's `outputs/grib.py` reduced to what one checkpoint needs. ecCodes cannot build
//! a message from nothing, so each field is a clone of a template message on the model grid with
//! its identity, time and values overwritten (`grib/encoding.py:207-341`). The templates are
//! anemoi's own built-in N320 samples; see scripts/extract_grib_templates.py.
//!
//! The two post-processing steps between the tensor and the file are the ones
//! data/inference.yaml lists for this checkpoint: `cos_mwd`/`sin_mwd` fold back into one `mwd`
//! field (`backward-transform-filter: cos_sin_mean_wave_direction`), and the accumulated
//! variables are clamped at zero (`accumulate_from_start_of_forecast`, `accumulate.py:83-84`).

use std::error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use burn::prelude::*;
use chrono::{DateTime, Datelike, Timelike, Utc};
use eccodes::{BufMessage, CodesError, CodesFile, FallibleIterator, KeyWrite, ProductKind};

use crate::grib::CODES_MUTEX;
use crate::metadata::{Mars, Metadata};

#[derive(Debug)]
pub enum Error {
    Template(PathBuf, io::Error),
    Io(PathBuf, io::Error),
    // The tensor could not be read back to the host.
    Data(String),
    Shape { expected: usize, found: usize },
    // An output channel whose variable has no MARS block, so no GRIB identity.
    MissingMars(String),
    Codes { field: String, error: CodesError },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Template(path, e) => write!(f, "reading template {}: {e}", path.display()),
            Error::Io(path, e) => write!(f, "writing {}: {e}", path.display()),
            Error::Data(message) => write!(f, "reading the output tensor: {message}"),
            Error::Shape { expected, found } => {
                write!(f, "output tensor has {found} values, expected {expected}")
            }
            Error::MissingMars(name) => write!(f, "{name} has no mars block in the metadata"),
            Error::Codes { field, error } => write!(f, "encoding {field}: {error}"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::Template(_, e) | Error::Io(_, e) => Some(e),
            Error::Codes { error, .. } => Some(error),
            Error::Data(_) | Error::Shape { .. } | Error::MissingMars(_) => None,
        }
    }
}

/// The two template messages, held as raw GRIB bytes: one for pressure levels, one for
/// everything else. ecCodes re-parses them per field, which at 1.5 KB each is nothing.
pub struct Templates {
    sfc: Vec<u8>,
    pl: Vec<u8>,
}

impl Templates {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let read = |name: &str| {
            let path = dir.join(name);
            fs::read(&path).map_err(|e| Error::Template(path, e))
        };
        Ok(Self {
            sfc: read("n320-sfc.grib2")?,
            pl: read("n320-pl.grib2")?,
        })
    }

    fn for_levtype(&self, levtype: &str) -> &[u8] {
        if levtype == "pl" { &self.pl } else { &self.sfc }
    }
}

/// The keys inference.yaml's `output.grib.encoding` stamps on every message.
pub struct Encoding {
    pub class: &'static str,
    pub data_type: &'static str,
    pub generating_process_identifier: i64,
}

pub const AIFS_SINGLE: Encoding = Encoding {
    class: "ai",
    data_type: "fc",
    generating_process_identifier: 5,
};

// A missing point is stored as this value under a bitmap. anemoi's default; the value is
// arbitrary so long as no real field reaches it.
const MISSING_VALUE: f64 = -9999.0;

const WAVE_DIRECTION: &str = "mwd";

// One message's worth of data: the variable, its GRIB identity, and the values on the grid.
#[derive(Debug)]
pub(crate) struct Field {
    pub name: String,
    pub mars: Mars,
    pub accumulated: bool,
    pub values: Vec<f64>,
}

/// Write one forecast step. `y` is `predict_step`'s `[grid, num_output_channels]` in physical
/// units; `reference` is the forecast base time and `lead` the offset of this step from it.
/// Returns the number of messages written.
pub fn write_step<B: Backend>(
    path: &Path,
    y: Tensor<B, 2>,
    metadata: &Metadata,
    templates: &Templates,
    reference: DateTime<Utc>,
    lead: Duration,
    encoding: &Encoding,
) -> Result<usize, Error> {
    let host = y
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| Error::Data(format!("{e:?}")))?;
    write_host(path, &host, metadata, templates, reference, lead, encoding)
}

/// `write_step` after the device read-back: `host` is row-major `[grid, num_output_channels]`.
pub fn write_host(
    path: &Path,
    host: &[f32],
    metadata: &Metadata,
    templates: &Templates,
    reference: DateTime<Utc>,
    lead: Duration,
    encoding: &Encoding,
) -> Result<usize, Error> {
    let fields = fields(host, metadata)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Io(parent.to_path_buf(), e))?;
    }
    // write_to_file opens without truncating, so a shorter run over an existing file would leave
    // stale messages at the end. Start from an empty file and append every message.
    fs::File::create(path).map_err(|e| Error::Io(path.to_path_buf(), e))?;

    // ecCodes has no internal locking, see the comments around CODES_MUTEX.
    let _serialised = CODES_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    for field in &fields {
        let codes = |error| Error::Codes {
            field: field.name.clone(),
            error,
        };
        let message = encode(field, templates, reference, lead, encoding).map_err(codes)?;
        message.write_to_file(path, true).map_err(codes)?;
    }
    Ok(fields.len())
}

// Split the tensor into named columns and apply the two output post-processors.
pub(crate) fn fields(host: &[f32], metadata: &Metadata) -> Result<Vec<Field>, Error> {
    let names = &metadata.output_channel_to_var;
    let vars = names.len();
    let grid = metadata.latitudes.len();
    if host.len() != grid * vars {
        return Err(Error::Shape {
            expected: grid * vars,
            found: host.len(),
        });
    }
    let column = |c: usize| -> Vec<f64> { (0..grid).map(|i| host[i * vars + c] as f64).collect() };
    let mars = |name: &str| {
        metadata
            .mars
            .get(name)
            .cloned()
            .ok_or_else(|| Error::MissingMars(name.to_string()))
    };

    let mut fields = Vec::with_capacity(vars);
    for (c, name) in names.iter().enumerate() {
        // Emitted once, in the slot of whichever half comes first.
        if name == "cos_mwd" || name == "sin_mwd" {
            if fields.iter().any(|f: &Field| f.name == WAVE_DIRECTION) {
                continue;
            }
            let channel = |name: &str| metadata.output_channel(name);
            let (cos, sin) = match (channel("cos_mwd"), channel("sin_mwd")) {
                (Ok(cos), Ok(sin)) => (cos, sin),
                (Err(name), _) | (_, Err(name)) => return Err(Error::MissingMars(name)),
            };
            fields.push(Field {
                name: WAVE_DIRECTION.to_string(),
                mars: mars(WAVE_DIRECTION)?,
                accumulated: false,
                values: wave_direction(&column(cos), &column(sin)),
            });
            continue;
        }

        let accumulated = metadata.accumulations.contains(name);
        let mut values = column(c);
        if accumulated {
            clamp_accumulation(&mut values);
        }
        fields.push(Field {
            name: name.clone(),
            mars: mars(name)?,
            accumulated,
            values,
        });
    }
    Ok(fields)
}

// The inverse of the split in grib.rs: degrees in [0, 360), NaN wherever either half is.
pub(crate) fn wave_direction(cos: &[f64], sin: &[f64]) -> Vec<f64> {
    cos.iter()
        .zip(sin)
        .map(|(&c, &s)| {
            let degrees = s.atan2(c).to_degrees();
            if degrees < 0.0 {
                degrees + 360.0
            } else {
                degrees
            }
        })
        .collect()
}

// An accumulation cannot be negative; the model's residual can be. NaN is kept, which is why
// this is not f64::max -- that returns the other operand for NaN.
pub(crate) fn clamp_accumulation(values: &mut [f64]) {
    for v in values {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

// Clone the template and set the keys. The order follows anemoi's ORDERING (`encoding.py:38`):
// ecCodes re-derives dependent keys as each one lands, so the level type and step type have to
// be in place before the level and step values that depend on them, and `values` goes last.
fn encode(
    field: &Field,
    templates: &Templates,
    reference: DateTime<Utc>,
    lead: Duration,
    encoding: &Encoding,
) -> Result<BufMessage, CodesError> {
    let template = templates.for_levtype(&field.mars.levtype).to_vec();
    let mut file = CodesFile::new_from_memory(template, ProductKind::GRIB)?;
    let mut message = file
        .ref_message_iter()
        .next()?
        .ok_or_else(|| {
            CodesError::FileHandlingInterrupted(io::Error::new(
                io::ErrorKind::InvalidData,
                "template holds no message",
            ))
        })?
        .try_clone()?;

    if field.mars.levtype == "pl" {
        message.write_key_unchecked("typeOfLevel", "isobaricInhPa")?;
    }
    let lead_hours = (lead.as_secs() / 3600) as i64;
    if field.accumulated {
        message.write_key_unchecked("stepType", "accum")?;
        message.write_key_unchecked("productDefinitionTemplateNumber", 8)?;
    } else {
        message.write_key_unchecked("stepType", "instant")?;
    }

    message.write_key_unchecked("class", encoding.class)?;
    message.write_key_unchecked("dataType", encoding.data_type)?;
    message.write_key_unchecked(
        "generatingProcessIdentifier",
        encoding.generating_process_identifier,
    )?;
    message.write_key_unchecked("shortName", field.mars.param.as_str())?;
    if let Some(stream) = &field.mars.stream {
        message.write_key_unchecked("stream", stream.as_str())?;
    }

    let (year, month, day) = (reference.year(), reference.month(), reference.day());
    let date = year as i64 * 10000 + month as i64 * 100 + day as i64;
    let time = reference.hour() as i64 * 100 + reference.minute() as i64;
    message.write_key_unchecked("date", date)?;
    message.write_key_unchecked("time", time)?;
    if field.accumulated {
        message.write_key_unchecked("startStep", 0)?;
        message.write_key_unchecked("endStep", lead_hours)?;
    } else {
        message.write_key_unchecked("step", lead_hours)?;
    }

    if let Some(level) = field.mars.levelist {
        message.write_key_unchecked("level", level)?;
    }

    if field.values.iter().any(|v| v.is_nan()) {
        message.write_key_unchecked("missingValue", MISSING_VALUE)?;
        message.write_key_unchecked("bitmapPresent", 1)?;
        let masked: Vec<f64> = field
            .values
            .iter()
            .map(|&v| if v.is_nan() { MISSING_VALUE } else { v })
            .collect();
        message.write_key_unchecked("values", masked.as_slice())?;
    } else {
        message.write_key_unchecked("values", field.values.as_slice())?;
    }

    Ok(message)
}

#[cfg(test)]
#[path = "output_test.rs"]
mod tests;
