use std::{
    error, fmt,
    path::{Path, PathBuf},
    sync::Mutex,
};

use burn::prelude::*;
use burn_store::SafetensorsStore;
use chrono::{DateTime, NaiveDate, TimeDelta, Utc};
use eccodes::{CodesError, CodesFile, FallibleIterator, KeyRead, ProductKind};

use crate::{forcings, graph::snapshot, metadata::Metadata};

#[derive(Debug)]
pub enum Error {
    Codes(PathBuf, CodesError),
    // The regrid matrix is missing, truncated, or does not describe these two grids.
    Matrix(PathBuf, String),
    // One file per timestep is the contract; anything else cannot fill the time axis.
    Multistep {
        expected: usize,
        oper: usize,
        wave: usize,
    },
    Grid {
        path: PathBuf,
        expected: usize,
        found: usize,
    },
    // Channels no message wrote and no imputer fills. Porting anemoi's `tensors.py:168-172`.
    MissingChannels(Vec<String>),
    Time(String),
    Forcings(forcings::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Codes(path, e) => write!(f, "reading {}: {e}", path.display()),
            Error::Matrix(path, reason) => write!(f, "regrid matrix {}: {reason}", path.display()),
            Error::Multistep {
                expected,
                oper,
                wave,
            } => write!(
                f,
                "multistep is {expected}, so {expected} oper and {expected} wave files are needed, \
                 got {oper} and {wave}"
            ),
            Error::Grid {
                path,
                expected,
                found,
            } => write!(
                f,
                "{} is on a {found}-point grid, expected {expected}",
                path.display()
            ),
            Error::MissingChannels(names) => {
                write!(
                    f,
                    "no data for {} input channels: {}",
                    names.len(),
                    names.join(", ")
                )
            }
            Error::Time(reason) => f.write_str(reason),
            Error::Forcings(e) => write!(f, "forcings: {e}"),
        }
    }
}

impl error::Error for Error {}

// The geometry a field needs to be interpolated onto the model grid.
//
// Never coordinates: for the model grid the checkpoint ships latitudes/longitudes, and for the
// open-data grid the regrid matrix already encodes the layout. This carries only the handful of
// section-3 keys that say which column of that matrix a stored value belongs to.
#[derive(Debug, Clone)]
pub enum Grid {
    // 1440 x 721, the open-data grid.
    RegularLatLon {
        ni: usize,
        nj: usize,
        lon_first: f64, // First longitude shift.
        di: f64,        // Precision degree (i.e., 0.25)
    },
    // Anything else -- N320 in practice, where values are already in the model's own order.
    Stored,
}

// One GRIB message: the keys that identify the field, and the field itself.
//
// Routing is by (short_name, level_type, level) rather than param_id, because the model's own
// variable names are built from the same three -- see `variable_names`. param_id is carried for
// logging and for the rename table's comments.
#[derive(Debug, Clone)]
pub struct Field {
    pub param_id: i64,
    pub short_name: String,
    // ecCodes vocabulary, i.e., isobaricInhPa, surface, soilLayer, heightAboveGround and not the
    // MARS one (pl, sfc, sol, o2d).
    pub level_type: String,
    pub level: i64,
    // validityDate as yyyymmdd
    pub valid_date: i64,
    // validityTime as hhmm
    pub valid_time: i64,
    pub grid_type: String,
    pub grid: Grid,
    // Values in stored order, one per grid point, NaN wherever the message's bitmap masks a
    // point. Stored order means 542,080 values for the N320 reduced Gaussian grid, not the
    // 640x1280 = 819,200 that a reader which expands reduced grids would report.

    // Values in stored order, one per grid point. This can be NaN whenever masked over by message
    // bitmap.
    pub values: Vec<f32>,
}

// Host-side summary of a slice of values, NaN-aware.
//
// The point of computing this on the CPU is that it is the oracle: it never touches a backend
// reduction, so it is what `Tensor::min`/`max` get compared against. Those two return the
// reduction's identity (+/-f32::MAX) on wgpu when the tensor holds NaN, so a range printed
// through them cannot be trusted on any channel that is legitimately masked -- which is every
// wave field, every soil field, and `sd`. See docs/encoder-processor-decoder-review.md 4.1.
//
// `min`/`max`/`mean` are over the finite values only, and are NaN when there are none. That is
// deliberate: an all-masked channel should be visibly empty rather than silently reporting a
// sentinel that reads like data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub points: usize,
    pub nan: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
}

impl Stats {
    pub fn finite(&self) -> usize {
        self.points - self.nan
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pct = 100.0 * self.nan as f64 / self.points.max(1) as f64;
        if self.finite() == 0 {
            return write!(f, "{} points, all NaN", self.points);
        }
        write!(
            f,
            "[{}, {}] mean {:.6}, {} finite, {} NaN ({:.1}%)",
            self.min,
            self.max,
            self.mean,
            self.finite(),
            self.nan,
            pct
        )
    }
}

// Summarise values on the host. Iterates once; NaN never participates in the comparisons, so
// unlike the backend reductions a single masked point does not poison the result.
pub fn stats(values: &[f32]) -> Stats {
    let (mut nan, mut min, mut max, mut sum) = (0usize, f32::INFINITY, f32::NEG_INFINITY, 0f64);
    for &v in values {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    let finite = values.len() - nan;
    Stats {
        points: values.len(),
        nan,
        min: if finite == 0 { f32::NAN } else { min },
        max: if finite == 0 { f32::NAN } else { max },
        mean: if finite == 0 {
            f64::NAN
        } else {
            sum / finite as f64
        },
    }
}

// One channel of the assembled tensor, at one timestep.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub time: usize,
    pub channel: usize,
    pub stats: Stats,
}

impl Field {
    // Number of masked points -- ocean under a soil field, land under a wave field.
    pub fn missing(&self) -> usize {
        self.values.iter().filter(|v| v.is_nan()).count()
    }

    // The same accounting `scripts/parse_grib.py --missing` prints, computed from the values
    // this crate actually decoded. Comparing the two is what checks `unmask` and the ecCodes
    // read, independently of anything on the GPU.
    pub fn stats(&self) -> Stats {
        stats(&self.values)
    }

    // The message's own valid time. Anemoi calls this the field's date and keys the time axis
    // on it.
    pub fn valid(&self) -> Result<DateTime<Utc>, Error> {
        let (y, m, d) = (
            self.valid_date / 10_000,
            self.valid_date / 100 % 100,
            self.valid_date % 100,
        );
        let (hour, minute) = (self.valid_time / 100, self.valid_time % 100);
        NaiveDate::from_ymd_opt(y as i32, m as u32, d as u32)
            .and_then(|date| date.and_hms_opt(hour as u32, minute as u32, 0))
            .map(|naive| naive.and_utc())
            .ok_or_else(|| {
                Error::Time(format!(
                    "{} has an unreadable valid time {}T{:04}",
                    self.short_name, self.valid_date, self.valid_time
                ))
            })
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<8} {:>6}  {:>16}={:<6} {}T{:04}  {:<12} {} values",
            self.short_name,
            self.param_id,
            self.level_type,
            self.level,
            self.valid_date,
            self.valid_time,
            self.grid_type,
            self.values.len(),
        )
    }
}

pub fn load_grib<P: AsRef<Path>>(path: P) -> Result<Vec<Field>, Error> {
    let mut fields = Vec::new();
    for_each_field(path.as_ref(), |field| {
        fields.push(field);
        Ok(())
    })?;
    Ok(fields)
}

// TODO(saiputravu): Should think about recompiling this.
//
// ecCodes has no internal locking in this build -- nixpkgs compiles it with
// ENABLE_ECCODES_THREADS=OFF -- and its definitions parser is flex-generated global state. Two
// threads decoding at once therefore do not race quietly; they abort the process with "fatal
// flex scanner internal error" and a SIGSEGV. Every handle in this module is taken under this
// lock, which is what makes the module safe to call from anywhere. Cargo's test harness runs
// tests in parallel, which is how this was found.
static CODES_MUTEX: Mutex<()> = Mutex::new(());

// Decode a file one message at a time.
//
// The eager form costs 523 MB for the operational file alone (126 messages x 1,038,240 f32),
// which is more than the assembled tensor it feeds. Everything on the assembly path takes this
// form instead: decode, route, regrid, write, drop.
pub(crate) fn for_each_field(
    path: &Path,
    mut visit: impl FnMut(Field) -> Result<(), Error>,
) -> Result<(), Error> {
    // ecCodes has no internal locking and is compiled with threads off, see comments around CODES_MUTEX.
    let _serialised = CODES_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let codes = |e: CodesError| Error::Codes(path.to_path_buf(), e);
    let mut file = CodesFile::new_from_file(path, ProductKind::GRIB).map_err(codes)?;

    // A GRIB file has no index: reaching message n means walking the n-1 before it.
    while let Some(msg) = file.ref_message_iter().next().map_err(codes)? {
        // The type each key is read as has to be its ecCodes native type or the read fails.
        // missingValue is a *long* here despite naming a value in a double array.
        let missing: i64 = msg.read_key("missingValue").map_err(codes)?;
        let values: Vec<f64> = msg.read_key("values").map_err(codes)?;
        let grid_type: String = msg.read_key("gridType").map_err(codes)?;

        // Only regular_ll carries Ni/Nj; on a reduced grid they are the ecCodes MISSING
        // sentinel, so reading them at all is a mistake rather than a fallback.
        let grid = if grid_type == "regular_ll" {
            let ni: i64 = msg.read_key("Ni").map_err(codes)?;
            let nj: i64 = msg.read_key("Nj").map_err(codes)?;
            Grid::RegularLatLon {
                ni: ni as usize,
                nj: nj as usize,
                lon_first: msg
                    .read_key("longitudeOfFirstGridPointInDegrees")
                    .map_err(codes)?,
                di: msg
                    .read_key("iDirectionIncrementInDegrees")
                    .map_err(codes)?,
            }
        } else {
            Grid::Stored
        };

        visit(Field {
            param_id: msg.read_key("paramId").map_err(codes)?,
            short_name: msg.read_key("shortName").map_err(codes)?,
            level_type: msg.read_key("typeOfLevel").map_err(codes)?,
            level: msg.read_key("level").map_err(codes)?,
            valid_date: msg.read_key("validityDate").map_err(codes)?,
            valid_time: msg.read_key("validityTime").map_err(codes)?,
            grid_type,
            grid,
            // Apply the eccodes 9999.0 -> NaN map since eccodes represents as nan.
            values: unmask(values, missing as f64),
        })?;
    }

    Ok(())
}

// Rewrite ecCodes' masked points as NaN.
//
// ecCodes expands the bitmap for us but fills the masked points with missingValue -- 9999.0
// unless something sets it otherwise. The checkpoint's ConstantImputer only recognises NaN, so
// 9999 would reach the model as a real measurement of soil moisture or wave height. The
// substitution happens here at the boundary and nowhere else.
fn unmask(values: Vec<f64>, missing: f64) -> Vec<f32> {
    values
        .into_iter()
        .map(|v| if v == missing { f32::NAN } else { v as f32 })
        .collect()
}

// The interpolation from the open-data grid onto the model's own, as a sparse matrix. This is done
// via downloading a precompted sparse operator and mutliplying by the LL grid to interpolate.
// Earthkit does the same thing.
//
// CSR Representation of sparse matrix.
pub struct Regrid {
    indptr: Vec<i32>,  // Row based.
    indices: Vec<i32>, // Indicies of cols.
    values: Vec<f32>,  // Values.

    num_target: usize,
    num_source: usize,
}

impl Regrid {
    // Load the matrix into a sparse tensor representation.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Regrid, Error> {
        let path = path.as_ref();
        let bad = |reason: String| Error::Matrix(path.to_path_buf(), reason);

        let mut store = SafetensorsStore::from_file(path);
        let read = |store: &mut SafetensorsStore, name: &str| {
            snapshot(store, name).map_err(|e| bad(e.to_string()))
        };

        // Read out the sparse safetensor.
        let shape = read(&mut store, "shape")?
            .to_vec::<i32>()
            .map_err(|e| bad(format!("shape: {e:?}")))?;
        let [num_target, num_source] = shape[..] else {
            return Err(bad(format!(
                "shape has {} entries, expected 2",
                shape.len()
            )));
        };
        let (num_target, num_source) = (num_target as usize, num_source as usize);

        let indptr = read(&mut store, "indptr")?
            .to_vec::<i32>()
            .map_err(|e| bad(format!("indptr: {e:?}")))?;
        if indptr.len() != num_target + 1 {
            return Err(bad(format!(
                "indptr has {} entries for {num_target} target points",
                indptr.len()
            )));
        }

        let weights = read(&mut store, "weights")?
            .to_vec::<f32>()
            .map_err(|e| bad(format!("weights: {e:?}")))?;
        let indices = read(&mut store, "indices")?
            .to_vec::<i32>()
            .map_err(|e| bad(format!("indices: {e:?}")))?;
        if indices.len() != weights.len() {
            return Err(bad(format!(
                "{} indices against {} weights",
                indices.len(),
                weights.len()
            )));
        }

        Ok(Regrid {
            indptr,
            indices,
            values: weights,
            num_target,
            num_source,
        })
    }

    pub fn num_target(&self) -> usize {
        self.num_target
    }

    // Put one field on the model grid.
    //
    // A field that is already there -- lsm.grib is the only one -- passes through untouched.
    pub fn apply(&self, grid: &Grid, values: &[f32]) -> Result<Vec<f32>, Error> {
        let Grid::RegularLatLon {
            ni,
            nj,
            lon_first,
            di,
        } = *grid
        else {
            return if values.len() == self.num_target {
                Ok(values.to_vec())
            } else {
                Err(Error::Matrix(
                    PathBuf::new(),
                    format!(
                        "a {}-point field on an unmapped grid cannot reach the {}-point model grid",
                        values.len(),
                        self.num_target
                    ),
                ))
            };
        };

        // assert size check.
        if ni * nj != self.num_source || values.len() != self.num_source {
            return Err(Error::Matrix(
                PathBuf::new(),
                format!(
                    "matrix takes {} source points, field has {} ({ni} x {nj})",
                    self.num_source,
                    values.len()
                ),
            ));
        }

        // GRIB message has no coordinates, only a flat array. Position in that array is the coordinate
        // This is decoded from GRIB section-3 keys. If the files have
        // `longitudeOfFirstGridPointInDegrees = 180.0`, we rotate the column by 180 degrees.
        //
        // Example
        //   column     0      1    ...   719   720   721  ...  1439
        //   lon      180   180.25      359.75    0    0.25     179.75
        //            └──── eastern half ────┘    └─ western half ───┘
        //
        // So if shift is set, we do this rotation.
        let shift = (lon_first / di).round() as usize % ni;
        let source = if shift == 0 {
            values.to_vec()
        } else {
            let mut rolled = vec![0.0; values.len()];
            for row in 0..nj {
                let base = row * ni;
                for column in 0..ni {
                    rolled[base + column] = values[base + (column + ni - shift) % ni];
                }
            }
            rolled
        };

        // Compute the z@values where z is a sparse tensor. The NaN values propagate which acts
        // as a mask that propagates. These NaN fields are the zero-filled fields by the imputer.
        let out = vec![0.0; self.num_target]
            .iter()
            .enumerate()
            .map(|(pos, slot)| {
                // Note that indptr has N+1.
                //
                // This is a CSR sparse matrix mult against dense. From..To gives you the
                // number of elements on this row. We get the row index'd values via values[k] and
                // get the corresponding column via self.indices and index in to the other side
                // dense matrix. i.e. this is z@source.
                let from = self.indptr[pos] as usize;
                let to = self.indptr[pos + 1] as usize;
                let sum = (from..to).fold(*slot, |acc, k| {
                    acc + (self.values[k] * source[self.indices[k] as usize])
                });
                sum
            })
            .collect();

        Ok(out)
    }
}

// Soil levels are the one place the model's names are not derivable from the message.
// anemoi names fields from the ecCodes `mars` namespace, which gives `vsw_1` and `sot_1`
// where the checkpoint wants `swvl1` and `stl1`.
//
// paramIds: vsw 260199, sot 260360.
const SOIL_RENAMES: [(&str, i64, &str); 4] = [
    ("vsw", 1, "swvl1"),
    ("vsw", 2, "swvl2"),
    ("sot", 1, "stl1"),
    ("sot", 2, "stl2"),
];

// Mean wave direction (paramId 140230) is a transform, not a rename: one field becomes two
// channels. Averaging or interpolating degrees across the 0/360 discontinuity is
// meaningless, which is what the cos/sin pair exists to avoid.
const WAVE_DIRECTION: &str = "mwd";

// The model variable names a message carries, if any.
//
// Verified against ai-models.json and the inventory of both forecast files: this covers 96
// of the 97 retrieved input channels, with no duplicates and no false matches. The one it
// cannot cover is `sd`, which is in neither file.
pub fn variable_names(field: &Field) -> Vec<String> {
    if field.short_name == WAVE_DIRECTION {
        return vec!["cos_mwd".to_string(), "sin_mwd".to_string()];
    }
    if field.level_type == "soilLayer" {
        return SOIL_RENAMES
            .iter()
            .find(|(name, level, _)| *name == field.short_name && *level == field.level)
            .map(|(_, _, renamed)| vec![renamed.to_string()])
            .unwrap_or_default();
    }
    // Pressure levels are the only ones whose level is part of the name: `t_500`, `z_850`.
    // Surface z (orography, a forcing) and z_500 (a prognostic) separate here and nowhere else.
    if field.level_type == "isobaricInhPa" {
        return vec![format!("{}_{}", field.short_name, field.level)];
    }
    vec![field.short_name.clone()]
}

// Assemble the model's input tensor from the forecast files.
//
// Pass the oper/wave file combinations with older entries first.
// Returns `[batch, time, grid, vars]` = `[1, multistep, N320, 106 input vars]`. These are in
// physical units. These are routed, regridded and masked but are NOT imputed nor normalised. The
// preprocessors must still see these. You should call the preprocessor forward on this.
pub fn tensor_from<B: Backend>(
    oper_paths: &[&str],
    wave_paths: &[&str],
    lsm_path: &str,
    metadata: &Metadata,
    regrid: &Regrid,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Error> {
    let (t, _) = tensor_and_stats(oper_paths, wave_paths, lsm_path, metadata, regrid, device)?;
    Ok(t)
}

// Add the per-channel host-side stats taken off the buffer on its way the device.
pub fn tensor_and_stats<B: Backend>(
    oper_paths: &[&str],
    wave_paths: &[&str],
    lsm_path: &str,
    metadata: &Metadata,
    regrid: &Regrid,
    device: &B::Device,
) -> Result<(Tensor<B, 4>, Vec<ChannelStats>), Error> {
    let multistep = metadata.multistep;
    if oper_paths.len() != multistep || wave_paths.len() != multistep {
        return Err(Error::Multistep {
            expected: multistep,
            oper: oper_paths.len(),
            wave: wave_paths.len(),
        });
    }

    let num_grid = metadata.latitudes.len();
    let num_vars = metadata.model_input.full.len();
    if regrid.num_target() != num_grid {
        return Err(Error::Matrix(
            PathBuf::new(),
            format!(
                "matrix targets {} points, the checkpoint grid has {num_grid}",
                regrid.num_target()
            ),
        ));
    }

    let mut input_tensor = InputTensor::new(multistep, num_grid, num_vars);

    // lsm comes from its own file at native resolution and skips the regrid entirely. The
    // open-data copy in the operational file is 8-bit packed -- 129 distinct values against
    // 63,747 -- and a land fraction interpolated across a coastline and then quantised is a
    // materially different field. It is also the mask for step 5 below.
    let lsm = read_single(lsm_path, num_grid)?;

    let mut valid_times = Vec::with_capacity(multistep);
    for (time, (oper, wave)) in oper_paths.iter().zip(wave_paths).enumerate() {
        let mut file_time = None;

        for path in [oper, wave] {
            for_each_field(Path::new(path), |field| {
                if file_time.is_none() {
                    file_time = Some(field.valid()?);
                }

                for (index, name) in variable_names(&field).iter().enumerate() {
                    // Check if in metadata.
                    let Ok(channel) = metadata.input_channel(name) else {
                        continue; // Surplus: the 28 diagnostics, the 14 gh fields, fscov.
                    };

                    // The wave-direction split happens before the regrid, on the source grid,
                    // for the same reason the filter exists at all. See comments around the
                    // WAVE_DIRECTION const for more information.
                    let values = match field.short_name.as_str() {
                        WAVE_DIRECTION => {
                            // indices -> [cos, sin]
                            let sine = index == 1;
                            field
                                .values
                                .iter()
                                .map(|v| {
                                    let radians = v.to_radians();
                                    if sine { radians.sin() } else { radians.cos() }
                                })
                                .collect()
                        }
                        _ => field.values.clone(),
                    };

                    // Add the new variable after applying the regrid.
                    let values = &regrid.apply(&field.grid, &values)?;
                    input_tensor.write(time, channel, values);
                }
                Ok(())
            })?;
        }

        valid_times.push(file_time.ok_or_else(|| {
            Error::Time(format!(
                "{oper} and {wave} hold no messages, so no valid time"
            ))
        })?);
    }

    // lsm last, so the native field wins over the open-data copy the operational file also
    // carries under the same name.
    if let Ok(channel) = metadata.input_channel("lsm") {
        for time in 0..multistep {
            input_tensor.write(time, channel, &lsm);
        }
    }

    // apply-mask (inference.yaml:10-17): the three soil-ish fields are zeroed over sea.
    // `sd` is in neither file, so today this reaches two of the three.
    for name in ["sd", "swvl1", "swvl2"] {
        let Ok(channel) = metadata.input_channel(name) else {
            continue;
        };
        input_tensor.map(channel, |point, v| if lsm[point] == 0.0 { 0.0 } else { v });
    }

    // The nine computed forcings, per row at that row's own date. `lagged = [-6h, 0h]` relative
    // to the run time (anemoi's metadata.py:227-233), so the last file anchors the schedule.
    let timestep = TimeDelta::from_std(metadata.timestep)
        .map_err(|e| Error::Time(format!("timestep {:?}: {e}", metadata.timestep)))?;
    let anchor = valid_times[multistep - 1];
    for (time, valid) in valid_times.iter().enumerate() {
        let date = anchor - timestep * (multistep - 1 - time) as i32;
        if date != *valid {
            eprintln!(
                "warning: timestep {time} should be {date} but {} holds {valid}; its retrieved \
                 fields are stale, its forcings are not",
                oper_paths[time]
            );
        }
        // B::Device is an associated type, so the backend cannot be inferred from the device.
        write_forcings::<B>(&mut input_tensor, metadata, time, &date, device)?;
    }

    // The ConstantImputer used to run here. It belongs to processors::Processors::pre, which has
    // to see the NaN to record where they were: the post-processing stage restores them, and a
    // fill applied this early is indistinguishable from data by the time it gets there.
    check_channels(metadata, &input_tensor)?;

    // Taken before the move into TensorData, which consumes the buffer.
    let stats = input_tensor.channel_stats();

    let data = TensorData::new(input_tensor.buffer, [1, multistep, num_grid, num_vars]);
    Ok((Tensor::from_data(data, device), stats))
}

// Laid out (time * grid * vars) flat, so that the finished buffer can be created directly into
// a burn::Tensor.
struct InputTensor {
    buffer: Vec<f32>,
    written: Vec<bool>,
    multistep: usize,
    num_grid: usize,
    num_vars: usize,
}

impl InputTensor {
    fn new(multistep: usize, num_grid: usize, num_vars: usize) -> InputTensor {
        InputTensor {
            // NaN is the guard, not a fill: anemoi allocates the same way and asserts
            // afterwards that every channel was written.
            buffer: vec![f32::NAN; multistep * num_grid * num_vars],
            written: vec![false; multistep * num_vars],
            multistep,
            num_grid,
            num_vars,
        }
    }

    // One variable at one timestep, written down the channel axis.
    fn write(&mut self, time: usize, channel: usize, values: &[f32]) {
        debug_assert_eq!(values.len(), self.num_grid);
        let base = time * self.num_grid * self.num_vars;
        for (i, &value) in values.iter().enumerate() {
            self.buffer[base + i * self.num_vars + channel] = value;
        }
        self.written[time * self.num_vars + channel] = true;
    }

    // Stats for every (timestep, channel), read straight out of the flat buffer
    // before it ever reaches a device.
    fn channel_stats(&self) -> Vec<ChannelStats> {
        let mut column = Vec::with_capacity(self.num_grid);
        let mut out = Vec::with_capacity(self.multistep * self.num_vars);
        for time in 0..self.multistep {
            let base = time * self.num_grid * self.num_vars;
            for channel in 0..self.num_vars {
                column.clear();
                column.extend(
                    (0..self.num_grid)
                        .map(|point| self.buffer[base + point * self.num_vars + channel]),
                );
                out.push(ChannelStats {
                    time,
                    channel,
                    stats: stats(&column),
                });
            }
        }
        out
    }

    // Rewrite one channel in place, given each point's index -- what the mask and the imputer
    // need. Applied to every timestep, since both are time-invariant rules.
    fn map(&mut self, channel: usize, f: impl Fn(usize, f32) -> f32) {
        for time in 0..self.multistep {
            let base = time * self.num_grid * self.num_vars;
            for point in 0..self.num_grid {
                let slot = base + point * self.num_vars + channel;
                self.buffer[slot] = f(point, self.buffer[slot]);
            }
        }
    }
}

// One field from a single-message file.
fn read_single(path: &str, num_grid: usize) -> Result<Vec<f32>, Error> {
    let fields = load_grib(path)?;
    let field = fields.into_iter().next().ok_or_else(|| Error::Grid {
        path: PathBuf::from(path),
        expected: num_grid,
        found: 0,
    })?;
    if field.values.len() != num_grid {
        return Err(Error::Grid {
            path: PathBuf::from(path),
            expected: num_grid,
            found: field.values.len(),
        });
    }
    Ok(field.values)
}

fn write_forcings<B: Backend>(
    tensor: &mut InputTensor,
    metadata: &Metadata,
    time: usize,
    date: &DateTime<Utc>,
    device: &B::Device,
) -> Result<(), Error> {
    let to_tensor = |degrees: &[f64]| {
        let values: Vec<f32> = degrees.iter().map(|&v| v as f32).collect();
        Tensor::<B, 1>::from_floats(values.as_slice(), device)
    };
    let computed = forcings::compute_forcings(
        to_tensor(&metadata.latitudes),
        to_tensor(&metadata.longitudes),
        date,
    )
    .map_err(Error::Forcings)?;

    // These names are the checkpoint's own, so they index var_to_input_channel directly
    // (forcings.rs:106-107).
    let named = [
        ("sin_latitude", computed.sin_latitude),
        ("cos_latitude", computed.cos_latitude),
        ("sin_longitude", computed.sin_longitude),
        ("cos_longitude", computed.cos_longitude),
        ("sin_julian_day", computed.sin_julian_day),
        ("cos_julian_day", computed.cos_julian_day),
        ("sin_local_time", computed.sin_local_time),
        ("cos_local_time", computed.cos_local_time),
        ("insolation", computed.insolation),
    ];

    for (name, values) in named {
        let Some(&channel) = metadata.var_to_input_channel.get(name) else {
            continue;
        };
        let values: Vec<f32> = values
            .into_data()
            .to_vec()
            .map_err(|e| Error::Time(format!("reading computed {name}: {e:?}")))?;
        if values.len() != tensor.num_grid {
            return Err(Error::Grid {
                path: PathBuf::from(name),
                expected: tensor.num_grid,
                found: values.len(),
            });
        }
        tensor.write(time, channel, &values);
    }

    Ok(())
}

// Anemoi asserts every one of the 106 channels was written before it ever runs a forward pass,
// and names the ones that were not (tensors.py:168-172). One carve-out: a channel the imputer
// zero-fills is survivable, because step 7 gave it exactly the value the reference pipeline
// would have. Today that is `sd` -- absent from both files -- and nothing else.
fn check_channels(metadata: &Metadata, tensor: &InputTensor) -> Result<(), Error> {
    let mut channel_to_var = vec![String::new(); tensor.num_vars];
    for (name, &channel) in &metadata.var_to_input_channel {
        if let Some(slot) = channel_to_var.get_mut(channel) {
            *slot = name.clone();
        }
    }

    let mut missing = Vec::new();
    let mut imputed = Vec::new();
    for time in 0..tensor.multistep {
        for channel in 0..tensor.num_vars {
            if tensor.written[time * tensor.num_vars + channel] {
                continue;
            }
            let name = &channel_to_var[channel];
            if metadata.imputer_zero.contains(name) {
                if !imputed.contains(name) {
                    imputed.push(name.clone());
                }
            } else if !missing.contains(name) {
                missing.push(name.clone());
            }
        }
    }

    if !imputed.is_empty() {
        eprintln!(
            "warning: no data for {}, left to the imputer's zero fill",
            imputed.join(", ")
        );
    }
    if !missing.is_empty() {
        return Err(Error::MissingChannels(missing));
    }
    Ok(())
}

#[cfg(test)]
#[path = "grib_test.rs"]
mod tests;
