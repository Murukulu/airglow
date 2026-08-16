//! The regrid case is six N320 points and the fifteen 0.25-degree source points they draw on,
//! carrying real 2t values. Temperature rather than the land-sea mask on purpose: lsm is 0 or 1
//! almost everywhere, so a wrong source point or a dropped weight still lands on the right
//! answer, while 14 distinct temperatures around 300 K leave nowhere to hide.
//!
//! Reproduce with the checkpoint's venv:
//!
//!     import numpy as np, eccodes, earthkit.data as ekd
//!     from earthkit.regrid.db import SYS_DB
//!     z = SYS_DB.load_matrix(SYS_DB.find_entry(
//!             {"grid": [0.25, 0.25]}, {"grid": "N320"}, "linear")).tocsr()
//!     t2m = [f for f in ekd.from_source("file", "data/20260810000000-0h-oper-fc.grib2")
//!            if f.metadata("shortName") == "2t"][0].to_numpy(flatten=True)
//!     gw = np.roll(t2m.reshape(721, 1440), 720, axis=1).ravel()   # dateline -> Greenwich
//!     out = (z @ gw.reshape(-1, 1)).ravel()                       # out[target] is EXPECTED
//!     # SOURCE holds t2m at the *stored* index of each column: j*1440 + (i - 720) % 1440.
//!
//! The GRIB decode itself has no unit test here -- it needs a real message, and exercising it
//! is what `input_tensor_smoke_test` in main.rs does against the actual files.

use super::*;

// The open-data grid. 1440 columns starting at the dateline, 721 rows from the north pole.
const NI: usize = 1440;
const NJ: usize = 721;
const NUM_SOURCE: usize = NI * NJ;

fn open_data_grid() -> Grid {
    Grid::RegularLatLon {
        ni: NI,
        nj: NJ,
        lon_first: 180.0,
        di: 0.25,
    }
}

fn field(short_name: &str, level_type: &str, level: i64) -> Field {
    Field {
        param_id: 0,
        short_name: short_name.to_string(),
        level_type: level_type.to_string(),
        level,
        valid_date: 20260810,
        valid_time: 0,
        grid_type: "regular_ll".to_string(),
        grid: Grid::Stored,
        values: Vec::new(),
    }
}

// A matrix holding only the rows under test, against the full 1,038,240-column source grid --
// so the column indices, and therefore the longitude rotation, are the real ones.
fn matrix(indptr: &[i32], indices: &[i32], weights: &[f32]) -> Regrid {
    Regrid {
        indptr: indptr.to_vec(),
        indices: indices.to_vec(),
        values: weights.to_vec(),
        num_target: indptr.len() - 1,
        num_source: NUM_SOURCE,
    }
}

// A source field that is zero everywhere except the points the rows actually read.
fn source(values: &[(usize, f32)]) -> Vec<f32> {
    let mut field = vec![0.0; NUM_SOURCE];
    for &(index, value) in values {
        field[index] = value;
    }
    field
}

#[test]
fn routes_messages_to_checkpoint_variable_names() {
    let cases: [(&str, &str, i64, &[&str]); 10] = [
        // Pressure levels carry the level in the name; nothing else does.
        ("t", "isobaricInhPa", 500, &["t_500"]),
        ("q", "isobaricInhPa", 1000, &["q_1000"]),
        // Surface z is orography, a forcing. z_850 is a prognostic. Same shortName.
        ("z", "surface", 0, &["z"]),
        ("z", "isobaricInhPa", 850, &["z_850"]),
        // Height above ground is already part of the shortName.
        ("10u", "heightAboveGround", 10, &["10u"]),
        ("2t", "heightAboveGround", 2, &["2t"]),
        // The soil rename table: not derivable from any key.
        ("vsw", "soilLayer", 1, &["swvl1"]),
        ("sot", "soilLayer", 2, &["stl2"]),
        // One message, two channels.
        ("mwd", "surface", 0, &["cos_mwd", "sin_mwd"]),
        // A soil level the checkpoint does not use routes nowhere.
        ("vsw", "soilLayer", 3, &[]),
    ];

    for (short_name, level_type, level, expected) in cases {
        let names = variable_names(&field(short_name, level_type, level));
        assert_eq!(
            names, expected,
            "{short_name} @ {level_type}={level} routed to {names:?}"
        );
    }
}

// Raw values from `codes_get_array(gid, "values")` on the vsw soilLayer=1 message, at the ocean
// -> land transition nearest the middle of the field. 663,990 of its 1,038,240 points sit at the
// sentinel -- numberOfValues is 374,250 -- because soil moisture is undefined over sea and ice.
const VSW_AT: usize = 447_416;
const VSW_MISSING: f64 = 9999.0;
const VSW_RAW: [f64; 8] = [
    9999.0,
    9999.0,
    9999.0,
    9999.0,
    0.433425903,
    0.459014893,
    0.492706299,
    0.485107422,
];

#[test]
fn masked_points_become_nan() {
    let values = unmask(VSW_RAW.to_vec(), VSW_MISSING);

    assert!(
        values[..4].iter().all(|v| v.is_nan()),
        "sentinel survived at {VSW_AT}..: {:?}",
        &values[..4]
    );
    // Real measurements pass through untouched, to f32 precision.
    for (got, want) in values[4..].iter().zip(&VSW_RAW[4..]) {
        assert!((*got as f64 - want).abs() < 1e-7, "{got} against {want}");
    }
    // The sentinel is a value, not a flag: nothing else may be treated as missing.
    assert!(!values.iter().any(|v| *v == 9999.0));
}

// Six N320 points and the fifteen source points feeding them, with 2t as the payload.
const INDPTR: [i32; 7] = [0, 2, 5, 7, 10, 13, 15];
const INDICES: [i32; 15] = [
    0, 1680, 273805, 275246, 273806, 519840, 518400, 597119, 598560, 597120, 793070, 794510,
    793071, 1036800, 1036720,
];
const WEIGHTS: [f32; 15] = [
    0.139507904,
    0.860492096,
    0.0743164815,
    0.819809846,
    0.105873672,
    0.562060622,
    0.437939378,
    0.37479415,
    0.519772258,
    0.105433592,
    0.0627228349,
    0.538304364,
    0.398972801,
    0.139507904,
    0.860492096,
];
const SOURCE: [(usize, f32); 15] = [
    (720, 272.43022),
    (2400, 272.33647),
    (274525, 300.49272),
    (274526, 300.21147),
    (275966, 300.68022),
    (519120, 296.55522),
    (520560, 296.55522),
    (596399, 298.74272),
    (596400, 298.71147),
    (597840, 298.52397),
    (792350, 280.05522),
    (792351, 279.99272),
    (793790, 279.93022),
    (1036000, 219.11772),
    (1037520, 216.33647),
];
// scipy's `z @ values` at targets 3, 90000, 271040, 333333, 470000 and 542079.
const EXPECTED: [f32; 6] = [
    272.34955, 300.61666, 296.55522, 298.62573, 279.963, 218.72972,
];

#[test]
fn regrid_reproduces_earthkit() {
    let regrid = matrix(&INDPTR, &INDICES, &WEIGHTS);
    let out = regrid.apply(&open_data_grid(), &source(&SOURCE)).unwrap();

    assert_eq!(out.len(), EXPECTED.len());
    for (got, want) in out.iter().zip(EXPECTED) {
        // f32 weights against scipy's f64 accumulation, on values around 300 K.
        assert!(
            (got - want).abs() < 1e-3,
            "{got} K against earthkit's {want} K"
        );
    }
}

#[test]
fn regrid_without_the_longitude_rotation_is_wrong() {
    // Same everything, except a field claiming to start at Greenwich. The matrix's columns are
    // numbered from Greenwich and these files start at the dateline, so dropping the rotation
    // reads half a world away. This is the assertion that would fail if `shift` were ever
    // removed as a no-op -- the land-fraction check it used to have could not see it, because
    // rotating a field does not change its mean.
    let regrid = matrix(&INDPTR, &INDICES, &WEIGHTS);
    let unrotated = Grid::RegularLatLon {
        ni: NI,
        nj: NJ,
        lon_first: 0.0,
        di: 0.25,
    };
    let out = regrid.apply(&unrotated, &source(&SOURCE)).unwrap();

    let worst = out
        .iter()
        .zip(EXPECTED)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 100.0,
        "the rotation made only {worst} K of difference"
    );
}

// swh target 483, a coastal point drawing on two masked source points and one real one.
const SWH_INDICES: [i32; 3] = [17299, 18740, 17300];
const SWH_WEIGHTS: [f32; 3] = [0.805703812, 0.0856501249, 0.108646063];
const SWH_SOURCE: [(usize, f32); 3] = [(18019, f32::NAN), (18020, f32::NAN), (19460, 0.010266054)];

#[test]
fn regrid_propagates_bitmap_nan() {
    // scipy's `z @ values` gives NaN here, so a target touching masked water stays masked and
    // the mask grows by about a cell. That is what feeds the imputer, and it is why the
    // substitution in `unmask` has to happen before the regrid rather than after.
    let regrid = matrix(&[0, 3], &SWH_INDICES, &SWH_WEIGHTS);
    let out = regrid
        .apply(&open_data_grid(), &source(&SWH_SOURCE))
        .unwrap();

    assert_eq!(out.len(), 1);
    assert!(out[0].is_nan(), "expected NaN, got {}", out[0]);
}

#[test]
fn regrid_rejects_a_field_from_the_wrong_grid() {
    let regrid = matrix(&INDPTR, &INDICES, &WEIGHTS);
    let short = vec![0.0; 10];

    // Too few points on a grid the matrix knows.
    assert!(regrid.apply(&open_data_grid(), &short).is_err());
    // No geometry at all, and not already on the model grid.
    assert!(regrid.apply(&Grid::Stored, &short).is_err());
    // Already on the model grid: pass straight through, which is how lsm.grib arrives.
    let native = vec![0.5; regrid.num_target()];
    assert_eq!(regrid.apply(&Grid::Stored, &native).unwrap(), native);
}

#[test]
fn input_tensor_writes_down_the_channel_axis() {
    // 2 timesteps, 3 grid points, 4 channels. The layout is [time][grid][vars], so a channel is
    // a stride-4 walk and a timestep is a contiguous block of 12 -- exactly what
    // TensorData::new reads back as [1, 2, 3, 4].
    let mut tensor = InputTensor::new(2, 3, 4);
    tensor.write(1, 2, &[7.0, 8.0, 9.0]);

    assert_eq!(tensor.buffer[12 + 2], 7.0);
    assert_eq!(tensor.buffer[12 + 4 + 2], 8.0);
    assert_eq!(tensor.buffer[12 + 8 + 2], 9.0);
    assert!(tensor.buffer[2].is_nan(), "timestep 0 was written too");
    assert!(tensor.written[1 * 4 + 2] && !tensor.written[2]);

    // map covers every timestep, since both the mask and the imputer are time-invariant rules.
    tensor.map(2, |point, v| if point == 1 { -1.0 } else { v });
    assert_eq!(tensor.buffer[12 + 4 + 2], -1.0);
    assert_eq!(tensor.buffer[4 + 2], -1.0);
    assert_eq!(tensor.buffer[12 + 2], 7.0);
}
