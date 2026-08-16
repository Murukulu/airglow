use super::*;

use chrono::Utc;

// wgpu rather than ndarray to match the rest of the suite. Nothing here depends on the backend
// the way common_test.rs does -- these are all elementwise ops -- but the golden values are only
// meaningful against the backend we actually ship.
type TestBackend = burn::backend::wgpu::Wgpu;

// Reference values stay f64 -- that is what earthkit produced, and rounding them to f32 here
// would quietly widen the tolerance by exactly the amount we are trying to measure.
fn assert_close(got: Vec<f32>, want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let g = *g as f64;
        assert!(
            (g - w).abs() < tol,
            "element {}: got {}, want {} (tol {})",
            i,
            g,
            w,
            tol
        );
    }
}

// Every golden value below came out of the real earthkit stack, not from re-deriving the
// formulas here -- otherwise the test would only prove we can transcribe our own transcription:
//
//   import datetime, numpy as np, earthkit.data as ekd
//   fl = ekd.from_source('forcings', latitudes=LATS, longitudes=LONS, date=DATES, param=PARAMS)
//   for f in fl: print(f.metadata('param'), f.to_numpy().flatten())"
//
// earthkit-data 0.20.0, whose ForcingMaker.insolation delegates to
// earthkit.meteo.solar.array.solar.cos_solar_zenith_angle.
//
// The grid deliberately avoids 45/90/180/270: those angles collapse most of the nine forcings to
// exact 0s and +/-1s, which a swapped sin/cos or a sign error would sail straight through.
const LATS: [f32; 5] = [0.0, 51.5, -33.9, 89.0, 12.25];
const LONS: [f32; 5] = [0.0, 103.8, 151.2, 288.7, 359.0];

// Position-only, so the same for every date.
const SIN_LATITUDE: [f64; 5] = [0.0, 0.78260816, -0.55774511, 0.99984770, 0.21217767];
const COS_LATITUDE: [f64; 5] = [1.0, 0.62251464, 0.83001229, 0.01745241, 0.97723111];
const SIN_LONGITUDE: [f64; 5] = [0.0, 0.97113428, 0.48175367, -0.94721028, -0.01745241];
const COS_LONGITUDE: [f64; 5] = [1.0, -0.23853346, -0.87630668, 0.32061299, 0.99984770];

struct Case {
    // year, month, day, hour, minute -- all UTC.
    date: (i32, u32, u32, u32, u32),
    julian_day: f64,
    local_time: [f64; 5],
    sin_julian_day: [f64; 5],
    cos_julian_day: [f64; 5],
    sin_local_time: [f64; 5],
    cos_local_time: [f64; 5],
    insolation: [f64; 5],
}

const CASES: [Case; 4] = [
    // Solstice at 00Z. The hour angle here is (0 - 12) * 15, which is what used to underflow
    // when the hour was read out of chrono as a u32.
    Case {
        date: (2024, 6, 21, 0, 0),
        julian_day: 172.0,
        local_time: [0.0, 6.92, 10.08, 19.24666667, 23.93333333],
        sin_julian_day: [0.18175979; 5],
        cos_julian_day: [-0.98334296; 5],
        sin_local_time: [0.0, 0.97113428, 0.48175367, -0.94721028, -0.01745241],
        cos_local_time: [1.0, -0.23853346, -0.87630668, 0.32061299, 0.99984770],
        insolation: [0.0, 0.44404246, 0.44279100, 0.39294398, 0.0],
    },
    // Solstice at 18Z. local_time wraps past 24 at lon 103.8 and lon 151.2.
    Case {
        date: (2024, 6, 21, 18, 0),
        julian_day: 172.75,
        local_time: [18.0, 0.92, 4.08, 13.24666667, 17.93333333],
        sin_julian_day: [0.16905810; 5],
        cos_julian_day: [-0.98560609; 5],
        sin_local_time: [-1.0, 0.23853346, 0.87630668, -0.32061299, -0.99984770],
        cos_local_time: [-0.0, 0.97113428, 0.48175367, -0.94721028, -0.01745241],
        insolation: [0.00674789, 0.0, 0.0, 0.41315839, 0.10668893],
    },
    // julian_day == 0.0 exactly.
    Case {
        date: (2024, 1, 1, 0, 0),
        julian_day: 0.0,
        local_time: [0.0, 6.92, 10.08, 19.24666667, 23.93333333],
        sin_julian_day: [0.0; 5],
        cos_julian_day: [1.0; 5],
        sin_local_time: [0.0, 0.97113428, 0.48175367, -0.94721028, -0.01745241],
        cos_local_time: [1.0, -0.23853346, -0.87630668, 0.32061299, 0.99984770],
        insolation: [0.0, 0.0, 0.88297153, 0.0, 0.0],
    },
    // 06:30 -- the only case with a genuinely fractional julian_day and a half-hour local time.
    // If julian_day ever loses its fraction again, the two julian_day rows here move and the
    // three whole-hour cases above do not. insolation still uses hour 6: earthkit discards
    // minutes in the hour angle (solar.py:178), and so do we.
    Case {
        date: (2024, 6, 21, 6, 30),
        julian_day: 172.27083333333334,
        local_time: [6.5, 13.42, 16.58, 1.74666667, 6.43333333],
        sin_julian_day: [0.17717645; 5],
        cos_julian_day: [-0.98417910; 5],
        sin_local_time: [0.99144486, -0.36325123, -0.93169123, 0.44150585, 0.99357186],
        cos_local_time: [
            -0.13052619,
            -0.93169123,
            -0.36325123,
            0.89725837,
            -0.11320321,
        ],
        insolation: [0.0, 0.86702465, 0.14941175, 0.38277587, 0.06262595],
    },
];

// f32 tensors against f64 numpy. The widest gap is in insolation, which accumulates a handful of
// products before the clamp.
const TOL: f64 = 1e-6;

#[test]
fn forcings_match_earthkit() {
    let device = Default::default();

    for case in &CASES {
        let (y, m, d, h, min) = case.date;
        let date = Utc.with_ymd_and_hms(y, m, d, h, min, 0).unwrap();
        let lat = Tensor::<TestBackend, 1>::from_floats(LATS, &device);
        let long = Tensor::<TestBackend, 1>::from_floats(LONS, &device);

        let got = compute_forcings(lat, long, &date).expect("compute_forcings");

        let fields: [(&str, Tensor<TestBackend, 1>, &[f64; 5]); 9] = [
            ("sin_latitude", got.sin_latitude, &SIN_LATITUDE),
            ("cos_latitude", got.cos_latitude, &COS_LATITUDE),
            ("sin_longitude", got.sin_longitude, &SIN_LONGITUDE),
            ("cos_longitude", got.cos_longitude, &COS_LONGITUDE),
            ("sin_julian_day", got.sin_julian_day, &case.sin_julian_day),
            ("cos_julian_day", got.cos_julian_day, &case.cos_julian_day),
            ("sin_local_time", got.sin_local_time, &case.sin_local_time),
            ("cos_local_time", got.cos_local_time, &case.cos_local_time),
            ("insolation", got.insolation, &case.insolation),
        ];

        for (name, tensor, want) in fields {
            let values = tensor.into_data().to_vec::<f32>().unwrap();
            for (i, (g, w)) in values.iter().zip(want).enumerate() {
                let g = *g as f64;
                assert!(
                    (g - w).abs() < TOL,
                    "{name}[{i}] at {date}: got {g}, want {w} (earthkit, tol {TOL})",
                );
            }
        }
    }
}

// julian_day is a zero-based *fractional* day of year. Both halves matter and both were once
// wrong: the fraction was dropped (year_start kept the time of day) and the whole-day part was
// counted twice (chrono's num_seconds is the total, not Python's within-day remainder).
//
// Reference: earthkit.meteo.solar.array.solar.julian_day.
#[test]
fn julian_day_is_a_fractional_day_of_year() {
    for case in &CASES {
        let (y, m, d, h, min) = case.date;
        let date = Utc.with_ymd_and_hms(y, m, d, h, min, 0).unwrap();
        let got = julian_day(&date).expect("julian_day");
        assert!(
            (got - case.julian_day).abs() < 1e-9,
            "julian_day({date}): got {got}, want {} (earthkit)",
            case.julian_day,
        );
    }

    // The last hour of a leap year, well past the 365.25 divisor.
    let end_of_year = Utc.with_ymd_and_hms(2024, 12, 31, 23, 0, 0).unwrap();
    let got = julian_day(&end_of_year).expect("julian_day");
    assert!((got - 365.9583333333333).abs() < 1e-9, "got {got}");
}

// local_time has to move with the clock. It once did not: the day start was computed with
// with_day(date.day()), a no-op that left hours_since_midnight pinned at zero, which would have
// frozen sin/cos_local_time across a rollout without failing anything loudly.
//
// Reference: the `local_time` param of earthkit's ForcingMaker
// (earthkit/data/sources/forcings.py:150-159).
#[test]
fn local_time_tracks_the_clock() {
    let device = Default::default();

    for case in &CASES {
        let (y, m, d, h, min) = case.date;
        let date = Utc.with_ymd_and_hms(y, m, d, h, min, 0).unwrap();
        let long = Tensor::<TestBackend, 1>::from_floats(LONS, &device);

        let got = local_time(long, &date).into_data().to_vec::<f32>().unwrap();
        assert_close(got, &case.local_time, 1e-5);
    }

    // At the prime meridian local time is just the UTC hour, across all four AIFS cycles. This
    // is the assertion that pins the bug most directly: it fails with every value at 0.
    for hour in [0, 6, 12, 18] {
        let date = Utc.with_ymd_and_hms(2024, 3, 15, hour, 0, 0).unwrap();
        let long = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let got = local_time(long, &date).into_data().to_vec::<f32>().unwrap();
        assert_close(got, &[hour as f64], 1e-6);
    }
}

// Night is exactly zero, never negative -- earthkit clips at 0 (solar.py:182) and the checkpoint
// was trained on clipped values.
#[test]
fn insolation_is_never_negative() {
    let device = Default::default();

    // A coarse global grid, every three hours through a day, so both poles and both sides of the
    // terminator are covered.
    let mut lats = Vec::new();
    let mut lons = Vec::new();
    for lat_step in 0..37 {
        for lon_step in 0..72 {
            lats.push(-90.0 + lat_step as f32 * 5.0);
            lons.push(lon_step as f32 * 5.0);
        }
    }

    for hour in (0..24).step_by(3) {
        let date = Utc.with_ymd_and_hms(2024, 6, 21, hour, 0, 0).unwrap();
        let lat = Tensor::<TestBackend, 1>::from_floats(lats.as_slice(), &device);
        let long = Tensor::<TestBackend, 1>::from_floats(lons.as_slice(), &device);

        let got = insolation(lat, long, &date).expect("insolation");
        let min = got.min().into_scalar();
        assert!(min >= 0.0, "insolation went negative ({min}) at {date}");
    }
}
