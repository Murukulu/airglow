use std::collections::HashMap;
use std::path::Path;

use chrono::TimeZone;

use super::*;
use crate::grib;
use crate::metadata::IndexSet;

const N320_POINTS: usize = 542_080;
const TEMPLATE_DIR: &str = "./data/templates";

fn mars(param: &str, levtype: &str, levelist: Option<i64>, stream: &str) -> Mars {
    Mars {
        param: param.to_string(),
        levtype: levtype.to_string(),
        levelist,
        stream: Some(stream.to_string()),
    }
}

// Only what `fields` reads: the output channel names, the grid size, the MARS blocks and the
// accumulation list. Everything else is inert, as in bounding_test.rs.
fn metadata(outputs: &[&str], grid: usize) -> Metadata {
    let set = || IndexSet {
        full: Vec::new(),
        prognostic: Vec::new(),
        diagnostic: Vec::new(),
        forcing: Vec::new(),
    };
    let names: Vec<String> = outputs.iter().map(|s| s.to_string()).collect();
    Metadata {
        variables: names.clone(),
        multistep: 2,
        timestep: Duration::from_secs(6 * 60 * 60),
        data_input: set(),
        data_output: set(),
        model_input: set(),
        model_output: IndexSet {
            full: (0..names.len()).collect(),
            ..set()
        },
        var_to_input_channel: HashMap::new(),
        var_to_output_channel: names
            .iter()
            .enumerate()
            .map(|(c, name)| (name.clone(), c))
            .collect(),
        output_channel_to_var: names,
        computed_forcing: Vec::new(),
        constant_in_time: Vec::new(),
        imputer_zero: Vec::new(),
        boundings: Vec::new(),
        accumulations: vec!["tp".to_string()],
        mars: HashMap::from([
            ("2t".to_string(), mars("2t", "sfc", None, "oper")),
            ("z_500".to_string(), mars("z", "pl", Some(500), "oper")),
            ("tp".to_string(), mars("tp", "sfc", None, "oper")),
            ("mwd".to_string(), mars("mwd", "sfc", None, "wave")),
        ]),
        nan_postprocessor_reference: String::new(),
        nan_postprocessor_vars: Vec::new(),
        latitudes: vec![0.0; grid],
        longitudes: vec![0.0; grid],
    }
}

#[test]
fn wave_direction_inverts_the_split() {
    let cos = [1.0, 0.0, -1.0, 0.0, f64::NAN, 0.5];
    let sin = [0.0, 1.0, 0.0, -1.0, 0.0, f64::NAN];
    let degrees = wave_direction(&cos, &sin);
    assert_eq!(degrees[..4], [0.0, 90.0, 180.0, 270.0]);
    assert!(degrees[4].is_nan() && degrees[5].is_nan());

    // Round trip through the forward transform in grib.rs for an arbitrary direction.
    let theta: f64 = 213.7;
    let back = wave_direction(&[theta.to_radians().cos()], &[theta.to_radians().sin()]);
    assert!((back[0] - theta).abs() < 1e-9, "{}", back[0]);
}

#[test]
fn accumulations_clamp_at_zero_and_keep_nan() {
    let mut values = [-1.0, 0.0, 0.5, f64::NAN];
    clamp_accumulation(&mut values);
    assert_eq!(values[..3], [0.0, 0.0, 0.5]);
    assert!(values[3].is_nan());
}

// Two grid points, four channels, the wave pair split around another variable so the merge has
// to find its partner rather than assume adjacency.
#[test]
fn fields_split_columns_and_merge_wave_direction() {
    let metadata = metadata(&["2t", "cos_mwd", "tp", "sin_mwd"], 2);
    #[rustfmt::skip]
    let host = [
        280.0, 0.0, -0.5,  1.0,   // point 0: mwd = 90, tp clamps to 0
        290.0, 1.0,  2.0,  0.0,   // point 1: mwd = 0
    ];

    let fields = fields(&host, &metadata).unwrap();
    let names: Vec<_> = fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["2t", "mwd", "tp"]);

    assert_eq!(fields[0].values, [280.0, 290.0]);
    assert_eq!(fields[1].values, [90.0, 0.0]);
    assert_eq!(fields[1].mars.stream.as_deref(), Some("wave"));
    assert_eq!(fields[2].values, [0.0, 2.0]);
    assert!(fields[2].accumulated && !fields[0].accumulated);
}

#[test]
fn fields_reject_a_tensor_of_the_wrong_size() {
    let metadata = metadata(&["2t"], 3);
    match fields(&[1.0, 2.0], &metadata) {
        Err(Error::Shape {
            expected: 3,
            found: 2,
        }) => {}
        other => panic!("{other:?}"),
    }
}

// Write on the real grid with the real templates and read it back through the input decoder.
// This is the check that the encoded messages are ones ecCodes itself agrees with: shortName,
// level and step resolve, the bitmap restores NaN, and the point count is the grid's.
#[test]
fn round_trip_through_eccodes() {
    let templates = Templates::load(Path::new(TEMPLATE_DIR)).expect("data/templates");
    let metadata = metadata(&["2t", "z_500", "tp"], N320_POINTS);

    // A ramp per channel so a transposed or shifted column would be visible, one NaN in 2t, a
    // negative in tp for the clamp.
    let vars = 3;
    let mut host = vec![0f32; N320_POINTS * vars];
    for i in 0..N320_POINTS {
        host[i * vars] = 250.0 + (i % 1000) as f32 * 0.05;
        host[i * vars + 1] = 50_000.0 + i as f32;
        host[i * vars + 2] = if i % 2 == 0 { 0.001 } else { -0.001 };
    }
    host[7 * vars] = f32::NAN;

    let path = std::env::temp_dir().join(format!("aifsv2-output-{}.grib2", std::process::id()));
    let reference = Utc.with_ymd_and_hms(2026, 8, 31, 0, 0, 0).unwrap();
    let written = write_host(
        &path,
        &host,
        &metadata,
        &templates,
        reference,
        Duration::from_secs(6 * 60 * 60),
        &AIFS_SINGLE,
    )
    .unwrap();
    assert_eq!(written, 3);

    let mut fields = Vec::new();
    grib::for_each_field(&path, |field| {
        fields.push(field);
        Ok(())
    })
    .unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(fields.len(), 3);
    for field in &fields {
        assert_eq!(field.values.len(), N320_POINTS, "{}", field.short_name);
        assert_eq!(field.grid_type, "reduced_gg");
        assert_eq!(
            (field.valid_date, field.valid_time),
            (20260831, 600),
            "{}",
            field.short_name
        );
    }

    let t2 = &fields[0];
    assert_eq!(
        (t2.short_name.as_str(), t2.level_type.as_str()),
        ("2t", "heightAboveGround")
    );
    assert!(t2.values[7].is_nan(), "the bitmap did not survive");
    assert!((t2.values[8] - (250.0 + 8.0 * 0.05)).abs() < 1e-2);
    assert_eq!(t2.values.iter().filter(|v| v.is_nan()).count(), 1);

    let z = &fields[1];
    assert_eq!(
        (z.short_name.as_str(), z.level_type.as_str(), z.level),
        ("z", "isobaricInhPa", 500)
    );
    // CCSDS is lossless on the packed integers but the packing itself quantises; 24 bits over
    // this range is far below 1.
    assert!((z.values[N320_POINTS - 1] - (50_000.0 + (N320_POINTS - 1) as f32)).abs() < 1.0);

    let tp = &fields[2];
    assert_eq!(tp.short_name, "tp");
    assert!(tp.values.iter().all(|v| *v >= 0.0), "clamp did not apply");
}

// Every output channel of the real checkpoint, through the real templates. The identity keys
// are the part ecCodes can refuse -- a shortName its tables do not know, a level type that does
// not take a level -- and it refuses per variable, so only the full set proves the writer.
#[test]
fn every_checkpoint_output_encodes() {
    let templates = Templates::load(Path::new(TEMPLATE_DIR)).expect("data/templates");
    let metadata =
        Metadata::load(Path::new("./data/quiet_grub/anemoi-metadata")).expect("checkpoint");
    let vars = metadata.output_channel_to_var.len();
    let grid = metadata.latitudes.len();
    assert_eq!((grid, vars), (N320_POINTS, 120));

    // Channel c holds the constant c, so a message's mean names the column it came from.
    let host: Vec<f32> = (0..grid * vars).map(|i| (i % vars) as f32).collect();
    let path = std::env::temp_dir().join(format!("aifsv2-all-{}.grib2", std::process::id()));
    let reference = Utc.with_ymd_and_hms(2026, 8, 31, 0, 0, 0).unwrap();
    let written = write_host(
        &path,
        &host,
        &metadata,
        &templates,
        reference,
        metadata.timestep,
        &AIFS_SINGLE,
    )
    .unwrap();
    assert_eq!(written, vars - 1, "cos/sin_mwd fold into one mwd message");

    let mut read = Vec::new();
    grib::for_each_field(&path, |field| {
        read.push(field);
        Ok(())
    })
    .unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(read.len(), written);

    // Walk the two lists together: same order, and each message names its own column.
    let expected = fields(&host, &metadata).unwrap();
    for (want, got) in expected.iter().zip(&read) {
        assert_eq!(got.short_name, want.mars.param, "{}", want.name);
        assert_eq!(got.values.len(), N320_POINTS, "{}", want.name);
        if let Some(level) = want.mars.levelist {
            assert_eq!(
                (got.level_type.as_str(), got.level),
                ("isobaricInhPa", level)
            );
        }
        // Sum in f64: 542,080 values of ~80 exceed f32's exact integer range and drift.
        let mean = got.values.iter().map(|v| *v as f64).sum::<f64>() / got.values.len() as f64;
        let want_mean = want.values.iter().sum::<f64>() / want.values.len() as f64;
        assert!(
            (mean - want_mean).abs() < 1e-3,
            "{}: {mean} vs {want_mean}",
            want.name
        );
    }
}
