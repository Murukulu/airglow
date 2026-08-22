use std::collections::HashMap;
use std::time::Duration;

use super::*;
use crate::metadata::IndexSet;

type TestBackend = burn::backend::wgpu::Wgpu;

// A four-channel output space. `tp` is the referent for the FractionBounding cases and `cp` the
// fraction of it; `other` is a channel no bounding names, which catches a mask that is too wide.
//
// Deliberately, neither `tp` nor any bounded variable sits at channel 0. An earlier layout with
// `tp` at 0 passed against a `select` that was silently returning channel 0 whatever index it was
// given -- see the note on `FractionBounding::total_var`.
const OTHER: usize = 0;
const CP: usize = 1;
const TCC: usize = 2;
const TP: usize = 3;
const VARS: usize = 4;
const ROWS: usize = 3;

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

// Only `var_to_output_channel` and `model_output.full` are read by the boundings; the rest is
// inert, as in aifs_test.rs.
fn metadata() -> Metadata {
    let set = |full: Vec<usize>| IndexSet {
        full,
        prognostic: Vec::new(),
        diagnostic: Vec::new(),
        forcing: Vec::new(),
    };
    let var_to_output_channel = HashMap::from([
        ("other".to_string(), OTHER),
        ("cp".to_string(), CP),
        ("tcc".to_string(), TCC),
        ("tp".to_string(), TP),
    ]);
    Metadata {
        variables: Vec::new(),
        multistep: 2,
        timestep: Duration::from_secs(6 * 60 * 60),
        data_input: set(Vec::new()),
        data_output: set(Vec::new()),
        model_input: set(Vec::new()),
        model_output: set((0..VARS).collect()),
        var_to_input_channel: HashMap::new(),
        var_to_output_channel,
        output_channel_to_var: names(&["other", "cp", "tcc", "tp"]),
        computed_forcing: Vec::new(),
        constant_in_time: Vec::new(),
        imputer_zero: Vec::new(),
        boundings: Vec::new(),
        nan_postprocessor_reference: String::new(),
        nan_postprocessor_vars: Vec::new(),
        latitudes: Vec::new(),
        longitudes: Vec::new(),
    }
}

fn build(conf: BoundingConfig) -> Result<BoundingType<TestBackend>, metadata::Error> {
    let device = Default::default();
    BoundingType::from_bounding_config(&metadata(), &conf, &ChannelKind::Output, &device)
}

// Given channel-major -- one row per channel, in channel order -- because every assertion here is
// about a single channel across all grid points.
fn tensor(channels: [[f32; ROWS]; VARS]) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    let mut flat = Vec::with_capacity(ROWS * VARS);
    for row in 0..ROWS {
        for channel in channels.iter() {
            flat.push(channel[row]);
        }
    }
    Tensor::from_data(TensorData::new(flat, [ROWS, VARS]), &device)
}

fn column(x: &Tensor<TestBackend, 2>, channel: usize, rows: usize) -> Vec<f32> {
    x.clone()
        .slice([0..rows, channel..channel + 1])
        .into_data()
        .to_vec::<f32>()
        .expect("column is not f32")
}

fn assert_column(x: &Tensor<TestBackend, 2>, channel: usize, want: [f32; ROWS]) {
    for (i, (g, w)) in column(x, channel, ROWS).iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "channel {channel} row {i}: got {g}, want {w}",
        );
    }
}

#[test]
fn relu_bounds_only_its_own_channels() {
    let bounding = build(BoundingConfig::Relu {
        variables: names(&["tp", "cp"]),
    })
    .unwrap();

    let out = bounding.forward(tensor([
        [-9.0, -8.0, 5.0], // other: untouched
        [-3.0, 1.0, -0.5], // cp:    bounded
        [-7.0, 4.0, -1.0], // tcc:   untouched
        [-1.0, 0.0, 2.0],  // tp:    bounded
    ]));

    assert_column(&out, TP, [0.0, 0.0, 2.0]);
    assert_column(&out, CP, [0.0, 1.0, 0.0]);
    // The surviving negatives are the point: a mask one channel too wide would zero them.
    assert_column(&out, TCC, [-7.0, 4.0, -1.0]);
    assert_column(&out, OTHER, [-9.0, -8.0, 5.0]);
}

#[test]
fn hardtanh_clamps_to_range() {
    let bounding = build(BoundingConfig::Hardtanh {
        variables: names(&["tcc"]),
        min_val: 0.0,
        max_val: 1.0,
    })
    .unwrap();

    let out = bounding.forward(tensor([
        [0.0, 0.0, 0.0],   // other
        [0.0, 0.0, 0.0],   // cp
        [-0.25, 0.5, 3.0], // tcc: below, inside, above
        [5.0, -5.0, 0.5],  // tp:  untouched, and outside [0, 1] on purpose
    ]));

    assert_column(&out, TCC, [0.0, 0.5, 1.0]);
    assert_column(&out, TP, [5.0, -5.0, 0.5]);
}

#[test]
fn fraction_multiplies_by_total_var() {
    let bounding = build(BoundingConfig::Fraction {
        variables: names(&["cp"]),
        min_val: 0.0,
        max_val: 1.0,
        total_var: "tp".to_string(),
    })
    .unwrap();

    let out = bounding.forward(tensor([
        [7.0, 7.0, 7.0],  // other
        [0.5, 2.0, -1.0], // cp:  in range, above max, below min
        [0.0, 0.0, 0.0],  // tcc
        [2.0, 4.0, 10.0], // tp:  the referent
    ]));

    // clamp(cp, 0, 1) * tp: 0.5*2, 1.0*4, 0.0*10.
    assert_column(&out, CP, [1.0, 4.0, 0.0]);
    // The referent is read, never written.
    assert_column(&out, TP, [2.0, 4.0, 10.0]);
    assert_column(&out, OTHER, [7.0, 7.0, 7.0]);
}

// The two FractionBoundings in the real checkpoint take their fraction of a variable an earlier
// entry has already clamped. Reversed, the negative `tp` here survives into the product.
#[test]
fn boundings_apply_in_list_order() {
    let device = Default::default();
    let metadata = metadata();
    let build =
        |conf| BoundingType::from_bounding_config(&metadata, &conf, &ChannelKind::Output, &device);
    let relu = || {
        build(BoundingConfig::Relu {
            variables: names(&["tp"]),
        })
    };
    let fraction = || {
        build(BoundingConfig::Fraction {
            variables: names(&["cp"]),
            min_val: 0.0,
            max_val: 1.0,
            total_var: "tp".to_string(),
        })
    };

    let x = tensor([
        [0.0, 0.0, 0.0],  // other
        [0.5, 0.5, 0.5],  // cp
        [0.0, 0.0, 0.0],  // tcc
        [-4.0, 4.0, 0.0], // tp: negative at row 0
    ]);

    let ordered = Bounding::<TestBackend>::new(vec![relu().unwrap(), fraction().unwrap()]);
    let out = ordered.forward(x.clone());

    assert_column(&out, TP, [0.0, 4.0, 0.0]);
    // Row 0: tp was relu'd to 0 first, so cp lands at 0 rather than at 0.5 * -4 = -2.
    assert_column(&out, CP, [0.0, 2.0, 0.0]);

    let reversed = Bounding::<TestBackend>::new(vec![fraction().unwrap(), relu().unwrap()]);
    assert_column(&reversed.forward(x), CP, [-2.0, 2.0, 0.0]);
}

#[test]
fn unresolvable_variable() {
    let error_msg = build(BoundingConfig::Relu {
        variables: names(&["tp", "not_a_variable"]),
    })
    .unwrap_err()
    .to_string();
    assert_eq!(
        error_msg,
        "mask added variables not in Output channel: \"not_a_variable\""
    );
}

// The forward reads the total column from the tensor it then writes, so an overlap would put the
// port and anemoi in different orders.
#[test]
#[should_panic(expected = "its own variables")]
fn fraction_total_var_may_not_be_one_of_its_variables() {
    build(BoundingConfig::Fraction {
        variables: names(&["cp", "tp"]),
        min_val: 0.0,
        max_val: 1.0,
        total_var: "tp".to_string(),
    })
    .unwrap();
}

// The cases above are all one or two channels wide over four. The checkpoint's first bounding is
// 26 wide over 120, and index-space mistakes are exactly the kind that only show up at the real
// widths and the real channel numbers, so run the real config too.
#[test]
fn checkpoint_boundings_hold_their_invariants() {
    let device = Default::default();
    let metadata = Metadata::load(std::path::Path::new(
        "./data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata",
    ))
    .expect("checkpoint metadata");

    let width = metadata.model_output.full.len();
    let boundings = Bounding::<TestBackend>::new(
        metadata
            .boundings
            .iter()
            .map(|conf| {
                BoundingType::from_bounding_config(&metadata, conf, &ChannelKind::Output, &device)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
    );

    // Spread across zero so every branch of every clamp is taken.
    const N: usize = 8;
    let x = Tensor::<TestBackend, 1, Int>::arange(0..(N * width) as i64, &device)
        .float()
        .reshape([N, width])
        * 0.01
        - 4.0;

    let out = boundings.forward(x);
    let channel = |name: &str| metadata.output_channel(name).expect(name);
    let values = |name: &str| column(&out, channel(name), N);

    // ReluBounding, first and widest.
    for name in [
        "tp", "ro", "tcw", "ssrd", "sd", "q_50", "swh", "mwp", "h2530",
    ] {
        for (i, v) in values(name).iter().enumerate() {
            assert!(*v >= 0.0, "{name} row {i}: got {v}, want >= 0");
        }
    }

    // HardtanhBounding(0, 1).
    for name in ["tcc", "swvl1", "swvl2", "snowc"] {
        for (i, v) in values(name).iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(v),
                "{name} row {i}: got {v}, want [0, 1]",
            );
        }
    }

    // The two FractionBoundings: each variable is a share of a total the earlier entries have
    // already made non-negative, so it lands in [0, total].
    for (total, vars) in [("tp", &["cp", "sf"][..]), ("tcc", &["lcc", "mcc", "hcc"])] {
        let total_values = values(total);
        for name in vars {
            for (i, (v, t)) in values(name).iter().zip(&total_values).enumerate() {
                assert!(
                    *v >= 0.0 && v <= t,
                    "{name} row {i}: got {v}, want [0, {total}={t}]",
                );
            }
        }
    }

    // A channel no bounding names comes through untouched.
    let untouched = channel("2t");
    let want: Vec<f32> = (0..N)
        .map(|row| (row * width + untouched) as f32 * 0.01 - 4.0)
        .collect();
    for (i, (g, w)) in column(&out, untouched, N).iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-3, "2t row {i}: got {g}, want {w}");
    }
}
