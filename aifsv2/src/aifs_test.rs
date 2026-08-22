use std::collections::HashMap;
use std::time::Duration;

use super::*;
use burn::module::{ModuleMapper, Param};

use crate::metadata::IndexSet;

// wgpu rather than ndarray for the same reason as the other suites: the scatter-adds in
// graph_tranformer_conv and in the prognostic residual both hit duplicate indices, and
// duplicate-index safety is a property of the kernel.
type TestBackend = burn::backend::wgpu::Wgpu;

// The real channel counts, so the input layout under test is the one the real model sees.
const NUM_INPUT_CHANNELS: usize = 106;
const NUM_OUTPUT_CHANNELS: usize = 120;
const MULTISTEP: usize = 2;

// Everything else is shrunk to fit. 64 / 8 = 8 channels per head.
const NUM_CHANNELS: usize = 64;
const NUM_HEADS: usize = 8;
const N_DATA: usize = 6;
const N_HIDDEN: usize = 3;

// The first few entries of data_indices.model.{input,output}.prognostic. The two index spaces are
// offset from each other by design -- output channel 2 is input channel 0 -- so a test that used
// the same list for both would pass even if forward mixed them up.
const INPUT_PROGNOSTIC: [usize; 5] = [0, 1, 2, 3, 4];
const OUTPUT_PROGNOSTIC: [usize; 5] = [2, 3, 4, 5, 6];

// AifsV2Config carries the whole Metadata, so the test has to build one. Only the two widths,
// multistep and the two prognostic index sets are read by init; the rest is inert here.
fn metadata() -> Metadata {
    let set = |full: Vec<usize>, prognostic: Vec<usize>| IndexSet {
        full,
        prognostic,
        diagnostic: Vec::new(),
        forcing: Vec::new(),
    };
    Metadata {
        variables: Vec::new(),
        multistep: MULTISTEP,
        timestep: Duration::from_secs(6 * 60 * 60),
        data_input: set(Vec::new(), Vec::new()),
        data_output: set(Vec::new(), Vec::new()),
        model_input: set((0..NUM_INPUT_CHANNELS).collect(), INPUT_PROGNOSTIC.to_vec()),
        model_output: set(
            (0..NUM_OUTPUT_CHANNELS).collect(),
            OUTPUT_PROGNOSTIC.to_vec(),
        ),
        var_to_input_channel: HashMap::new(),
        var_to_output_channel: HashMap::new(),
        output_channel_to_var: Vec::new(),
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

fn small_config() -> AifsV2Config {
    AifsV2Config::new(NUM_CHANNELS, metadata())
        .with_num_heads(NUM_HEADS)
        .with_num_processor_layers(2)
        .with_num_processor_chunks(1)
}

fn input(batch: usize, device: &Device<TestBackend>) -> Tensor<TestBackend, 4> {
    let n = batch * MULTISTEP * N_DATA * NUM_INPUT_CHANNELS;
    Tensor::<TestBackend, 1, Int>::arange(0..n as i64, device)
        .float()
        .reshape([batch, MULTISTEP, N_DATA, NUM_INPUT_CHANNELS])
        * 0.01
}

#[test]
fn forward_maps_input_grid_to_output_channels() {
    let device = Default::default();
    let graph = GraphData::<TestBackend>::synthetic(N_DATA, N_HIDDEN, &device);
    let model = small_config().init::<TestBackend>(&graph, &device).unwrap();

    let out = model.forward(input(1, &device));

    assert_eq!(out.shape().dims::<2>(), [N_DATA, NUM_OUTPUT_CHANNELS]);
    // Catches both NaN and inf: the sparse softmax divides by a per-destination sum, so an
    // isolated node or an underflowed denominator would show up here rather than as a bad shape.
    let total = out.sum().into_scalar();
    assert!(total.is_finite(), "forward produced {total}");
}

// Time is the outer index of the assembled channel axis: channels 0..vars are t-6h and
// vars..2*vars are t. Nothing downstream is shaped differently if the two are swapped, so this is
// the only place the ordering is checked.
#[test]
fn assemble_input_folds_time_into_the_outer_channel_index() {
    let device: Device<TestBackend> = Default::default();
    let graph = GraphData::<TestBackend>::synthetic(N_DATA, N_HIDDEN, &device);
    let model = small_config().init::<TestBackend>(&graph, &device).unwrap();

    let ((data_latent, hidden_latent), skip) = model.assemble_input(input(1, &device));

    // multistep * vars, then the node attributes: 2 coordinate columns doubled to sin/cos, plus
    // the 8 trainable columns.
    const ATTRS: usize = 2 * 2 + 8;
    assert_eq!(
        data_latent.shape().dims::<2>(),
        [N_DATA, MULTISTEP * NUM_INPUT_CHANNELS + ATTRS]
    );
    assert_eq!(hidden_latent.shape().dims::<2>(), [N_HIDDEN, ATTRS]);
    assert_eq!(skip.shape().dims::<2>(), [N_DATA, NUM_INPUT_CHANNELS]);

    let got = data_latent.into_data().to_vec::<f32>().unwrap();
    let width = MULTISTEP * NUM_INPUT_CHANNELS + ATTRS;
    for t in 0..MULTISTEP {
        for g in 0..N_DATA {
            for v in [0, 1, NUM_INPUT_CHANNELS - 1] {
                // input() is an arange over [batch, time, grid, vars] scaled by 0.01.
                let want = ((t * N_DATA + g) * NUM_INPUT_CHANNELS + v) as f32 * 0.01;
                let got = got[g * width + t * NUM_INPUT_CHANNELS + v];
                assert!(
                    (got - want).abs() < 1e-4,
                    "grid {g} channel {v} at t={t}: got {got}, want {want}",
                );
            }
        }
    }
}

// The graph is disconnected per batch element (expand_edges offsets each copy), so a batch of 2
// must produce twice the rows and nothing else may change.
#[test]
fn forward_batches_along_the_grid_axis() {
    let device = Default::default();
    let graph = GraphData::<TestBackend>::synthetic(N_DATA, N_HIDDEN, &device);
    let model = small_config().init::<TestBackend>(&graph, &device).unwrap();

    let out = model.forward(input(2, &device));

    assert_eq!(out.shape().dims::<2>(), [2 * N_DATA, NUM_OUTPUT_CHANNELS]);
}

// The residual is the one place the two index spaces meet. With every weight zeroed the network
// contributes nothing, so the output is exactly the scattered skip: channel OUTPUT_PROGNOSTIC[k]
// holds input channel INPUT_PROGNOSTIC[k] of the *last* timestep, and every other channel is 0.
#[test]
fn prognostic_residual_scatters_the_last_timestep() {
    let device: Device<TestBackend> = Default::default();
    let graph = GraphData::<TestBackend>::synthetic(N_DATA, N_HIDDEN, &device);
    let model = small_config()
        .init::<TestBackend>(&graph, &device)
        .unwrap()
        .map(&mut ZeroParams);

    let x = input(1, &device);
    let out = model.forward(x.clone());

    // [N_DATA, NUM_INPUT_CHANNELS], the t=MULTISTEP-1 slice.
    let x_skip = x
        .slice([
            0..1,
            MULTISTEP - 1..MULTISTEP,
            0..N_DATA,
            0..NUM_INPUT_CHANNELS,
        ])
        .reshape([N_DATA, NUM_INPUT_CHANNELS]);

    let column = |t: Tensor<TestBackend, 2>, cols: usize, c: usize| {
        t.slice([0..N_DATA, c..c + 1])
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_else(|_| panic!("column {c} of {cols}"))
    };

    for (&in_c, &out_c) in INPUT_PROGNOSTIC.iter().zip(&OUTPUT_PROGNOSTIC) {
        let want = column(x_skip.clone(), NUM_INPUT_CHANNELS, in_c);
        let got = column(out.clone(), NUM_OUTPUT_CHANNELS, out_c);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "output channel {out_c} row {i}: got {g}, want input channel {in_c} = {w}",
            );
        }
    }

    // Output channel 0 is named by no prognostic index, so it stays at zero.
    for (i, v) in column(out, NUM_OUTPUT_CHANNELS, 0).iter().enumerate() {
        assert!(v.abs() < 1e-4, "output channel 0 row {i}: got {v}, want 0");
    }
}

// Zeroes every float parameter, so the only surviving path through forward is the skip connection.
// Same trick as transformer_test.rs, and for the same reason: the sub-modules keep their fields
// private, so they cannot be zeroed tensor by tensor from here.
struct ZeroParams;

impl<B: Backend> ModuleMapper<B> for ZeroParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        Param::from_tensor(param.val().zeros_like())
    }
}
