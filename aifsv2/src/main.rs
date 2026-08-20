use burn::{backend::wgpu::Wgpu, cubecl::MemoryUsage, prelude::*, tensor::ElementConversion};
// CubeRuntime and Fusion are not re-exported through the burn facade, so the MemoryReport impls
// have to reach into the two internal crates directly. Both already resolve to the copies
// burn-wgpu uses, but all three versions have to move together on an upgrade.
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use burn_fusion::{Fusion, FusionBackend};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use chrono::{TimeZone, Utc};

use std::path::Path;

use crate::{
    aifs::AifsV2Config, graph::GraphData, metadata::Metadata, processors::Processors,
};

mod aifs;
mod block;
mod common;
mod decoder;
mod encoder;
mod forcings;
mod graph;
mod grib;
mod metadata;
mod named_node_attributes;
mod processors;
mod transformer;

type MyBackend = Wgpu;

const METADATA_DIR: &str = "./data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata";
const GRAPH_PATH: &str = "./data/aifs-single-mse-2.0_graph.safetensors";
const CHECKPOINT_PATH: &str = "./data/aifs-single-mse-2.0.safetensors";
// Datafiles. Fetched from ECMWF.
const OPER_PATH: &str = "./data/20260810000000-0h-oper-fc.grib2";
const WAVE_PATH: &str = "./data/20260810000000-0h-wave-fc.grib2";
// Native N320, so it needs no regrid: the lsm forcing channel and the apply-mask source at once.
const LSM_PATH: &str = "./data/lsm.grib";
// The 0.25 degree -> N320 interpolation operator, from scripts/fetch_regrid_matrix.py.
const REGRID_PATH: &str = "./data/regrid-0p25-to-n320.safetensors";

// config.model.num_channels, the latent width. Metadata does not parse it out of the raw JSON.
const NUM_CHANNELS: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device: Device<MyBackend> = Default::default();
    let metadata = Metadata::load(Path::new(METADATA_DIR))?;
    // No PyTorchToBurnAdapter here: these are raw arrays, not module weights, so the
    // adapter's [out, in] -> [in, out] transpose would corrupt them.
    let mut graph_store = SafetensorsStore::from_file(GRAPH_PATH);
    let graph_data = GraphData::<MyBackend>::from_safetensors_store(&mut graph_store, &device)?;

    // Same file as the model weights, different key namespace: these are the pre/post-processor
    // coefficient arrays, which are not module weights and so are read directly.
    let mut processor_store = SafetensorsStore::from_file(CHECKPOINT_PATH);
    let processors = Processors::<MyBackend>::load(&mut processor_store, &metadata, &device)?;

    smoke_tests::<MyBackend>(&graph_data, &processors, &metadata, &device)
}

// `MemoryReport` rather than `Backend`: forward_smoke_test reports the allocator, so the backend
// has to be one that can answer. That bound is the only thing keeping this from running on any
// backend at all.
fn smoke_tests<B: MemoryReport>(
    graph_data: &GraphData<B>,
    processors: &Processors<B>,
    metadata: &Metadata,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{} variables, {} in / {} out, {} grid points",
        metadata.variables.len(),
        metadata.model_input.full.len(),
        metadata.model_output.full.len(),
        metadata.latitudes.len(),
    );

    input_tensor_smoke_test::<B>(processors, metadata, device)?;

    let [_, num_encoder_edges] = graph_data.data_to_hidden_edge_index.shape().dims();
    let [_, num_decoder_edges] = graph_data.hidden_to_data_edge_index.shape().dims();
    println!(
        "{} data nodes, {} hidden nodes",
        graph_data.num_data_nodes, graph_data.num_hidden_nodes
    );
    println!("  data -> hidden: {num_encoder_edges} edges (encoder, CutOffEdges)");
    println!("  hidden -> data: {num_decoder_edges} edges (decoder, KNNEdges)");

    // The metadata and the graph have to describe the same grid, or every per-node lookup is
    // silently off.
    assert_eq!(
        metadata.latitudes.len(),
        graph_data.num_data_nodes,
        "metadata has {} grid points but the graph has {} data nodes",
        metadata.latitudes.len(),
        graph_data.num_data_nodes,
    );

    load_checkpoint(metadata, graph_data, device)?;
    forward_smoke_test::<B>(processors, metadata, device);
    forcings_smoke_test::<B>(metadata, device)?;

    Ok(())
}

// Build the real input tensor out of the GRIB files and report what came back.
//
// Nothing here feeds it to the model: at 542,080 data nodes a forward pass is ~27 GB against 18 GB
// of unified memory, which is why forward_smoke_test shrinks the grid instead. The tensor itself
// is the deliverable, so the checks are on its contents.
//
// Only one valid time is on disk -- everything is stepRange = 0 -- so the same pair of files fills
// both timesteps. tensor_from warns about that; the 5 retrieved forcings are constant in time and
// the 9 computed ones are evaluated per row, so only the 92 prognostics are stale.
fn input_tensor_smoke_test<B: Backend>(
    processors: &Processors<B>,
    metadata: &Metadata,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let regrid = grib::Regrid::load(REGRID_PATH)?;
    let oper = vec![OPER_PATH; metadata.multistep];
    let wave = vec![WAVE_PATH; metadata.multistep];

    let x = grib::tensor_from::<B>(&oper, &wave, LSM_PATH, metadata, &regrid, device)?;
    println!("\ninput tensor {:?}", x.shape().dims::<4>());

    // The imputer is the last thing that can remove a NaN, and it runs in pre. Everything the
    // model sees has to be finite; nothing before this point is required to be.
    //
    // is_nan, not the x != x trick: on wgpu the comparison reports 0 against a tensor that min and
    // max both agree is all-NaN, so it cannot be used to assert anything.
    let nan_count = |x: Tensor<B, 4>| x.is_nan().int().sum().into_scalar().elem::<i32>();
    let before = nan_count(x.clone());
    let after = nan_count(processors.pre(x.clone()).x);
    println!("  {before} NaN in physical units, {after} after pre-processing");
    assert_eq!(after, 0, "{after} NaN survived the imputer");

    let last = metadata.multistep - 1;
    let channel = |name: &str| metadata.var_to_input_channel[name];
    let column = |name: &str, time: usize| {
        let c = channel(name);
        x.clone()
            .slice([0..1, time..time + 1, 0..metadata.latitudes.len(), c..c + 1])
            .reshape([metadata.latitudes.len()])
    };

    for name in ["2t", "sp", "lsm", "swvl1", "cos_mwd", "insolation", "z_500"] {
        let values = column(name, last);
        println!(
            "  {name:<11} ch {:>3}  [{}, {}]",
            channel(name),
            values.clone().min().into_scalar(),
            values.max().into_scalar(),
        );
    }

    // The two timesteps must be distinguishable, and distinguishable in the right direction: a
    // constant_in_time channel is identical across them, a time-dependent forcing is not. This is
    // the assertion that catches a transposed time axis, which no shape check downstream would.
    let spread = |name: &str| {
        (column(name, 0) - column(name, last))
            .abs()
            .max()
            .into_scalar()
    };
    let (constant, varying) = (spread("lsm"), spread("insolation"));
    println!("  across the time axis: lsm moves {constant}, insolation moves {varying}");
    assert_eq!(
        constant.elem::<f32>(),
        0.0,
        "lsm is constant_in_time but differs by timestep"
    );
    assert!(
        varying.elem::<f32>() > 0.0,
        "insolation is identical at both timesteps"
    );

    regrid_oracle(&regrid)
}

// The regrid's whole-file check, which grib_test.rs cannot run: it needs the matrix on disk and
// both copies of the land-sea mask.
//
// The operational file carries lsm at 0.25 degrees and lsm.grib carries it natively on N320, so
// regridding the first has to land on the second -- but never exactly. Open data is a derived
// product: ECMWF interpolated the native field onto 0.25 degrees and repacked it at 8 bits, which
// leaves 129 distinct land fractions where the native file has 63,747. Interpolating back cannot
// recover what that quantisation dropped.
//
// What is left over is entirely coastal. Open ocean is exactly 0 and continental interior exactly
// 1 in both files, and neither the interpolation nor the repacking moves them; only a cell that is
// part land and part sea holds a fraction that can drift. A coastal point going from 0.42 to 0.61
// is therefore not a bug, so this cannot assert agreement -- it counts the points that swap side.
fn regrid_oracle(regrid: &grib::Regrid) -> Result<(), Box<dyn std::error::Error>> {
    let native = grib::load_grib(LSM_PATH)?.remove(0).values;

    let mut open_data = None;
    grib::for_each_field(Path::new(OPER_PATH), |field| {
        if field.short_name == "lsm" && field.level_type == "surface" {
            open_data = Some(field);
        }
        Ok(())
    })?;
    let open_data = open_data.ok_or("no lsm message in the operational file")?;
    let regridded = regrid.apply(&open_data.grid, &open_data.values)?;

    let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    let differing = native
        .iter()
        .zip(&regridded)
        .filter(|(a, b)| (**a - **b).abs() > 0.5)
        .count();
    println!(
        "  regridded lsm: land fraction {:.7} against {:.7} native, {differing} points \
         ({:.3}%) differ by more than half",
        mean(&regridded),
        mean(&native),
        100.0 * differing as f64 / native.len() as f64,
    );

    // 48 points swap side in practice, 0.009%. Dropping the longitude rotation moves 179,641 --
    // a third of the globe -- while leaving the land fraction untouched to five decimal places,
    // which is why the count is the assertion and the mean is only printed.
    assert!(
        differing * 50 < native.len(),
        "{differing} of {} points disagree -- more than coastlines",
        native.len()
    );
    Ok(())
}

// Nine of the fourteen forcings are pure functions of (date, lat, lon), so they can be computed
// on the real grid before any GRIB reading exists.
// Run them once to confirm they hold up at
// 542,080 points rather than at the five points forcings_test.rs checks against earthkit.
//
// The date is arbitrary -- there is no input data to match yet. A solstice at 00Z is the most
// informative choice: the day/night terminator is at its most extreme, and 00Z is the hour whose
// (hour - 12) hour angle is furthest negative.
fn forcings_smoke_test<B: Backend>(
    metadata: &Metadata,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let date = Utc.with_ymd_and_hms(2024, 6, 21, 0, 0, 0).unwrap();

    let to_tensor = |degrees: &[f64]| {
        let values: Vec<f32> = degrees.iter().map(|&v| v as f32).collect();
        Tensor::<B, 1>::from_floats(values.as_slice(), device)
    };
    let lat = to_tensor(&metadata.latitudes);
    let long = to_tensor(&metadata.longitudes);

    let forcings = forcings::compute_forcings(lat, long, &date)?;

    // Night is exactly 0, never negative -- earthkit clips, and the checkpoint was trained on
    // clipped values. A negative here would be a sign error that normalisation would happily
    // carry all the way into the forecast.
    let sunlit = forcings
        .insolation
        .clone()
        .greater_elem(0.0)
        .int()
        .sum()
        .into_scalar();

    println!(
        "\ncomputed forcings at {date} over {} grid points",
        metadata.latitudes.len(),
    );
    for (name, values) in [
        ("sin_latitude", forcings.sin_latitude),
        ("cos_latitude", forcings.cos_latitude),
        ("sin_longitude", forcings.sin_longitude),
        ("cos_longitude", forcings.cos_longitude),
        // The two julian_day rows are constant across the grid, so their range collapses.
        ("sin_julian_day", forcings.sin_julian_day),
        ("cos_julian_day", forcings.cos_julian_day),
        ("sin_local_time", forcings.sin_local_time),
        ("cos_local_time", forcings.cos_local_time),
        ("insolation", forcings.insolation),
    ] {
        // into_scalar gives B::FloatElem, which is only an Element -- it has no is_finite and no
        // ordering against f64. Every backend's float converts to f32, so compare there.
        let min = values.clone().min().into_scalar().elem::<f32>();
        let max = values.max().into_scalar().elem::<f32>();
        assert!(
            min.is_finite() && max.is_finite(),
            "{name} is not finite at {date}: [{min}, {max}]",
        );
        if name == "insolation" {
            assert!(min >= 0.0, "insolation went negative ({min}) at {date}");
            println!("  {name:<15} [{min}, {max}], {sunlit} points sunlit");
        } else {
            println!("  {name:<15} [{min}, {max}]");
        }
    }

    Ok(())
}

// Build the model at the checkpoint's real dimensions and load all 290 tensors into it. This is
// the only thing that proves our Burn field paths line up with the PyTorch key namespace.
//
// We deliberately do NOT run a forward on this model. The decoder's 1,626,240 edges projected to
// 1024 channels is a 6.7 GB f32 activation, and graph_tranformer_conv materialises q_i, k_j, v_j
// and msg at that size -- roughly 27 GB live, against 18 GB of unified memory. The encoder is
// ~12 GB on its own. See forward_smoke_test for the runnable version.
fn load_checkpoint<B: Backend>(
    metadata: &Metadata,
    graph_data: &GraphData<B>,
    device: &Device<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = AifsV2Config::from_metadata(metadata, NUM_CHANNELS);
    let mut model = config.init::<B>(graph_data, device);

    let mut store = aifs::checkpoint_store(CHECKPOINT_PATH);
    let result = model.load_from(&mut store)?;

    println!("\nloaded {CHECKPOINT_PATH}");
    println!(
        "  applied {}, missing {}, unused {}, skipped {}, errors {}",
        result.applied.len(),
        result.missing.len(), // burn-side paths not gotten data.
        result.unused.len(),  // store-side paths not sent data.
        result.skipped.len(),
        result.errors.len(),
    );

    // TODO(saiputravu): Remove this ai explanation.
    // Expected: 268 applied, 0 missing.
    //
    // `unused` over-reports at 100 and is not a useful signal here. It breaks down as
    // 16 + 2 + 82, of which only 22 are genuinely unused:
    //
    //   16  pre/post-processor normalizer arrays -- they belong to the pre/post-processing stage
    //       rather than the model. Genuinely unused.
    //    2  {encoder,decoder}.edge_inc -- we derive edge_inc from the graph. Genuinely unused.
    //   82  every LayerNorm weight/bias in the checkpoint. 4 of these are genuinely unused (the
    //       bare proc.layer_norm_attention, which anemoi aliases to layer_norm_attention_dest).
    //       The other 78 *are* applied, via the adapter's weight -> gamma alternative-name
    //       lookup, but never get credited: the applier inserts the module-side path (`...gamma`)
    //       into visited_paths and deliberately does not insert the alternative it resolved
    //       through (burn-store/src/applier.rs:180), while `unused` is the set of store-side keys
    //       (`...weight`) absent from visited_paths (applier.rs:104).
    //
    // So `unused` cannot distinguish "we never asked for this" from "we asked for it under a
    // different name". `missing` is the field to watch, and the spot check below is what confirms
    // the 78 actually arrived.
    if !result.missing.is_empty() {
        let mut missing: Vec<_> = result.missing.iter().map(|(path, _)| path).collect();
        missing.sort();
        println!("  missing:");
        for path in missing {
            println!("    {path}");
        }
    }
    for error in &result.errors {
        println!("  error: {error:?}");
    }

    spot_check(&model);

    Ok(())
}

// Matching key counts says the paths line up. It does not say the bytes arrived intact, which is
// the failure mode the adapter can produce: a [out, in] -> [in, out] transpose that silently
// became a reinterpret would keep every shape plausible and every count correct.
//
// So read three params back out of the loaded module and compare against values pulled straight
// from the safetensors file with numpy. One of each kind the adapter treats differently.
//
// This is ai checking the actual values of the params
// TODO(saiputravu): Remove this.
fn spot_check<B: Backend>(model: &aifs::AifsV2<B>) {
    // Expected values are checkpoint ground truth, read with scripts/parse_safetensors.py.
    const CASES: [(&str, &[usize], [f32; 4]); 3] = [
        // A Linear weight, stored [1024, 224] and held [224, 1024]. These are column 0 of the
        // PyTorch tensor, not row 0 -- had the adapter reinterpreted rather than transposed we
        // would see row 0, which is [-0.02591774, 0.01462184, -0.00610578, 0.04098628].
        (
            "encoder.emb_nodes_src.weight",
            &[224, 1024],
            [-0.02591774, 0.0060528, 0.0155118, -0.01580161],
        ),
        // A LayerNorm scale. Burn initialises gamma to ones, so anything but ones proves both the
        // weight -> gamma rename and the layer_norm_attention_dest -> _dst remap landed.
        (
            "encoder.proc.layer_norm_attention_dst.gamma",
            &[1024],
            [0.13032259, 0.1298658, 0.2181605, 0.18291621],
        ),
        // A bare Param the adapter must leave alone: [sin_lat, sin_lon, cos_lat, cos_lon] of grid
        // point 0. A transpose here would be catastrophic and completely invisible in the counts.
        (
            "named_attribute.latlons_data",
            &[542080, 4],
            [0.99999297, 0.0, 0.00375456, 1.0],
        ),
    ];

    let snapshots = model.collect(None, None, false);
    println!("  spot check:");
    for (path, shape, want) in CASES {
        let snapshot = snapshots
            .iter()
            .find(|s| s.full_path() == path)
            .unwrap_or_else(|| panic!("{path} not found in the loaded module"));

        assert_eq!(snapshot.shape, Shape::from(shape.to_vec()), "{path} shape",);

        let data = snapshot.to_data().expect("reading back a loaded param");
        let got = data.to_vec::<f32>().expect("param is not f32");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "{path}[{i}]: got {g}, want {w} from the checkpoint",
            );
        }
        println!("    {path} {shape:?} ok");
    }
}

// The same wiring at a size that fits: a small synthetic graph, a narrower latent, and fewer
// processor layers. The channel counts stay real (106 in, 120 out, multistep 2) so the input
// layout under test is the one the real model sees. Weights are the lazily-initialised random
// ones, not the checkpoint -- this exercises the chain, not the forecast.
fn forward_smoke_test<B: MemoryReport>(
    processors: &Processors<B>,
    metadata: &Metadata,
    device: &B::Device,
) {
    const NUM_DATA_NODES: usize = 64;
    const NUM_HIDDEN_NODES: usize = 16;
    const SMALL_NUM_CHANNELS: usize = 256;

    let graph_data = GraphData::<B>::synthetic(NUM_DATA_NODES, NUM_HIDDEN_NODES, device);
    let model = AifsV2Config::from_metadata(metadata, SMALL_NUM_CHANNELS)
        // 256 / 8 = 32 channels per head, satisfying the divisibility asserts in both the graph
        // transformer block and the processor's self-attention.
        .with_num_heads(8)
        .with_num_processor_layers(2)
        .with_num_processor_chunks(1)
        .init::<B>(&graph_data, device);

    // Stands in for the assembled GRIB input: [batch, time, grid, vars]. NaN in one imputed
    // channel, so the round trip through pre and post has something to carry: the imputer fills
    // it on the way in and has to put it back on the way out.
    let vars = metadata.model_input.full.len();
    let x = Tensor::<B, 4>::zeros([1, metadata.multistep, NUM_DATA_NODES, vars], device);
    let nan_channel = metadata.imputer_zero.iter().find_map(|name| {
        let input = metadata.input_channel(name)?;
        metadata.output_channel(name).map(|output| (name, input, output))
    });
    let x = match nan_channel {
        Some((_, input, _)) => x.slice_fill([0..1, 0..1, 0..1, input..input + 1], f32::NAN),
        None => x,
    };

    // Note: do not clone the model before this call. Burn's Param is lazily initialised, so a
    // clone re-draws every Linear.
    let before = B::memory_usage(device);
    let out = model.predict_step(processors, x);
    let after = B::memory_usage(device);

    let dims = out.shape().dims::<2>();
    assert_eq!(dims, [NUM_DATA_NODES, metadata.model_output.full.len()]);
    println!(
        "\nsynthetic predict_step ({NUM_DATA_NODES} data / {NUM_HIDDEN_NODES} hidden nodes, {SMALL_NUM_CHANNELS} channels)"
    );

    // The imputed point has to come back NaN, and only that point: the whole reason post takes
    // PreProcessed is to carry that one bit of state across the model.
    if let Some((name, _, output)) = nan_channel {
        let column = out
            .clone()
            .slice([0..NUM_DATA_NODES, output..output + 1])
            .into_data()
            .to_vec::<f32>()
            .expect("output column is not f32");
        assert!(column[0].is_nan(), "{name} was imputed but came back finite");
        assert!(
            column[1..].iter().all(|v| v.is_finite()),
            "{name} went NaN at a point that was never imputed",
        );
        println!("  {name} -> output channel {output}: NaN restored at 1 of {NUM_DATA_NODES} points");
    }

    println!("  out {:?}", dims);
    println!(
        "  device memory: {} reserved before, {} after (peak allocs {})",
        mib(before.bytes_reserved),
        mib(after.bytes_reserved),
        after.number_allocs,
    );
}

trait MemoryReport: Backend {
    fn memory_usage(device: &Device<Self>) -> MemoryUsage;
}

impl<B: FusionBackend + MemoryReport> MemoryReport for Fusion<B> {
    fn memory_usage(device: &Device<Self>) -> MemoryUsage {
        B::memory_usage(device)
    }
}

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> MemoryReport
    for CubeBackend<R, F, I, BT>
{
    fn memory_usage(device: &R::Device) -> MemoryUsage {
        R::client(device)
            .memory_usage()
            .expect("cubecl client did not report memory usage")
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}
