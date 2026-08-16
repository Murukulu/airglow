use burn::{backend::wgpu, prelude::*};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use chrono::{TimeZone, Utc};

use std::path::Path;

use crate::{aifs::AifsV2Config, graph::GraphData, metadata::Metadata};

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
mod transformer;

type MyBackend = wgpu::Wgpu;

const METADATA_DIR: &str = "./data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata";
const GRAPH_PATH: &str = "./data/aifs-single-mse-2.0_graph.safetensors";
const CHECKPOINT_PATH: &str = "./data/aifs-single-mse-2.0.safetensors";
const OPER_PATH: &str = "./data/20260810000000-0h-oper-fc.grib2";

// config.model.num_channels, the latent width. Metadata does not parse it out of the raw JSON.
const NUM_CHANNELS: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    crate::grib::load_grib(OPER_PATH);
    return Ok(());

    let device: Device<MyBackend> = Default::default();

    let metadata = Metadata::load(Path::new(METADATA_DIR))?;
    println!(
        "{} variables, {} in / {} out, {} grid points",
        metadata.variables.len(),
        metadata.model_input.full.len(),
        metadata.model_output.full.len(),
        metadata.latitudes.len(),
    );

    // No PyTorchToBurnAdapter here: these are raw arrays, not module weights, so the
    // adapter's [out, in] -> [in, out] transpose would corrupt them.
    let mut graph_store = SafetensorsStore::from_file(GRAPH_PATH);
    let graph_data = GraphData::<MyBackend>::from_safetensors_store(&mut graph_store, &device)?;

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

    load_checkpoint(&metadata, &graph_data, &device)?;
    forward_smoke_test(&metadata, &device);
    forcings_smoke_test(&metadata, &device)?;

    Ok(())
}

// Nine of the fourteen forcings are pure functions of (date, lat, lon), so they can be computed
// on the real grid before any GRIB reading exists. Run them once to confirm they hold up at
// 542,080 points rather than at the five points forcings_test.rs checks against earthkit.
//
// The date is arbitrary -- there is no input data to match yet. A solstice at 00Z is the most
// informative choice: the day/night terminator is at its most extreme, and 00Z is the hour whose
// (hour - 12) hour angle is furthest negative.
fn forcings_smoke_test(
    metadata: &Metadata,
    device: &Device<MyBackend>,
) -> Result<(), Box<dyn std::error::Error>> {
    let date = Utc.with_ymd_and_hms(2024, 6, 21, 0, 0, 0).unwrap();

    let to_tensor = |degrees: &[f64]| {
        let values: Vec<f32> = degrees.iter().map(|&v| v as f32).collect();
        Tensor::<MyBackend, 1>::from_floats(values.as_slice(), device)
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
        let min = values.clone().min().into_scalar();
        let max = values.max().into_scalar();
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
fn load_checkpoint(
    metadata: &Metadata,
    graph_data: &GraphData<MyBackend>,
    device: &Device<MyBackend>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = AifsV2Config::from_metadata(metadata, NUM_CHANNELS);
    let mut model = config.init::<MyBackend>(graph_data, device);

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
fn spot_check(model: &aifs::AifsV2<MyBackend>) {
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
fn forward_smoke_test(metadata: &Metadata, device: &Device<MyBackend>) {
    const NUM_DATA_NODES: usize = 64;
    const NUM_HIDDEN_NODES: usize = 16;
    const SMALL_NUM_CHANNELS: usize = 256;

    let graph_data = GraphData::<MyBackend>::synthetic(NUM_DATA_NODES, NUM_HIDDEN_NODES, device);
    let model = AifsV2Config::from_metadata(metadata, SMALL_NUM_CHANNELS)
        // 256 / 8 = 32 channels per head, satisfying the divisibility asserts in both the graph
        // transformer block and the processor's self-attention.
        .with_num_heads(8)
        .with_num_processor_layers(2)
        .with_num_processor_chunks(1)
        .init::<MyBackend>(&graph_data, device);

    // Stands in for the assembled GRIB input: [batch, time, grid, vars].
    let x = Tensor::<MyBackend, 4>::zeros(
        [
            1,
            metadata.multistep,
            NUM_DATA_NODES,
            metadata.model_input.full.len(),
        ],
        device,
    );

    // Note: do not clone the model before this call. Burn's Param is lazily initialised, so a
    // clone re-draws every Linear.
    let before = memory_usage(device);
    let out = model.forward(x);
    let after = memory_usage(device);

    let dims = out.shape().dims::<2>();
    assert_eq!(dims, [NUM_DATA_NODES, metadata.model_output.full.len()]);
    println!(
        "\nsynthetic forward ({NUM_DATA_NODES} data / {NUM_HIDDEN_NODES} hidden nodes, {SMALL_NUM_CHANNELS} channels)"
    );
    println!("  out {:?}, mean {}", dims, out.mean().into_scalar());
    println!(
        "  device memory: {} reserved before, {} after (peak allocs {})",
        mib(before.bytes_reserved),
        mib(after.bytes_reserved),
        after.number_allocs,
    );
}

// GPU allocator stats from the CubeCL memory pool. `bytes_reserved` is what the pool has claimed
// from the driver and will reuse, so it is a high-water mark rather than a live set -- which is
// the number that decides whether a given graph size fits.
fn memory_usage(device: &Device<MyBackend>) -> burn::cubecl::MemoryUsage {
    use burn::backend::wgpu::WgpuRuntime;
    use burn::cubecl::Runtime;

    WgpuRuntime::client(device)
        .memory_usage()
        .expect("wgpu client did not report memory usage")
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}
