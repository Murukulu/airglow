use super::*;
use burn_store::ModuleSnapshot;

// Must be wgpu, not ndarray, for the same reason as common_test.rs: duplicate-index safety is a
// property of the kernel, not of Burn, and the decoder's mean destination degree is 3.
type TestBackend = burn::backend::wgpu::Wgpu;

// Small, deliberately distinct widths. hidden_dim (8) != in_channels_dst (5) != out_channels_dst
// (3) so that a mixed-up dimension cannot accidentally typecheck, and edge_dim (6) is split as
// 4 base attributes + 2 trainable, mirroring the real 3 + 8 = 11.
const IN_SRC: usize = 8; // must equal HIDDEN -- see param_paths_and_shapes_match_checkpoint
const IN_DST: usize = 5;
const OUT_DST: usize = 3;
const HIDDEN: usize = 8;
const HEADS: usize = 2;
const ATTN: usize = 8;
const EDGE_BASE: usize = 4;
const TRAINABLE: usize = 2;
const EDGE_DIM: usize = EDGE_BASE + TRAINABLE;

const N_SRC: usize = 3;
const N_DST: usize = 2;
const E: usize = 3;

fn small_config() -> GraphTransformerBackwardMapperConfig {
    GraphTransformerBackwardMapperConfig::new(
        IN_SRC, IN_DST, OUT_DST, HIDDEN, 2.0, HEADS, ATTN, EDGE_DIM, E, TRAINABLE,
    )
}

// arange rather than zeros so every layer norm sees a non-constant row.
fn ramp(rows: usize, cols: usize, device: &Device<TestBackend>) -> Tensor<TestBackend, 2> {
    Tensor::<TestBackend, 1, Int>::arange(0..(rows * cols) as i64, device)
        .float()
        .reshape([rows as i32, cols as i32])
        * 0.1
}

// Destination-sorted bipartite graph: src 0,1 -> dst 0 and src 2 -> dst 1.
fn small_edges(device: &Device<TestBackend>) -> EdgeIndex<TestBackend> {
    EdgeIndex {
        src: Tensor::from_ints([0, 1, 2], device),
        dst: Tensor::from_ints([0, 0, 1], device),
        num_src: N_SRC,
        num_dst: N_DST,
    }
}

fn edge_inc(
    n_src: usize,
    n_dst: usize,
    device: &Device<TestBackend>,
) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_ints([[n_src as i64], [n_dst as i64]], device)
}

// The load-bearing shape contract of the backward mapper: it consumes a pair and returns a single
// destination-domain tensor at out_channels_dst, NOT at hidden_dim.
//
// Catches post_process being skipped (output would be [N_DST, HIDDEN] = [2, 8]), applied before the
// block instead of after, or out_channels_dst wired into the wrong Linear. It also pins the input
// contract that x_src arrives at hidden_dim (8) and not at in_channels_dst (5): the decoder has no
// emb_nodes_src, so lin_key is Linear(hidden_dim, _) and consumes x_src directly.
#[test]
fn forward_maps_dst_to_out_channels_dst() {
    let device = Default::default();
    let mapper: GraphTransformerBackwardMapper<TestBackend> = small_config().init(&device);

    let out = mapper.forward(
        (ramp(N_SRC, IN_SRC, &device), ramp(N_DST, IN_DST, &device)),
        ramp(E, EDGE_BASE, &device),
        small_edges(&device),
        edge_inc(N_SRC, N_DST, &device),
        1,
    );

    assert_eq!(out.shape().dims::<2>(), [N_DST, OUT_DST]);
    for v in out.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite(), "non-finite value in output: {}", v);
    }
}

// Pins checkpoint loadability: the exact set of Burn parameter paths and shapes, built with the
// real model dimensions, against the 32 `model.decoder.*` keys in
// data/aifs-single-mse-2.0.safetensors.
//
// Being an EXACT set comparison, this also catches structural drift in either direction -- adding an
// emb_nodes_src (the forward mapper's field, which the backward mapper must not have) fails here,
// as does dropping node_data_extractor_norm.
//
// Two things to know when reading the expectations:
//   * Burn stores Linear weights transposed relative to PyTorch, so [in, out] here against
//     [out, in] in the checkpoint. PyTorchToBurnAdapter handles that at load; it is not a defect.
//   * The paths below are Burn's, not the checkpoint's. The remaps still needed are recorded in
//     docs/graph-transformer-backward-mapper-review.md section 2.1 and 2.7.
#[test]
fn param_paths_and_shapes_match_checkpoint() {
    let device = Default::default();
    // in_src, in_dst, out_dst, hidden, mlp_ratio, heads, attn, edge_dim, num_edges, trainable.
    // num_edges is 4 rather than the real 1_626_240 -- allocating the real trainable tensor in a
    // unit test buys nothing, and no assertion below depends on it.
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        GraphTransformerBackwardMapperConfig::new(1024, 224, 120, 1024, 4.0, 16, 1024, 11, 4, 8)
            .init(&device);

    let mut got: Vec<String> = mapper
        .collect(None, None, true)
        .iter()
        .map(|t| format!("{} {:?}", t.full_path(), t.shape.to_vec()))
        .collect();
    got.sort();

    let want = [
        "emb_nodes_dst.bias [1024]",
        "emb_nodes_dst.weight [224, 1024]",
        "node_data_extractor.bias [120]",
        "node_data_extractor.weight [1024, 120]",
        "node_data_extractor_norm.beta [1024]",
        "node_data_extractor_norm.gamma [1024]",
        "proc.layer_norm_attention_dst.beta [1024]",
        "proc.layer_norm_attention_dst.gamma [1024]",
        "proc.layer_norm_attention_src.beta [1024]",
        "proc.layer_norm_attention_src.gamma [1024]",
        "proc.layer_norm_mlp_dst.beta [1024]",
        "proc.layer_norm_mlp_dst.gamma [1024]",
        "proc.lin_edge.bias [1024]",
        "proc.lin_edge.weight [11, 1024]",
        "proc.lin_key.bias [1024]",
        "proc.lin_key.weight [1024, 1024]",
        "proc.lin_query.bias [1024]",
        "proc.lin_query.weight [1024, 1024]",
        "proc.lin_self.bias [1024]",
        "proc.lin_self.weight [1024, 1024]",
        "proc.lin_value.bias [1024]",
        "proc.lin_value.weight [1024, 1024]",
        "proc.node_dst_mlp.layers.0.bias [4096]",
        "proc.node_dst_mlp.layers.0.weight [1024, 4096]",
        "proc.node_dst_mlp.layers.1.bias [1024]",
        "proc.node_dst_mlp.layers.1.weight [4096, 1024]",
        "proc.projection.bias [1024]",
        "proc.projection.weight [1024, 1024]",
        "trainable.trainable [4, 8]",
    ];

    assert_eq!(got, want);
}

// The only place graph::expand_edges and TrainableTensor::forward are exercised together, and
// neither mapper had such a test.
//
// TrainableTensor tiles the base edge attributes batch_size times along dim 0 and concatenates the
// tiled trainable block along dim 1; expand_edges tiles the edge list and offsets each copy by
// i * edge_inc, with graph::cat summing num_src / num_dst across copies. The two must agree on
// batch-major ordering. graph_tranformer_conv asserts its node counts against edge_index.num_src /
// num_dst, so a disagreement panics here rather than silently mixing batches together.
#[test]
fn batch_size_two_expands_edges_and_trainable() {
    let device = Default::default();
    let mapper: GraphTransformerBackwardMapper<TestBackend> = small_config().init(&device);

    // Nodes are supplied for the whole batch; edges and edge attributes are per-batch and tiled.
    let out = mapper.forward(
        (
            ramp(2 * N_SRC, IN_SRC, &device),
            ramp(2 * N_DST, IN_DST, &device),
        ),
        ramp(E, EDGE_BASE, &device),
        small_edges(&device),
        edge_inc(N_SRC, N_DST, &device),
        2,
    );

    assert_eq!(out.shape().dims::<2>(), [2 * N_DST, OUT_DST]);
    for v in out.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite(), "non-finite value in output: {}", v);
    }
}

// A destination with no incoming edges must be finite, and must be unaffected by the other
// destinations' edges.
//
// sparse_segment_softmax shifts by a GLOBAL max rather than a per-segment one (common.rs:105-115),
// which is exact only because the shift cancels in the ratio. This test is what that claim rests
// on at the mapper level: changing the attributes of edges that feed destinations 0 and 1 moves the
// global max, and destination 2's output row must not move with it. Its accumulator row is never
// written by select_assign and its denominator row is never gathered, so it must also never reach
// the 1e-16 guard and emit a NaN.
#[test]
fn zero_degree_destination_is_isolated_and_finite() {
    let device = Default::default();
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        GraphTransformerBackwardMapperConfig::new(
            IN_SRC, IN_DST, OUT_DST, HIDDEN, 2.0, HEADS, ATTN, EDGE_DIM, E, TRAINABLE,
        )
        .init(&device);

    // Three destinations, but no edge names destination 2.
    let edges = EdgeIndex {
        src: Tensor::from_ints([0, 1, 2], &device),
        dst: Tensor::from_ints([0, 0, 1], &device),
        num_src: N_SRC,
        num_dst: 3,
    };

    let x = (ramp(N_SRC, IN_SRC, &device), ramp(3, IN_DST, &device));
    let inc = edge_inc(N_SRC, 3, &device);

    let run = |edge_attr: Tensor<TestBackend, 2>| {
        mapper
            .forward(x.clone(), edge_attr, edges.clone(), inc.clone(), 1)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };

    let baseline = run(ramp(E, EDGE_BASE, &device));
    // Same graph, wildly different edge attributes on the edges into destinations 0 and 1.
    let perturbed = run(ramp(E, EDGE_BASE, &device) * 25.0 - 7.0);

    assert_eq!(baseline.len(), 3 * OUT_DST);
    for v in baseline.iter().chain(perturbed.iter()) {
        assert!(v.is_finite(), "non-finite value in output: {}", v);
    }

    // Row 2 is the zero-degree destination. Rows 0 and 1 are expected to differ.
    for i in 0..OUT_DST {
        let (b, p) = (baseline[2 * OUT_DST + i], perturbed[2 * OUT_DST + i]);
        assert!(
            (b - p).abs() < 1e-6,
            "zero-degree destination leaked edge information at column {}: {} vs {}",
            i,
            b,
            p
        );
    }
    assert!(
        (0..OUT_DST).any(|i| (baseline[i] - perturbed[i]).abs() > 1e-6),
        "destination 0 did not respond to its own edge attributes -- the test proves nothing"
    );
}
