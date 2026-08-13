use super::*;
use burn_store::ModuleSnapshot;

// Must be wgpu, not ndarray, for the same reason as common_test.rs: duplicate-index safety is a
// property of the kernel, not of Burn, and the decoder's mean destination degree is 3.
type TestBackend = burn::backend::wgpu::Wgpu;

// Small, deliberately distinct widths. hidden_dim (8) != in_channels_dst (5) != out_channels_dst
// (3) so that a mixed-up dimension cannot accidentally typecheck.
//
// edge_dim is no longer a config field -- the mapper derives it as EDGE_BASE + TRAINABLE = 6 from
// the graph it is handed, mirroring the real 3 + 8 = 11.
const IN_SRC: usize = 8; // must equal HIDDEN -- see param_paths_and_shapes_match_checkpoint
const IN_DST: usize = 5;
const OUT_DST: usize = 3;
const HIDDEN: usize = 8;
const HEADS: usize = 2;
const EDGE_BASE: usize = 4;
const TRAINABLE: usize = 2;

const N_SRC: usize = 3;
const N_DST: usize = 2;
const E: usize = 3;

fn small_config() -> GraphTransformerBackwardMapperConfig {
    GraphTransformerBackwardMapperConfig::new(
        IN_SRC, IN_DST, OUT_DST, HIDDEN, 2.0, HEADS, TRAINABLE,
    )
}

// arange rather than zeros so every layer norm sees a non-constant row.
fn ramp(rows: usize, cols: usize, device: &Device<TestBackend>) -> Tensor<TestBackend, 2> {
    Tensor::<TestBackend, 1, Int>::arange(0..(rows * cols) as i64, device)
        .float()
        .reshape([rows as i32, cols as i32])
        * 0.1
}

// The mapper reads its edges from GraphData at init rather than taking them per forward call, so
// the tests build one. Only the hidden -> data half is read by the backward mapper; the data ->
// hidden half and the node coordinates are correctly-ranked placeholders.
//
// dir_cols is a parameter because EDGE_BASE here is split 1 length + 3 dirs, keeping the derived
// edge_dim distinct from the other widths, whereas the real model is 1 + 2 = 3.
fn decoder_graph(
    num_hidden_nodes: usize,
    num_data_nodes: usize,
    src: &[i64],
    dst: &[i64],
    dir_cols: usize,
    device: &Device<TestBackend>,
) -> GraphData<TestBackend> {
    assert_eq!(src.len(), dst.len(), "src and dst must name the same edges");
    let edges = src.len();
    let ints = |v: &[i64]| {
        Tensor::<TestBackend, 1, Int>::from_data(TensorData::new(v.to_vec(), [v.len()]), device)
    };

    GraphData {
        data_x: Tensor::zeros([num_data_nodes, 2], device),
        hidden_x: Tensor::zeros([num_hidden_nodes, 2], device),

        // Unused by the backward mapper, present only to satisfy the struct.
        data_to_hidden_edge_index: Tensor::zeros([2, 1], device),
        data_to_hidden_edge_direction: Tensor::zeros([1, dir_cols], device),
        data_to_hidden_edge_length: Tensor::zeros([1, 1], device),

        hidden_to_data_edge_index: Tensor::stack::<2>(vec![ints(src), ints(dst)], 0),
        hidden_to_data_edge_direction: ramp(edges, dir_cols, device),
        hidden_to_data_edge_length: ramp(edges, 1, device),

        data_area_weight: Tensor::zeros([num_data_nodes, 1], device),
        num_data_nodes,
        num_hidden_nodes,
        // Coordinate width of data_x / hidden_x above: [lat, lon], as in the real graph.
        num_data_attr: 2,
        num_hidden_attr: 2,
    }
}

// Destination-sorted bipartite graph: src 0,1 -> dst 0 and src 2 -> dst 1.
fn small_graph(device: &Device<TestBackend>) -> GraphData<TestBackend> {
    decoder_graph(N_SRC, N_DST, &[0, 1, 2], &[0, 0, 1], EDGE_BASE - 1, device)
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
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        small_config().init(&small_graph(&device), &device);

    let out = mapper.forward(
        (ramp(N_SRC, IN_SRC, &device), ramp(N_DST, IN_DST, &device)),
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
    // Both the trainable tensor and edge_dim are sized from the graph rather than from config, so
    // the graph carries 4 edges rather than the real 1_626_240 -- allocating that in a unit test
    // buys nothing, and the only assertion that depends on it is `trainable.trainable [4, 8]`. Its
    // 1 length + 2 dirs is the real split, which is what makes lin_edge come out at 3 + 8 = 11.
    let graph = decoder_graph(4, 4, &[0, 1, 2, 3], &[0, 0, 1, 2], 2, &device);

    // in_src, in_dst, out_dst, hidden, mlp_ratio, heads, trainable.
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        GraphTransformerBackwardMapperConfig::new(1024, 224, 120, 1024, 4.0, 16, 8)
            .init(&graph, &device);

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
// i * edge_inc. The two must agree on batch-major ordering. graph_tranformer_conv asserts its node
// counts -- which the mapper scales as n_{src,dst}_base * batch_size -- against the node tensors it
// is handed, so a disagreement panics here rather than silently mixing batches together.
#[test]
fn batch_size_two_expands_edges_and_trainable() {
    let device = Default::default();
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        small_config().init(&small_graph(&device), &device);

    // Nodes are supplied for the whole batch; edges and edge attributes are per-batch and tiled.
    let out = mapper.forward(
        (
            ramp(2 * N_SRC, IN_SRC, &device),
            ramp(2 * N_DST, IN_DST, &device),
        ),
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
    // Three destinations, but no edge names destination 2.
    let graph = decoder_graph(N_SRC, 3, &[0, 1, 2], &[0, 0, 1], EDGE_BASE - 1, &device);
    let mapper: GraphTransformerBackwardMapper<TestBackend> =
        GraphTransformerBackwardMapperConfig::new(
            IN_SRC, IN_DST, OUT_DST, HIDDEN, 2.0, HEADS, TRAINABLE,
        )
        .init(&graph, &device);

    let x = (ramp(N_SRC, IN_SRC, &device), ramp(3, IN_DST, &device));

    // The edge attributes now live on the mapper, so perturbing them means mutating the field
    // between the two runs. Do NOT reach for `mapper.clone()` here: Burn initialises Param lazily,
    // so cloning before the first forward materialises re-draws every Linear, and the two runs
    // would differ for reasons that have nothing to do with the edges.
    let mut mapper = mapper;
    let base_edge_attr = mapper.edge_attr.clone();

    let baseline = mapper
        .forward(x.clone(), 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // Same graph and same weights, wildly different attributes on the edges into destinations
    // 0 and 1.
    mapper.edge_attr = base_edge_attr * 25.0 - 7.0;
    let perturbed = mapper
        .forward(x.clone(), 1)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

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

