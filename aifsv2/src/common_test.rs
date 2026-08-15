use super::*;

// Must be wgpu, not ndarray. Duplicate-index safety is a property of the kernel, not of Burn:
// burn-ndarray accumulates duplicates correctly under every primitive (one sequential host
// loop), so a CPU test would pass on an aggregation that is wrong on the backend we ship.
type TestBackend = burn::backend::wgpu::Wgpu;

fn assert_close(got: Vec<f32>, want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
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

// The graph worked through by hand in docs/graph-transformer-explained.md section 6, with a
// fourth destination added that has no incoming edges.
//
// Covers, in one test: both gathers, the 1/sqrt(C) scale, per-segment softmax over three
// distinct segments, duplicate-safe aggregation (destinations 0 and 2 have degree 3), the
// degree-1 case where a one-element softmax is exactly 1, and the zero-degree destination
// whose row is never gathered and so never divided by the 1e-16 guard.
#[test]
fn conv_matches_worked_example() {
    let device = Default::default();

    let query = Tensor::<TestBackend, 3>::from_floats(
        [[[1., 0.]], [[0., 1.]], [[1., 1.]], [[0., 0.]]],
        &device,
    ); // [4, 1, 2]
    let key = Tensor::<TestBackend, 3>::from_floats(
        [[[1., 0.]], [[0., 1.]], [[1., 1.]], [[2., 0.]]],
        &device,
    ); // [4, 1, 2]
    let value = Tensor::<TestBackend, 3>::from_floats(
        [[[10., 0.]], [[0., 10.]], [[5., 5.]], [[1., 1.]]],
        &device,
    ); // [4, 1, 2]
    let edges = Tensor::<TestBackend, 3>::zeros([7, 1, 2], &device);

    let edge_index_src = Tensor::from_ints([0, 1, 3, 2, 0, 2, 3], &device);
    let edge_index_dst = Tensor::from_ints([0, 0, 0, 1, 2, 2, 2], &device);

    let out = graph_tranformer_conv(
        query,
        key,
        value,
        edges,
        edge_index_src,
        edge_index_dst,
        4, // n_src
        4, // n_dst
    );
    assert_eq!(out.shape().dims::<3>(), [4, 1, 2]);

    // dst 0: degree 3. dst 1: degree 1, so out == v_j exactly. dst 2: degree 3.
    // dst 3: no incoming edges, so the accumulator row is never written.
    let want = [
        3.415917, 1.976257, //
        5.0, 5.0, //
        4.384432, 2.406672, //
        0.0, 0.0,
    ];
    assert_close(out.into_data().to_vec::<f32>().unwrap(), &want, 1e-4);
}

// The same projected edge tensor is added to BOTH the key and the value. The worked-example test
// runs with edges = 0 and so cannot see this at all.
//
// Two edges into one destination, H = 1 and C = 1 so the dot product is a plain multiply and
// 1/sqrt(C) = 1. key and value are zero, making k_j and v_j entirely the edge term:
//
//   k_j    = [1, 0]  ->  logits [1, 0]  ->  softmax [e/(1+e), 1/(1+e)]
//   v_j    = [1, 0]
//   out[0] = 0.7310586
//
// Dropping edges from the key gives logits [0, 0] -> 0.5; dropping them from the value gives
// 0.0. Both are far outside the tolerance.
#[test]
fn conv_adds_edges_to_key_and_value() {
    let device = Default::default();

    let query = Tensor::<TestBackend, 3>::from_floats([[[1.0]]], &device); // [1, 1, 1]
    let key = Tensor::<TestBackend, 3>::zeros([2, 1, 1], &device);
    let value = Tensor::<TestBackend, 3>::zeros([2, 1, 1], &device);
    let edges = Tensor::<TestBackend, 3>::from_floats([[[1.0]], [[0.0]]], &device); // [2, 1, 1]

    let edge_index_src = Tensor::from_ints([0, 1], &device);
    let edge_index_dst = Tensor::from_ints([0, 0], &device);

    let out = graph_tranformer_conv(
        query,
        key,
        value,
        edges,
        edge_index_src,
        edge_index_dst,
        2, // n_src
        1, // n_dst
    );
    assert_eq!(out.shape().dims::<3>(), [1, 1, 1]);
    assert_close(out.into_data().to_vec::<f32>().unwrap(), &[0.7310586], 1e-5);
}
