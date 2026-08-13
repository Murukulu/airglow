use super::*;
use burn::tensor::Int;

type TestBackend = burn::backend::wgpu::Wgpu;

// Regression test for get_qkve being handed attn_channels where it needs out_channels_conv.
// The einops line this ports splits the feature axis into (heads, out_channels_conv), so with
// attn_channels = 8 and num_heads = 2 the per-head width is 4, not 8.
//
// What makes the bug dangerous is that reshape([-1, heads, c]) only fails when the element
// count does not divide: at production scale [40320, 1024] against 16 * 1024 divides evenly
// and silently yields [2520, 16, 1024]. Here the source side does not divide (3 * 8 = 24
// against 2 * 8 = 16), so the old code trips Burn's reshape check instead. Red either way.
#[test]
fn block_forward_preserves_node_shapes() {
    let device = Default::default();

    // in, out, mlp hidden, heads, attn channels, edge dim, qk_norm, edge_pre_mlp.
    let block =
        GraphTransformerProcessorBlockConfig::new(8, 8, 16, 2, 8, 4, false, false).init(&device);

    // arange rather than zeros so the layer norms see non-constant rows.
    let x_src = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
        .float()
        .reshape([3, 8]);
    let x_dst = Tensor::<TestBackend, 1, Int>::arange(0..16, &device)
        .float()
        .reshape([2, 8]);
    let edge_attr = Tensor::<TestBackend, 1, Int>::arange(0..12, &device)
        .float()
        .reshape([3, 4]);

    let edge_index = EdgeIndex {
        src: Tensor::from_ints([0, 1, 2], &device),
        dst: Tensor::from_ints([0, 0, 1], &device),
        num_src: 3,
        num_dst: 2,
    };

    let (out_src, out_dst) = block.forward((x_src, x_dst), edge_attr, edge_index);

    assert_eq!(out_src.shape().dims::<2>(), [3, 8]);
    assert_eq!(out_dst.shape().dims::<2>(), [2, 8]);

    // Also covers the zero-variance layer norm path.
    for v in out_dst.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite(), "non-finite value in output: {}", v);
    }
}
