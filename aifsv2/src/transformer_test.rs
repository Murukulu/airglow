use super::*;
use burn::module::{ModuleMapper, Param};
use burn_store::ModuleSnapshot;

// wgpu rather than ndarray for consistency with the rest of the suite: the mapper tests need the
// real kernels for duplicate-index aggregation, and there is no reason for the processor to be
// tested on a different backend from the modules it sits between.
type TestBackend = burn::backend::wgpu::Wgpu;

// Small and deliberately distinct: N (4) != C (8) != HIDDEN (16), so a transposed or swapped
// dimension cannot accidentally typecheck. HEADS divides C, which MultiHeadAttentionConfig
// requires.
const N: usize = 4;
const C: usize = 8;
const HIDDEN: usize = 16;
const HEADS: usize = 2;
const WINDOW: usize = 2;

// num_channels, hidden_dim, window_size, num_heads.
fn small_block_config() -> TransformerProcessorBlockConfig {
    TransformerProcessorBlockConfig::new(C, HIDDEN, WINDOW, HEADS)
}

// arange rather than zeros so every layer norm sees a non-constant row.
fn ramp(rows: usize, cols: usize, device: &Device<TestBackend>) -> Tensor<TestBackend, 2> {
    Tensor::<TestBackend, 1, Int>::arange(0..(rows * cols) as i64, device)
        .float()
        .reshape([rows as i32, cols as i32])
        * 0.1
        + 1.0
}

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

// Zeroes every float parameter of whatever module it is applied to. Used to switch off a whole
// residual branch without reaching into another module's private fields -- MultiLayerPreceptron
// keeps `layers` private to common.rs, so it cannot be zeroed tensor by tensor from here.
struct ZeroParams;

impl<B: Backend> ModuleMapper<B> for ZeroParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        Param::from_tensor(param.val().zeros_like())
    }
}

// The most basic contract a transformer block has: [N, C] in, [N, C] out.
//
// Pins the squeeze at transformer.rs:57. The original `flatten(0, 0)` merged dims 0..=0, which
// merges nothing, so a 3-dim result was handed to a `Tensor<B, 2>` binding and TensorCheck::flatten
// panicked. It compiled because the rank is inferred from the `+` rather than derived from the
// shape arithmetic, so only a test that actually calls forward can catch it.
#[test]
fn block_forward_preserves_shape() {
    let device = Default::default();
    let block: TransformerProcessorBlock<TestBackend> = small_block_config().init(&device);

    let out = block.forward(ramp(N, C, &device));

    assert_eq!(out.shape().dims::<2>(), [N, C]);
    for v in out.into_data().to_vec::<f32>().unwrap() {
        assert!(v.is_finite(), "non-finite value in output: {}", v);
    }
}

// Both residuals, pinned without needing a reference implementation.
//
// Zeroing the attention's output projection and the whole MLP makes both branch outputs identically
// zero, whatever the norms and the remaining weights do. A block with both skips is then exactly
// the identity; a block missing the MLP skip returns exactly zero. Two maximally separated
// outcomes, so the assertion needs no tolerance argument to be convincing.
//
// The bug this catches is transformer.rs returning `mlp(layer_norm_mlp(x))` where upstream
// (block.py:126-131) is `x + mlp(layer_norm_mlp(x))`. Shapes are unaffected either way, so nothing
// else in the file would notice.
#[test]
fn block_forward_keeps_both_residuals() {
    let device = Default::default();
    let mut block: TransformerProcessorBlock<TestBackend> = small_block_config().init(&device);

    block.attention.projection = block.attention.projection.clone().map(&mut ZeroParams);
    block.mlp = block.mlp.clone().map(&mut ZeroParams);

    let x = ramp(N, C, &device);
    let want = x.clone().into_data().to_vec::<f32>().unwrap();

    // Without this the test would also pass on a block that returns zeros for a zero input.
    assert!(
        want.iter().any(|v| v.abs() > 1e-6),
        "input is all zeros -- the test proves nothing"
    );

    let got = block.forward(x).into_data().to_vec::<f32>().unwrap();

    assert_close(got, &want, 1e-5);
}

// Blocks within a chunk must be independent modules.
//
// The trap is `vec![expr; n]`, which evaluates once and Clones the result. Param::clone preserves
// the ParamId (burn-core-0.21.0/src/module/param/base.rs:439-460), so a chunk built that way holds
// one block repeated, sharing parameter identities and values. transformer.rs now clones the
// *config* and calls init per block, which is the fix -- this test is what keeps it that way.
//
// A checkpoint load would overwrite the values, so this would not be a wrong-numbers bug on the
// path we care about. It is wrong on every other path -- fresh init, Module::map, Module::visit.
//
// attention.lin_q.weight rather than a layer norm: gamma is initialised to ones, so it is
// legitimately identical across blocks and could not tell the two cases apart.
#[test]
fn chunk_blocks_are_independently_initialised() {
    let device = Default::default();
    let chunk: TransformerProcessorChunk<TestBackend> =
        TransformerProcessorChunkConfig::new(C, 2, WINDOW)
            .with_num_heads(HEADS)
            .init(&device);

    assert_eq!(chunk.blocks.len(), 2);

    assert_ne!(
        chunk.blocks[0].attention.lin_q.weight.id.val(),
        chunk.blocks[1].attention.lin_q.weight.id.val(),
        "blocks 0 and 1 share a ParamId, so the chunk holds one block cloned twice"
    );

    let a = chunk.blocks[0]
        .attention
        .lin_q
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let b = chunk.blocks[1]
        .attention
        .lin_q
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(
        a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-9),
        "blocks 0 and 1 were initialised to identical weights"
    );
}

// The processor holds num_layers blocks in total, spread over num_chunks chunks -- at the real
// config, 2 chunks of 8, not 16 chunks of 16.
//
// The bug this catches is spending num_layers twice: once as the blocks-per-chunk handed to
// TransformerProcessorChunkConfig and once as the chunk count, which gives 16 x 16 = 256 blocks
// against the checkpoint's 208 tensors. Upstream keeps the two apart, computing
// chunk_size = num_layers // num_chunks (processor.py:53).
//
// Uses the real 16 and 2 so the arithmetic under test is the arithmetic that ships.
#[test]
fn processor_splits_num_layers_across_num_chunks() {
    const LAYERS: usize = 16;
    const CHUNKS: usize = 2;

    let device = Default::default();
    // num_channels, num_layers, num_chunks, num_heads, window_size.
    let processor: TransformerProcessor<TestBackend> =
        TransformerProcessorConfig::new(C, LAYERS, CHUNKS, HEADS, WINDOW).init(&device);

    assert_eq!(processor.proc.len(), CHUNKS);
    for (i, chunk) in processor.proc.iter().enumerate() {
        assert_eq!(
            chunk.blocks.len(),
            LAYERS / CHUNKS,
            "chunk {} holds {} blocks, expected {}",
            i,
            chunk.blocks.len(),
            LAYERS / CHUNKS
        );
    }

    let total: usize = processor.proc.iter().map(|c| c.blocks.len()).sum();
    assert_eq!(total, LAYERS);
}

// An uneven split would silently drop the remainder -- 16 layers over 5 chunks gives 5 chunks of 3,
// so the processor would hold 15 blocks and quietly disagree with the checkpoint. Anemoi asserts
// the same at processor.py:57-59.
#[test]
#[should_panic(expected = "has to be divisible by")]
fn processor_rejects_layers_not_divisible_by_chunks() {
    let device = Default::default();
    let _: TransformerProcessor<TestBackend> =
        TransformerProcessorConfig::new(C, 16, 5, HEADS, WINDOW).init(&device);
}

// Pins checkpoint loadability: the exact set of Burn parameter paths and shapes for one block at
// production dimensions, against the 13 `model.processor.proc.{c}.blocks.{b}.*` keys recorded in
// docs/transformer-processor-review.md section 1.
//
// One block rather than the whole processor. The full 16 blocks would be ~209M parameters and
// would pin nothing that 13 tensors at ~13M do not -- the per-block key set is identical for all
// of them.
//
// RED against the code as it stands -- review section 3.4. Burn's MultiHeadAttention names its
// projections query/key/value/output where the checkpoint says lin_q/lin_k/lin_v/projection, and
// gives all four a bias where the checkpoint has none on q/k/v (qkv_bias=False,
// attention.py:45,116-118). Three orphan parameters per block, 48 across the model.
//
// Two things to know when reading the expectations:
//   * Burn stores Linear weights transposed relative to PyTorch, so [in, out] here against
//     [out, in] in the checkpoint. PyTorchToBurnAdapter handles that at load, along with the
//     LayerNorm weight/bias -> gamma/beta rename; neither is a defect.
//   * These are Burn paths. The remaps still needed are recorded in review section 4.
#[test]
fn block_param_paths_and_shapes_match_checkpoint() {
    let device = Default::default();
    // num_channels, hidden_dim (= mlp_hidden_ratio 4 x 1024), window_size, num_heads.
    let block: TransformerProcessorBlock<TestBackend> =
        TransformerProcessorBlockConfig::new(1024, 4096, 1120, 16).init(&device);

    let mut got: Vec<String> = block
        .collect(None, None, true)
        .iter()
        .map(|t| format!("{} {:?}", t.full_path(), t.shape.to_vec()))
        .collect();
    got.sort();

    let want = [
        "attention.lin_k.weight [1024, 1024]",
        "attention.lin_q.weight [1024, 1024]",
        "attention.lin_v.weight [1024, 1024]",
        "attention.projection.bias [1024]",
        "attention.projection.weight [1024, 1024]",
        "layer_norm_attention.beta [1024]",
        "layer_norm_attention.gamma [1024]",
        "layer_norm_mlp.beta [1024]",
        "layer_norm_mlp.gamma [1024]",
        "mlp.layers.0.bias [4096]",
        "mlp.layers.0.weight [1024, 4096]",
        "mlp.layers.1.bias [1024]",
        "mlp.layers.1.weight [4096, 1024]",
    ];

    assert_eq!(got, want);
}
