# Review: `TransformerProcessor`, as implemented

Companion to [`graph-transformer-forward-mapper-review.md`](./graph-transformer-forward-mapper-review.md)
and [`graph-transformer-backward-mapper-review.md`](./graph-transformer-backward-mapper-review.md),
which review the two mappers. This one reviews `src/transformer.rs` — the processor that sits
between them — against upstream: what is right, what is wrong, and what is missing.

Tracked by [airglow#17](https://github.com/Murukulu/airglow/issues/17), whose
[comment 5231875811](https://github.com/Murukulu/airglow/issues/17#issuecomment-5231875811) dumps
the 208 `model.processor.*` keys that are the target to hit.

Upstream is pinned to **anemoi-core `b666d5bf`** (`models-0.9.3`) — the checkout at
`~/Documents/projects/anemoi-core`, and the version the checkpoint was trained with. Anemoi paths
are relative to `models/src/anemoi/models/`; repo paths to `aifsv2/`.

Corroborating evidence: the safetensors keys from the issue comment (§1), the processor block of
`data/aifs-single-mse-2.0_metadata.json` (§1), and the burn 0.21 sources under
`~/.cargo/registry/src/index.crates.io-*/`.

The headline as first written: **three findings were load-bearing.** `forward` panicked on its first
call (§3.2); the processor built 256 blocks where the checkpoint has 16 (§3.1); and the MLP residual
was missing, which was silent (§3.3). Beyond those, the module cannot carry these weights at all
while it is built on Burn's `MultiHeadAttention` (§3.4), and it does not implement the sliding
window the checkpoint was trained with (§3.5).

> **Status.** All three blockers are fixed, along with §3.4 and §3.6 through §3.10. **The block's
> parameter tree now matches the checkpoint exactly, so nothing structural stands between this
> module and a load.** What remains is **§3.5** (windowed attention — the numbers
> are wrong at any grid wider than the band, and it is ~18x the compute it should be). §3.4.1,
> §3.11 and §3.12 are notes, not work. Each finding
> below carries its own status line; the test table in §5 is the machine-checkable version of the
> same claim.

---

## 1. Checkpoint ground truth

`config.model.processor` in `data/aifs-single-mse-2.0_metadata.json`:

```
_target_: anemoi.models.layers.processor.TransformerProcessor
num_layers: 16          num_chunks: 2           num_heads: 16
mlp_hidden_ratio: 4     window_size: 1120       qk_norm: false
dropout_p: 0.0          softcap: 0.0            use_alibi_slopes: false
use_rotary_embeddings: false
attention_implementation: flash_attention
Activation: torch.nn.GELU   LayerNorm: torch.nn.LayerNorm   Linear: torch.nn.Linear
```

With `num_channels: 1024`: **2 chunks × 8 blocks = 16 blocks**, MLP hidden width `4 × 1024 = 4096`,
`head_dim = 1024 / 16 = 64`.

Thirteen keys per block:

```
proc.{c}.blocks.{b}.attention.lin_q.weight       [1024, 1024]      <- no bias
proc.{c}.blocks.{b}.attention.lin_k.weight       [1024, 1024]      <- no bias
proc.{c}.blocks.{b}.attention.lin_v.weight       [1024, 1024]      <- no bias
proc.{c}.blocks.{b}.attention.projection.weight  [1024, 1024]
proc.{c}.blocks.{b}.attention.projection.bias    [1024]
proc.{c}.blocks.{b}.layer_norm_attention.weight  [1024]
proc.{c}.blocks.{b}.layer_norm_attention.bias    [1024]
proc.{c}.blocks.{b}.layer_norm_mlp.weight        [1024]
proc.{c}.blocks.{b}.layer_norm_mlp.bias          [1024]
proc.{c}.blocks.{b}.mlp.0.weight                 [4096, 1024]      <- Linear
proc.{c}.blocks.{b}.mlp.0.bias                   [4096]
proc.{c}.blocks.{b}.mlp.2.weight                 [1024, 4096]      <- Linear ( .1 is GELU )
proc.{c}.blocks.{b}.mlp.2.bias                   [1024]
```

`16 × 13 = 208`, which is exactly the "208 processor layer weights" that
`scripts/ckpt_to_safetensors.py` reports in its docstring. Nothing else lives under
`model.processor.*` — no `edge_inc`, no `trainable`, no graph buffers. **The processor is the one
module of the three that is fully described by the safetensors file**, and so the one that can be
loaded without extending the conversion script.

**`lin_q`, `lin_k` and `lin_v` carry no bias.** `MultiHeadSelfAttention.__init__` builds them with
`bias=qkv_bias` and `qkv_bias` defaults to `False` (`attention.py:45, 116-118`), which
`TransformerProcessorBlock` never overrides (`block.py:66-102`). Only `projection` is biased
(`attention.py:120`). The key list above confirms it: three weights, no biases.

---

## 2. Checked and correct

Recorded so it does not get re-litigated.

- **Pre-norm placement.** `layer_norm_attention` before the attention and `layer_norm_mlp` before
  the MLP (`transformer.rs:61, 68`) matches `block.py:123-128`. The port has the norms on the right
  side of the residual branch; what it is missing is one of the residuals themselves (§3.3).
- **GELU is the right activation and the right GELU.** `layer_kernels.Activation` is
  `torch.nn.GELU`, whose default is `approximate='none'` — the exact erf form — and Burn's `Gelu`
  is likewise the exact erf form. No `tanh` approximation mismatch to worry about.
- **LayerNorm epsilon agrees by default.** `LayerNormConfig` defaults `epsilon = 1e-5`
  (`burn-nn-0.21.0/src/modules/norm/layer.rs:19-21`), which is `torch.nn.LayerNorm`'s default. No
  override needed, and `transformer.rs:36-37` correctly does not add one.
- **The attention scale agrees.** flash-attn's default `softmax_scale` is `1/sqrt(head_dim)` with
  `head_dim = 64`; Burn divides by `sqrt(d_k)` where `d_k = d_model / n_heads = 64`
  (`mha.rs`, `attn_scores`). Whatever else is wrong with the attention, the scale is not.
- **`PyTorchToBurnAdapter` covers the parameter-level differences already.** It renames LayerNorm
  `weight`/`bias` to `gamma`/`beta` and transposes Linear weights from `[out, in]` to `[in, out]`
  (`burn-store-0.21.0/src/adapter.rs:318-337`). So the load work is entirely about _path_ names —
  see §3.4 and §3.10 — and not about layouts or parameter names.
- **Dropping `dropout_p`, `softcap`, `use_alibi_slopes`, `qk_norm` and `use_rotary_embeddings` is
  faithful.** The checkpoint sets every one of them to off or zero (§1). The
  `// We do not implement or use q_norm, k_norm` comment (`transformer.rs:21`) is correct _for this
  checkpoint_, and is worth keeping in that form rather than as an unqualified claim.
- **Three levels of nesting is the right shape.** Processor → chunk → block mirrors
  `TransformerProcessor` / `TransformerProcessorChunk` / `TransformerProcessorBlock`, and the
  checkpoint's `proc.{c}.blocks.{b}` path confirms both levels are real rather than an artefact of
  gradient checkpointing. Keep them; only the counts are wrong (§3.1).

---

## 3. Findings

Ordered by how much they will hurt.

### 3.1 `TransformerProcessorConfig::init` conflates chunk count with layer count

**Status: fixed.** `TransformerProcessorConfig` now carries `num_chunks`, `init` builds `num_chunks`
chunks of `chunk_size()` blocks, and the divisibility assert is ported. Pinned by
`processor_splits_num_layers_across_num_chunks` and
`processor_rejects_layers_not_divisible_by_chunks`.

As originally found:

```rust
let proc = vec![
    TransformerProcessorChunkConfig::new(self.conf.clone(), self.num_layers)
        .init(device);
    self.num_layers
];
```

`self.num_layers` is spent twice — once as the blocks-per-chunk passed into the chunk config, and
once as the number of chunks. At the real config (`num_layers: 16`) that is **16 chunks of 16
blocks = 256 blocks**, against the checkpoint's 2 × 8 = 16. Sixteen times the parameters, and no
arrangement of remaps makes those 256 blocks line up with 208 tensors.

Upstream splits the two explicitly. `BaseProcessor.__init__` computes
`self.chunk_size = num_layers // num_chunks` (`processor.py:53`) and asserts divisibility
(`processor.py:57-59`); `build_layers` then constructs `self.num_chunks` chunks
(`processor.py:65-75`), each given `num_layers=self.chunk_size` (`processor.py:154-167`).

`TransformerProcessorConfig` has no `num_chunks` field at all, so the distinction cannot currently
be expressed.

**Fix.** Add `num_chunks`, derive `chunk_size = num_layers / num_chunks`, port the divisibility
assert, and build `num_chunks` chunks of `chunk_size` blocks.

### 3.2 `forward` panics on its first call

**Status: fixed** — now `attn.context.squeeze::<2>()`, which drops the length-1 batch dimension and
asserts that it was length 1. Pinned by `block_forward_preserves_shape`.

As originally found:

```rust
let x = x + attn.context.flatten(0, 0);
```

`MhaOutput::context` is `[batch_size, seq_length, d_model]` — here `[1, N, C]`. Flattening dims
`0..=0` merges a single dimension with itself, so the result still has three dimensions, while the
`+` against `x: Tensor<B, 2>` forces the destination rank to 2. `TensorCheck::flatten` registers

> The destination dimension (2) must be large enough to accommodate the flattening operation.

whenever `D2 < D1 - (end_dim - start_dim)`, i.e. `2 < 3 - 0`
(`burn-tensor-0.21.0/src/tensor/api/check.rs:270-278`), and `check!` panics.

This compiles, which is why it survived — the rank is a const generic that is _inferred_ from the
addition rather than checked against the shape arithmetic, and the shape arithmetic is only
validated at runtime. `cargo check` cannot see it and nothing calls `forward`.

**Fix.** `flatten(0, 1)` merges `[1, N, C]` to `[N, C]`. `.squeeze::<2>(0)` says the same thing more
directly and asserts the leading extent is 1, which is the actual precondition while the block
hardcodes batch 1 (§3.12).

### 3.3 The MLP residual is dropped

**Status: fixed** — now `x.clone() + self.mlp.forward(self.layer_norm_mlp.forward(x))`. Pinned by
`block_forward_keeps_both_residuals`.

As originally found:

```rust
let x = self.layer_norm_mlp.forward(x);
let x = self.lin1.forward(x);
let x = self.activation.forward(x);
self.lin2.forward(x)
```

Upstream is `x = x + self.mlp(self.layer_norm_mlp(x))` (`block.py:126-131`). The attention residual
is present at `transformer.rs:65`; the MLP one is not.

This is the dangerous kind of wrong: shapes are unaffected, so nothing downstream complains. What
changes is that the block stops being a transformer block. Every path through the network is forced
through `lin2`, the identity path that residual stacks exist to preserve is gone, and with 16 blocks
composed the output is a different function of the weights — not a perturbed one.

**Fix.** `x.clone() + self.lin2.forward(...)`, keeping the pre-residual `x` rather than the
normalised one, exactly as the attention branch already does.

### 3.4 Burn's `MultiHeadAttention` cannot carry these weights

**Status: fixed.** Replaced by a local `MultiHeadSelfAttention` with `lin_q`/`lin_k`/`lin_v` at
`.with_bias(false)`, a bias-true `projection`, and the attend delegated to
`burn::tensor::module::attention` (§3.4.2). `block_param_paths_and_shapes_match_checkpoint` is green,
so the block's parameter tree now matches all 13 keys in §1 exactly. **Nothing further is needed
before a checkpoint load.**

Two bugs surfaced while wiring it up, both in the tail of `forward`, both shape-invisible in the way
§3.3 was:

- `projection` was constructed and in the parameter tree but never applied — 1,049,600 parameters
  per block, 16.8M across the model, loaded and ignored.
- The output un-permute assumed `attention()` returns `[b, g, H, D]`. It returns
  `[batch, num_heads, seq_q, val_dim]` (`kernel/attention/base.rs:129-136`), so the inverse is
  `swap_dims(1, 2)` then `reshape([b, g, c])` — the mirror of the input path.

As originally found: `init` and `forward` built and called
`burn::nn::attention::MultiHeadAttention`. Three separate problems, none of them fixable from the
outside:

- **Naming.** Burn's fields are `query`, `key`, `value`, `output` (`mha.rs`,
  `MultiHeadAttention`); the checkpoint says `lin_q`, `lin_k`, `lin_v`, `projection`. That is four
  remap rules per block, 64 across the model, where matching the field names costs nothing and
  needs zero.
- **Bias.** `MultiHeadAttentionConfig::init` builds all four projections with
  `LinearConfig::new(d_model, d_model)`, and `LinearConfig` defaults `bias: true`. The checkpoint
  has **no** `lin_q`/`lin_k`/`lin_v` bias (§1). That is three orphan parameters per block, 48 across
  the model. A strict load fails on them; a lenient one leaves them at their random init and
  silently perturbs every query, key and value in the processor.
- **Dropout.** `MultiHeadAttentionConfig` defaults `dropout: 0.1` against the checkpoint's
  `dropout_p: 0.0`, and `transformer.rs:39-40` does not override it. On `Wgpu` this is latent rather
  than live — Burn's `Dropout` is a no-op without autodiff — but it is a wrong number sitting in the
  config waiting for a backend change to make it real.
- **The batch axis is fabricated.** Harmless at `batch_size = 1`, which is what this checkpoint
  runs at — see §3.4.1.

#### 3.4.1 The processor's batch is not the mappers' batch

**Status: fixed.** `batch_size` is now threaded through all three `forward` levels, the block
unfolds `[batch * grid, channels]` to `[batch, grid, channels]` before attending and refolds after,
and `block_forward_does_not_attend_across_batch_elements` pins it. Callers pass 1.

Recorded below as originally found, because the reasoning is what the test rests on.

**It was never a live bug —** `aifs-single-mse-2.0` is the deterministic
single model, so there are no ensemble members to batch, and anemoi folds multi-step into the
feature axis rather than the batch axis. The only `batch_size` anywhere in the metadata is
`{"test": 4}` under the _training_ config. At inference, `batch_size` is 1 and the current code is
correct.

Recorded anyway, because the assumption is invisible and the pipeline around it is asymmetric.

Both carry nodes folded as `[batch × grid, channels]`, batch-major, and the mappers rely on that
folding being _safe_: `expand_edges` tiles the edge list and offsets copy `i` by `i * edge_inc`
(`graph.rs:74-86`), so batch `b`'s edges only ever reach batch `b`'s nodes. A batch really is one
disconnected graph there, and the edge list is what enforces it. `decoder_test.rs`'s
`batch_size_two_expands_edges_and_trainable` pins exactly that.

**Self-attention has no edge list.** It is all-to-all by construction, so nothing enforces the
separation and it has to be reintroduced as a real dimension. Upstream does so explicitly:

```python
einops.rearrange(
    t,
    "(batch grid) (heads vars) -> batch heads grid vars",
    batch=batch_size,
    heads=self.num_heads,
)  # attention.py:158-165
...
einops.rearrange(
    out, "batch heads grid vars -> (batch grid) (heads vars)"
)  # attention.py:191
```

So the block's inputs and outputs stay folded — matching the mappers — and only the attention
unfolds, attends within each element, and refolds.

`TransformerProcessorBlock::forward` instead does `.unsqueeze()`, producing `[1, B*G, C]`. At
`batch_size = 1` that is correct. At any larger batch, element 0's queries would attend to element
1's keys, and once §3.5 lands the sliding window would straddle the seam between elements as well.

Nor can it be asserted. The leading `1` is manufactured by `unsqueeze` rather than read from the
caller, so the matching `squeeze::<2>()` always succeeds no matter what batch the caller intended,
and `[N, C]` does not reveal whether `N` is one batch of `G` or two of `G/2`. There is no assertion
here, only the appearance of one.

**Left as is, on purpose.** Threading `batch_size` through three `forward` levels was tried and
reverted: `batch_size` is 1 at inference for this checkpoint, and the processor is
element-independent — no edges, no cross-element term — so a caller that ever needs more can loop
over elements and get a bit-exact result. Note `MultiHeadSelfAttention::forward` already handles
arbitrary batch correctly, taking `[b, g, c]` and reshaping with the real `b`; only the block folds
to 1.

The one thing worth remembering is the asymmetry: both mappers take `batch_size`, and
`decoder_test.rs`'s `batch_size_two_expands_edges_and_trainable` pins B=2 on that side. A
full-model caller could reasonably assume the processor batches too. It does not.

### 3.5 Windowed vs. dense attention

**Status: OPEN.** `window_size` is now plumbed from `TransformerProcessorConfig` down to
`TransformerProcessorBlockConfig`, but `TransformerProcessorBlockConfig::init` still does not read
it — the value reaches the block and stops there.

**The single largest gap, and it is deliberate — it is not built here and needs its own issue.**

**What the checkpoint was trained with.** `window_size: 1120` with
`attention_implementation: flash_attention`. `FlashAttentionWrapper.forward` passes
`window_size=(1120, 1120)` to `flash_attn_func` (`attention.py:323`), i.e. a **symmetric sliding
window**: query `i` attends only to keys `j` with `|i - j| <= 1120`, a band 2241 wide. The SDPA
fallback builds the same band explicitly as `|i - j| <= window_size` (`attention.py:220-226`), so
the two implementations agree and the SDPA one is a usable oracle without a flash-attn build.

**What `transformer.rs` does today.** `MhaInput::self_attn(x_norm)` with no mask
(`transformer.rs:62`) — every node attends to every other node.

**Why this is not an approximation.** The window runs over the flattened _node index_ of the hidden
mesh, not over geography, and the weights were fit under that constraint. Attending over 40,320 keys
instead of 2,241 redistributes every attention weight in the softmax; the result is a different
function of the same weights, not a noisier version of the right one. There is no tolerance at which
a fixture comparison against anemoi passes. **Until this is built, `TransformerProcessor` output
will not match the checkpoint at any grid larger than 2241 nodes.**

**What it costs.** This is the part an earlier draft of this review got wrong, so the corrected
version, at `N = 40,320` hidden nodes, `H = 16` heads, `D = 64`, f32:

| implementation                            | scores memory                         | compute              |
| ----------------------------------------- | ------------------------------------- | -------------------- |
| naive dense (`nn::MultiHeadAttention`)    | `[1, H, N, N]` = **104 GB** per block | O(N²)                |
| fused dense (`module::attention`, §3.4.2) | **O(N)** — never materialised         | O(N²), ~18x the band |
| fused band (2241)                         | O(N)                                  | O(N · 2241)          |

The 104 GB figure only applies to the naive implementation, which is exactly what
`nn::MultiHeadAttention` does and exactly what §3.4.2 replaces. **On the fused path, dense attention
fits.** It is a slow path — roughly 18x the arithmetic the ±1120 band needs, since
`40320 / 2241 ≈ 18` — not an impossible one.

So the case for windowing is now **correctness first, cost second**: the numbers are wrong at any
grid wider than the band, and the compute is ~18x what it should be.

Note this is the _processor's_ cost and is separate from the mappers' — backward-mapper review §3.4
puts `graph_tranformer_conv` at ~26.6 GB on the decoder side, which is a genuine memory wall.

**Why the fused op does not just solve it.** `AttentionModuleOptions` has `is_causal` but **no
window field** (`burn-backend-0.21.0/src/backend/ops/modules/base.rs:458-471`). The only general
hook is the bool `mask`, documented at `[batch, num_heads, seq_len_q, seq_len_k]` — 26 GB at this
size, or 1.6 GB if a `[1, 1, N, N]` mask broadcasts. Either defeats the point.

**Follow-up, to be filed separately.** Inside the `MultiHeadSelfAttention` from §3.4: iterate
queries in row-blocks of `blk`, slice keys and values to `[max(0, i0 - w), min(N, i1 + w))`, call
`module::attention` per block with a small `[1, 1, blk, band]` mask for the ragged edges, and
concatenate. That keeps the fused kernel _and_ stays O(N). Cross-check against
`SDPAAttentionWrapper`'s explicit mask at a small `N` where dense is affordable.

### 3.6 `mlp_hidden_ratio` and `hidden_dim` are both config fields, and only one is read

**Status: fixed.** `mlp_hidden_ratio` is now a `usize` on the chunk and processor configs, and the
chunk derives the block's `hidden_dim` as `num_channels * mlp_hidden_ratio`. The `f32` rounding
question is gone with it.

`transformer.rs:13` declares `hidden_dim` and `:19-20` declares `mlp_hidden_ratio` with a default of
`4.`. `init` spends `hidden_dim` on `lin1`/`lin2` (`:43-45`) and never reads the ratio.

Upstream has only the ratio, and derives the width at the call site:
`hidden_dim=(mlp_hidden_ratio * num_channels)` (`chunk.py:122`). Two fields with exactly one correct
relationship between them is a way to be inconsistent — the same class of defect as backward-mapper
review §3.3 (`attn_channels`) and forward-mapper review §3.5 (`edge_dim`).

**Fix.** Keep `mlp_hidden_ratio`, derive `hidden_dim` in `init`. Note `mlp_hidden_ratio` is an
integer in the metadata; `f32` here invites the same rounding question the mappers have with `f64`.

### 3.7 `TransformerProcessorBlockConfig` carries two fields a block does not use

**Status: half fixed.** `num_layers` is gone from the block config. `window_size` is still declared
and still unread, which is now purely a restatement of §3.5.

`num_layers` (`transformer.rs:11`) and `window_size` (`:14`) are both declared and neither is read
by `init` (`:35-55`). `num_layers` is meaningless at block granularity — the block _is_ one layer —
and should be deleted. `window_size` becomes live only once §3.5 is built, and until then is exactly
the "config field that is accepted and ignored" pattern the mapper reviews flagged twice; keep it
only if §3.5 lands in the same change.

### 3.8 `vec![block; n]` clones one initialised block

**Status: fixed**, and neatly — both sites still use `vec![expr; n]`, but the thing repeated is now
the _config_, with `.iter().map(|c| c.init(device))` after it. Cloning a config is free of the
`ParamId` problem entirely, because each `init` mints fresh parameters. Pinned by
`chunk_blocks_are_independently_initialised`.

`transformer.rs:88-99` and `:127-131` both use `vec![expr; n]`, which evaluates `expr` once and
`Clone`s the result `n - 1` times. `Param::clone` **preserves the `ParamId`**
(`burn-core-0.21.0/src/module/param/base.rs:439-460`), so every block in a chunk ends up sharing
parameter identities _and_ identical initial values.

A checkpoint load overwrites the values, so this is not a wrong-numbers bug on the path we care
about. It is wrong everywhere else: fresh initialisation gives 16 identical blocks, and duplicate
`ParamId`s are meaningful to `Module::map`, `Module::visit` and optimizer state. It also reads as
though sharing were intended.

**Fix.** `(0..n).map(|_| cfg.init(device)).collect()`.

### 3.9 The configs nest a whole block config

**Status: fixed.** All three configs now take flat scalars, and each level builds the next one's
config inside `init`, as `encoder.rs` and `decoder.rs` do.

`TransformerProcessorChunkConfig` (`transformer.rs:76-79`) and `TransformerProcessorConfig`
(`:114-118`) each hold `conf: TransformerProcessorBlockConfig` alongside their own `num_layers`. So
`num_layers` exists at three levels with three different meanings, which is the immediate cause of
§3.1, and `init` has to unpack and re-pack every block field by hand (`:88-97`).

`encoder.rs` and `decoder.rs` take flat scalars and build the inner config inside `init`
(`decoder.rs:52-62`). Match that.

### 3.10 The MLP duplicates `common::MultiLayerPreceptron`

**Status: fixed**, and confirmed against the checkpoint: the block's MLP parameters now land at
`mlp.layers.{0,1}.{weight,bias}` with shapes `[1024, 4096]` / `[4096]` / `[4096, 1024]` / `[1024]`,
which is exactly what §1 requires modulo the one shared `mlp.0`/`mlp.2` remap.

`MultiLayerPreceptronConfig` gained defaults for `n_extra_layers` and `layer_norm` to make the
three-argument call site read well. That changed its arity, so `block.rs`'s two `node_dst_mlp` /
`node_src_mlp` call sites shed their now-redundant trailing `0, false`. No behaviour change — the
values passed were the new defaults.

`lin1` / `activation` / `lin2` (`transformer.rs:29-31, 43-45`) is exactly

```rust
MultiLayerPreceptronConfig::new(num_channels, num_channels, hidden_dim, 0, false)
```

— GELU-activated, no extra layers, no final activation, no layer norm (`common.rs:34-63`).

Reusing it is not only less code: it puts the parameters at `mlp.layers.{0,1}.{weight,bias}`, which
need the **same** `mlp.0 → mlp.layers.0` / `mlp.2 → mlp.layers.1` remap already required for the
mappers' `node_dst_mlp` (see the expectations in `decoder_test.rs:136-139`). One remap rule for the
whole model instead of two, and the naming lines up with the checkpoint's `mlp.` prefix rather than
inventing `lin1`/`lin2`.

### 3.11 Everything is private, so the module is dead code

**Status: OPEN.** All three config and module types still have no `pub`, though
`TransformerProcessorConfig::init` does. `main.rs` declares `mod transformer;` and nothing else
references it. Same finding as backward-mapper review §3.6.

The tests reach the private types through `use super::*` from an inner module, so they do not need
this changed — which is why the module can now be fully exercised while still being dead to
`main.rs`. The `pub` is for the load wiring.

### 3.12 Minor

**Status: OPEN**, all four.

- **"I think this is an autoencoder"** (now above the `MultiLayerPreceptron` construction) is wrong.
  It is the standard transformer position-wise feed-forward network: widen to
  `4 × num_channels`, activate, narrow back. Nothing is being reconstructed.
- **Batching is hardcoded to 1** — see §3.4.1. Correct for this checkpoint; the gap is that the
  assumption is never stated.
- **`burn::prelude::*`** here where `encoder.rs` and `block.rs` list imports explicitly. Same nit as
  backward-mapper review §3.8; pick one.
- **`TransformerProcessorChunk::forward` and `TransformerProcessor::forward` are identical loops.**
  They have to stay separate — the checkpoint's path structure requires both levels — but they can
  share a helper.
- **`num_heads` is required on the processor config but defaults to 16 on the chunk config.** Only
  the processor path is reachable in practice, so the default is dead. Harmless, but it is the same
  "one correct value, unenforced" shape as §3.6 was.

---

## 4. The change set

| Done | File             | Change                                                                      | Finding      |
| ---- | ---------------- | --------------------------------------------------------------------------- | ------------ |
| ✅   | `transformer.rs` | `squeeze` so `forward` stops panicking                                      | §3.2         |
| ✅   | `transformer.rs` | restore the MLP residual                                                    | §3.3         |
| ✅   | `transformer.rs` | add `num_chunks`; chunks = `num_chunks`, blocks = `num_layers / num_chunks` | §3.1         |
| ✅   | `transformer.rs` | replace `lin1`/`activation`/`lin2` with `common::MultiLayerPreceptron`      | §3.10        |
| ✅   | `transformer.rs` | derive `hidden_dim` from `mlp_hidden_ratio`; drop block `num_layers`        | §3.6, §3.7   |
| ✅   | `transformer.rs` | repeat the config, not the initialised block                                | §3.8         |
| ✅   | `transformer.rs` | flatten the config nesting                                                  | §3.9         |
| ✅   | `transformer.rs` | hand-rolled `MultiHeadSelfAttention`, bias-free q/k/v, checkpoint names     | §3.4         |
| —    | `transformer.rs` | nothing to change; `batch_size` is 1 for this checkpoint                    | §3.4.1       |
| ✅   | `transformer.rs` | attend via `module::attention`, not a hand-written matmul + softmax         | §3.4.2       |
| ❌   | `transformer.rs` | `pub` on the types; fix the autoencoder comment                             | §3.11, §3.12 |

**The parameter tree is complete.** All 13 keys per block land where §1 requires, pinned by a green
`block_param_paths_and_shapes_match_checkpoint`. Nothing structural blocks a load.

What is left is behavioural, in priority order:

1. **§3.5, windowed attention.** The numbers are wrong at any grid wider than 2241, and the compute
   is ~18x what it should be. This is the one that matters.
2. **§3.11 / §3.12**, cosmetic. §3.4.1 is a recorded assumption, not an action.

**§3.5 is a separate issue and should not be folded in.** So is wiring `load_from` in `main.rs`,
which is blocked for the mappers on graph buffers the conversion script does not yet dump
(backward-mapper review §3.7) but is _not_ blocked for the processor — see §1. That makes the
processor the natural place to prove the load path end to end.

**Verification.** `cargo check && cargo clippy -p aifsv2 --all-targets && cargo test -p aifsv2`.

---

## 5. Test coverage, as it now stands

`src/transformer_test.rs` (six tests, `wgpu`, no fixture). **All green**, alongside the seven
pre-existing tests — 13 passing in total.

| Test                                               | What it pins                                                              | Finding     | Status |
| -------------------------------------------------- | ------------------------------------------------------------------------- | ----------- | ------ |
| `block_forward_preserves_shape`                    | `[N, C]` in, `[N, C]` out — the most basic thing a block must do          | §3.2        | green  |
| `block_forward_keeps_both_residuals`               | with both branch outputs zeroed, `forward(x) == x` exactly                | §3.3        | green  |
| `chunk_blocks_are_independently_initialised`       | blocks in a chunk have distinct `ParamId`s and distinct values            | §3.8        | green  |
| `processor_splits_num_layers_across_num_chunks`    | `num_chunks` chunks of `num_layers / num_chunks` blocks, at the real 16/2 | §3.1        | green  |
| `processor_rejects_layers_not_divisible_by_chunks` | an uneven split panics rather than silently dropping the remainder        | §3.1        | green  |
| `block_param_paths_and_shapes_match_checkpoint`    | one block at real dims against the 13 keys in §1, biases and all          | §3.4, §3.10 | green  |

`block_forward_keeps_both_residuals` is the one worth reading closely. It zeroes the attention's
output projection and the whole MLP so both residual branches contribute exactly zero, which makes
the block the identity **iff** both skips are present. With §3.3's missing skip the output was
identically zero instead — a maximally separated pair of outcomes that needs no reference
implementation to distinguish. It zeroes whole submodules via a `ModuleMapper` rather than named
tensors, because `MultiLayerPreceptron` keeps `layers` private to `common.rs`; that also makes the
test indifferent to the MLP's internal shape.

`chunk_blocks_are_independently_initialised` reads `attention.lin_q.weight` rather than a layer
norm's `gamma`. `gamma` initialises to ones, so it is legitimately identical across blocks and could
not distinguish independent blocks from clones — the assertion would have been vacuous.

`block_param_paths_and_shapes_match_checkpoint` builds a **single** block at production width rather
than the whole processor: 208 tensors at full width is ~209M parameters, and it would pin nothing
that 13 tensors at ~13M do not. It is an exact set comparison, so it catches structural drift in
both directions — a stray q/k/v bias as readily as a missing norm. It now matches §1 exactly:

```
attention.lin_q.weight [1024, 1024]        layer_norm_attention.{gamma,beta} [1024]
attention.lin_k.weight [1024, 1024]        layer_norm_mlp.{gamma,beta} [1024]
attention.lin_v.weight [1024, 1024]        mlp.layers.0.{weight,bias} [1024, 4096] / [4096]
attention.projection.{weight,bias}         mlp.layers.1.{weight,bias} [4096, 1024] / [1024]
```

Two remap rules are all that stand between this and a load: `mlp.0` → `mlp.layers.0` and `mlp.2` →
`mlp.layers.1`, which the mappers' `node_dst_mlp` already needs. The norms need none —
`PyTorchToBurnAdapter` handles `weight`/`bias` → `gamma`/`beta` (§2) — and neither does the
attention, now that the field names match.

**Three gaps, all deliberate:**

1. **No numerical oracle against anemoi.** The mappers have the same gap
   (backward-mapper review §6). For the processor it is cheaper than for either mapper — no
   `HeteroData` is needed, just a `TransformerProcessorBlock` and a tensor — and `SDPAAttentionWrapper`
   runs without flash-attn, so it is worth filing.
2. **Nothing windowed is tested**, because nothing windowed is implemented. Blocked on §3.5, and the
   fixture in gap 1 is what will pin it.
3. **No batch-isolation test**, deliberately: `batch_size` is 1 for this checkpoint (§3.4.1), and
   the processor is element-independent, so there is nothing for such a test to protect.
