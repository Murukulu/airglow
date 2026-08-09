# Review: `GraphTransformerForwardMapper`, as implemented

Companion to [`graph-transformer-forward-mapper.md`](./graph-transformer-forward-mapper.md), which
decided what to build, and to [`graph-transformer-explained.md`](./graph-transformer-explained.md),
which explains the operator. This one reviews the code that now exists in `src/encoder.rs`,
`src/common.rs` and `src/graph.rs` against upstream: what is wrong, what is missing, and what has
already been checked and is right.

Upstream is pinned to **anemoi-core `b666d5bf`** (`models-0.9.3`) — the checkout at
`~/Documents/projects/anemoi-core`, and the version the checkpoint was trained with
(`anemoi_models-0.9.3.dist-info` in `data/aifs-single-mse-2.0/quiet_grub/.venv`). Paths are
relative to `models/src/anemoi/models/` for anemoi and to `aifsv2/` for this repo.

> **The design note is pinned to a different commit.** It cites anemoi `0fa84c1` and still opens
> with "nothing has been implemented yet". Its substance holds, but a few of its line citations no
> longer resolve, and one of its claims does not survive contact with 0.9.3 — see §3.6.

Corroborating evidence used throughout:

- `data/aifs-single-mse-2.0.safetensors` — 31 `model.encoder.*` keys, read from the safetensors
  header directly rather than via `scripts/parse_safetensors.py`.
- `data/aifs-single-mse-2.0_metadata.json` — `num_channels: 1024`, `num_heads: 16`,
  `mlp_hidden_ratio: 4`, `qk_norm: false`, `trainable_size: 8`, `num_chunks: 4`, Activation
  `torch.nn.GELU`, LayerNorm `torch.nn.LayerNorm`.
- `cargo check` — the crate compiles. 26 warnings, all of them dead-code (nothing in `main.rs`
  reaches the encoder), two unused imports, and one `type_alias_bounds`.

---

## 1. Confirmed bugs

### 1.1 `get_qkve` is handed `attn_channels` where it needs `out_channels_conv`

`encoder.rs:189-194` calls `get_qkve(..., self.conf.num_heads, self.conf.attn_channels)`, and
`encoder.rs:250-254` spends those as `reshape([-1, h, c])`. The einops line it is porting
(`layers/block.py:667-672`) is

```python
"nodes (heads vars) -> nodes heads vars", heads=self.num_heads, vars=self.out_channels_conv
```

so the trailing extent is `out_channels_conv = attn_channels / num_heads` — **64, not 1024**.

The `-1` hides it, because the numbers happen to divide. `x_dst` is `[40320, 1024]`, so 41,287,680
elements against a row size of `16 × 1024 = 16384` gives `[2520, 16, 1024]`. No error. Everything
downstream then reads the wrong axis:

- `norm` in `graph_tranformer_conv` (`common.rs:169`) reads `query.dims()[2]`, so the attention
  scale becomes `1/√1024 = 1/32` instead of `1/√64 = 1/8`.
- `n_dst` (`common.rs:172`) becomes 2520 instead of 40320.

It does get caught — but only by `assert_eq!(n_dst, edge_index.num_dst)` at `common.rs:173`, which
is therefore doing real work (§2.1). Without that assert, `query.select(0, dst)` would index a
2520-row tensor with values up to 40319, and `select_assign` would write past the end of a 2520-row
output. Burn documents indices as unchecked on some backends (`burn-tensor-0.21.0`,
`src/tensor/api/base.rs:1668-1671`), and the cubecl `select_assign` kernel runs in checked-launch
mode, which **clamps** out-of-bounds accesses rather than panicking
(`burn-cubecl-0.21.0/src/kernel/index/select_assign.rs:9-11`). The failure mode absent the assert is
silently wrong numbers.

The two `TODO(saiputravu): double check these are indeed h, c` comments at `encoder.rs:192-193` were
right to be suspicious.

**Fix.** `get_qkve` is a method with `self.conf` in hand; it should not take `h` and `c` as
parameters at all. Read `num_heads` and `out_channels_conv()` inside it, and delete both arguments
along with both TODOs.

### 1.2 The mapper returns the embedded `x_src`; anemoi returns the raw input

`encoder.rs:333-337`:

```rust
let x = self.pre_process(x);                                   // embeds BOTH src and dst
let (x_src, x_dst) = self.proc.forward(x, edge_attr, edge_idx);
(x_src, x_dst)                                                 // x_src is [N_src, 1024]
```

Whatever `proc` returns first is 1024 wide, because `pre_process` embedded it. anemoi's
`GraphTransformerForwardMapper.forward` (`layers/mapper.py:599-620`) ends with

```python
return x[0], x_dst
```

where `x` is the argument as received, _before_ `pre_process`, i.e. the un-embedded 224-wide input.

This is not cosmetic bookkeeping. `models/encoder_processor_decoder.py:366` does

```python
x_data_latent, x_latent = self._run_mapper(
    self.encoder, (x_data_latent, x_hidden_latent), ...
)
```

rebinding `x_data_latent` to the encoder's first return, and passes it at line 391 as the decoder's
`x_dst`. The decoder is constructed with `in_channels_dst=self.input_dim` (line 119) — 224, matching
`emb_nodes_src.weight [1024, 224]`. Returning 1024 here breaks the decoder both at load and at run.

**Fix.** Capture `x.0` before `pre_process` and discard `proc`'s first return:

```rust
let x_src = x.0.clone();
let (_, x_dst) = self.proc.forward(self.pre_process(x), edge_attr, edge_idx);
(x_src, x_dst)
```

Discarding it is unconditional, and does not rest on `update_src_nodes`. That flag lives on the
**block**, not the mapper (`encoder.rs:32-33`, `#[config(default = false)]`), and
`GraphTransformerForwardMapperConfig::init` (`encoder.rs:300-310`) calls
`GraphTransformerProcessorBlockConfig::new` with the eight non-defaulted fields, so it never sets
it. anemoi is the same shape: `GraphTransformerBaseMapper` builds `GraphTransformerMapperBlock`
without passing `update_src_nodes` (`layers/mapper.py:276-284`), and both
`forward_with_heads_sharding` (`:478`) and `run_processor_chunk_edge_sharding` (`:388`) drop the
block's first return with `(_, x_dst)` before the forward mapper substitutes `x[0]`. So the src
branch of `proc.forward` (`encoder.rs:215-228`) is unreachable from this mapper either way — if
someone flips the flag on the block, its output is computed and thrown away, in the port and
upstream alike.

While in there: `post_process` (`encoder.rs:347`) returns `()` and is called at `encoder.rs:335` for
a side effect it does not have. It is dead — delete it. The backward mapper (#18) is where
`post_process` becomes real, since `BackwardMapperPostProcessMixin` (`layers/mapper.py:112-122`)
overrides it with `node_data_extractor`.

### 1.3 `graph::cat` produces a `colptr` that is not a `colptr`

`graph.rs:36-37`. Every per-batch block contributes its whole array, both endpoints included, so
concatenating `batch_size` blocks of `[0, …, E]` gives length `batch·(N_dst + 1)` with duplicated
boundaries, where a valid colptr is length `batch·N_dst + 1`. Separately,
`e.colptr.clone().max_dim(0)` is standing in for "the last element" — the same number for a sorted
colptr, but it states the wrong invariant.

Currently harmless, because `colptr` is read nowhere.

**Fix.** Delete the `colptr` field and the `colptrs` / `colptrs_max` machinery in `cat`. It is dead
code that is also wrong, and its only justification is a CubeCL kernel that does not exist yet. When
that work lands, the correct construction is one bincount plus a prefix sum over the _final_
concatenated `dst` — strictly simpler than threading a running offset through `cat`. The
`TODO(saiputravu)` at `graph.rs:34-35` asks exactly this; the answer is no.

### 1.4 `sparse_segment_softmax` can divide by exactly zero

`common.rs:107-133`. The global-max shift is mathematically exact, as the comment at
`common.rs:105-110` says. What it gives up — and the comment does not say — is the per-segment
guarantee that the denominator is at least 1. Under a global max, a destination whose logits all sit
more than ~88 below the **global** max (f32 `exp` underflow) has every `numerator` entry flush to
zero, so its denominator is exactly zero, and `common.rs:133` returns `0/0 = NaN`.

Reachability, stated honestly: logits are `q·k/8` over post-LayerNorm projections, so a global spread
above 88 across all `E · H ≈ 12M` logits is possible but not expected. The fix is one add.

The case that _looks_ dangerous and is not: a zero-degree destination leaves its denominator row at
zero, but `dst` never names that row, so `denominator.select(0, dst_idx)` never gathers it. No NaN
from that path.

**Fix.** `+ 1e-16` on the denominator, commented as covering underflow introduced by the global-max
choice — the hazard is created here, and does not exist in a per-segment formulation. The principled
fix is a per-segment max, which needs a duplicate-safe scatter-max that Burn 0.21 does not have; that
stays a follow-up.

---

## 2. Assertions

An assertion earns its place by catching a mistake this code can actually make. Fidelity to anemoi
is not a reason for one.

### 2.1 Present, and load-bearing

| Assert                                        | What it catches                                                                                                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `common.rs:173` `n_dst == edge_index.num_dst` | §1.1, and it is the **only** thing turning that bug into a panic instead of clamped out-of-bounds writes. The message should say why: the output buffer is sized from `query`, the write indices come from `edge_index`. |
| `graph.rs:56` `a == 2` in `EdgeIndex::add`    | The `slice(0..b)` / `slice(b..)` split at `graph.rs:71-72` assumes exactly two rows. Any other rank-2 shape splits the wrong way, silently.                                                                              |

### 2.2 Missing, and justified

**`sparse_segment_softmax` should assert the trailing extent is 1.** `common.rs:99` destructures
`let [e, h, _] = x.shape().dims();` and throws the last extent away. Were `x` an `[E, H, C]` with
`C > 1`, the accumulator is still built `[n_dst, h, 1]` at `common.rs:118`, and
`TensorCheck::select_assign` (`burn-tensor-0.21.0/src/tensor/api/base.rs:1683`) validates only that
the axis is in range and that `values.shape[dim] == indices.shape[0]`. It never compares the
non-indexed axes. The kernel then runs with mismatched strides and returns garbage without
panicking. This is a silent-corruption path Burn does not close.

**`graph_tranformer_conv` should assert `key.dims()[0] == edge_index.num_src`**, and the same for
`value`. `src` holds values in `[0, num_src)`; a `key` with fewer rows makes `common.rs:182` an
out-of-bounds `select`, documented as unchecked on some backends (`base.rs:1668-1671`). Nothing else
in the function would notice. Note this is the one pairing PyG happens to validate for free — it
falls out of `_set_size` on the fourth `_collect` iteration — but the reason to have it here is the
unchecked gather, not that PyG does it.

An assert on `edges.dims()[0]` is _not_ justified: `k_j = key.select(0, src) + edges`
(`common.rs:182`) is elementwise, and Burn's broadcast check rejects a mismatch one line later with
a clear error.

### 2.3 Present, and cannot fire

- **`common.rs:198-210`** — the shape check on `alpha` after `sparse_segment_softmax`. `alpha` is
  `numerator / denominator.select(0, dst_idx)`, and `numerator` derives from the `[E, H, 1]` input
  by shape-preserving ops only. The comparison `[_e, h, 1] == [__e, __h, __one]` is structurally
  guaranteed by the twelve lines above it. Thirteen lines that cannot report anything.
- **`graph.rs:57-63`** — the `b == 1 || b == e` arm. If `b` were neither, `lhs.src + top` is
  `[E] + [b]`, which Burn's elementwise check (equal-or-1 per axis) rejects with a clear error one
  line later. The `a == 2` check sharing the block is the one that matters (§2.1).

### 2.4 Diagnostics only

`attn_channels % num_heads == 0` at `GraphTransformerProcessorBlockConfig::init`. If it does not
hold, `out_channels_conv` (`encoder.rs:67-70`) truncates, the `lin_*` layers emit
`num_heads · out_channels_conv < attn_channels`, and the disagreement surfaces at
`self.projection.forward(msg + res)` (`encoder.rs:211`) as a matmul shape error — loud, but several
layers away from the cause. It prevents no corruption, so it is a legibility call rather than a
correctness one.

---

## 3. Missing, or worth changing

Ordered by how much they will hurt.

### 3.1 Nothing loads the checkpoint

`main.rs` is still the `Model { inp }` placeholder, which is why all 26 `cargo check` warnings are
dead-code. Wiring `load_from` to the 31 real keys needs these remaps:

| Checkpoint key                     | Burn path                          | Why                                                                                                                                                                                                        |
| ---------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.encoder.…`                  | `…`                                | prefix strip                                                                                                                                                                                               |
| `proc.node_dst_mlp.{0,2}.*`        | `proc.node_dst_mlp.layers.{0,1}.*` | `nn.Sequential` numbers the activation; `Vec<Linear>` does not                                                                                                                                             |
| `proc.layer_norm_attention_dest.*` | `proc.layer_norm_attention_dst.*`  | spelling                                                                                                                                                                                                   |
| `proc.layer_norm_attention.*`      | — (ignore)                         | `layers/block.py:643` aliases `layer_norm_attention_dest = layer_norm_attention`, so `state_dict()` emits both keys with identical values. It is a duplicate, not a third LayerNorm.                       |
| `model.encoder.edge_inc [2, 1]`    | no home yet                        | it **is** in the safetensors — a persistent buffer (`layers/mapper.py:165`). `edge_inc` is currently a `forward` argument (`encoder.rs:327`), so this key lands nowhere. Give it a field or allow-list it. |

`PyTorchToBurnAdapter`, already used in `main.rs:20`, handles the `Linear` weight transposition.

### 3.2 `edge_index` and the base edge attributes are still unobtainable

The safetensors carries only `trainable` and `edge_inc`. `edge_attr` and `edge_index_base` are
registered `persistent=False` (`layers/mapper.py:159-165`) and live in the `.ckpt`'s `graph_data`.
Reading the header confirms it: nothing else edge-shaped is present. **The encoder cannot run on
real weights until `scripts/ckpt_to_safetensors.py` dumps them** — this is the actual blocker on
end-to-end, not the layer code. Same conclusion as follow-up 1 of the design note, restated because
it has not moved.

### 3.3 No chunking

anemoi runs the encoder at `max(self.num_chunks, NUM_CHUNKS_INFERENCE_MAPPER)`
(`layers/mapper.py:432`) with `num_chunks: 4` in the metadata, chunked over destination nodes.
Chunking is exact here — softmax segments are per-destination, so a destination's result does not
depend on which chunk it lands in.

It is also not optional at scale. With `E = 748,348`, `H = 16`, `C = 64`, one `[E, H, C]` f32 is
~3.06 GB, and `graph_tranformer_conv` holds four live at once: `q_i`, `k_j`, `v_j` and `msg`
(`common.rs:181-215`) ≈ 12 GB. `GraphTransformerForwardMapperConfig::num_chunks`
(`encoder.rs:274-275`) exists and is ignored.

### 3.4 `GraphTransformerProcessorBlock` is the mapper block

The struct at `encoder.rs:39` is anemoi's **`GraphTransformerMapperBlock`** (`layers/block.py:581`):
pair tensor in and out, `layer_norm_attention_src` plus `_dest`, `num_chunks` forced to 1. anemoi's
actual `GraphTransformerProcessorBlock` (`layers/block.py:739`) takes a single tensor, has one
`layer_norm_attention`, no `_src` variants, and chunks its projection. Renaming before #17 avoids a
collision. Two spellings in the same neighbourhood: `graph_tranformer_conv` (`common.rs:155`) and
`MultiLayerPreceptron` (`common.rs:25`).

### 3.5 `edge_dim` can disagree with itself

It is a free config field (`encoder.rs:269`), but it is determined: 3 (`edge_length` + `edge_dirs`)

- `trainable_size` (8) = 11, matching `lin_edge.weight [1024, 11]`. Derive it inside
  `GraphTransformerForwardMapperConfig` from a base-attribute width and `trainable_size`, rather than
  accepting it and asserting about it — that removes the way to be inconsistent instead of detecting
  it. Also `edge_attr_shape` (`encoder.rs:271`) means "number of edges"; `num_edges` says so.

### 3.6 `edge_pre_mlp` does not exist in 0.9.3

`GraphTransformerBaseBlock.__init__` (`layers/block.py:389-459`) has no `edge_pre_mlp`, on either
the mapper or the processor path. The design note lists it among the config-gated `Option` fields to
keep for API fidelity; against this upstream there is no API to be faithful to. The field
(`encoder.rs:28`, `encoder.rs:279`) is threaded through `init` and never read — delete it.

### 3.7 `out_channels_dst` is unused

anemoi's forward mapper hard-codes `out_channels_dst=None` (`layers/mapper.py:583`). The field at
`encoder.rs:266` is never read. Either reject `Some(_)` or, simpler, drop it from the forward
mapper's config and reintroduce it in the backward mapper, which genuinely takes one.

### 3.8 qk-norm module type

Inert here — `qk_norm: false` — but for the record, anemoi's `QueryNorm` and `KeyNorm` resolve to
`anemoi.models.layers.normalization.AutocastLayerNorm` with `bias: false` (metadata
`layer_kernels`), not a bias-carrying `LayerNorm`. `encoder.rs:127-128` builds `LayerNormConfig`,
which defaults `bias: true`.

### 3.9 No tests

`data/aifs-single-mse-2.0/quiet_grub/.venv` has anemoi-models 0.9.3, torch 2.10 and
torch_geometric 2.8 installed, so both the real `GraphTransformerConv` and the whole
`GraphTransformerMapperBlock` are available as fixture oracles, as
[`graph-transformer-forward-mapper.md`](./graph-transformer-forward-mapper.md) §6 sketches. The test
that matters most is duplicate-index aggregation, and it has to run on `wgpu` — `burn-ndarray`
accumulates duplicates correctly under every primitive, so a CPU-only suite proves nothing about the
backend that ships.

### 3.10 Warnings

Two unused imports (`ops::InterpolateOptions` at `encoder.rs:6`, `graph::cat` at `encoder.rs:14`)
and `type_alias_bounds` on the `PairTensor` alias at `encoder.rs:154`.

---

## 4. Checked and correct

Recorded so they do not get re-litigated.

- **The conv matches `layers/conv.py:125-145` exactly.** Same projected `edges` added to _both_ key
  and value, `(q_i * k_j).sum_dim(-1)` scaled by `1/√C`, segment softmax over `dst`, `v_j * alpha`,
  `select_assign(0, dst, ·, Add)`.
- **`sum_dim(-1)` is valid and keeps the axis.** It takes `AsIndex`
  (`burn-tensor-0.21.0/src/tensor/api/numeric.rs:451`), so negative indexing works, and it returns
  `Self` — the rank is preserved. This is why the port needs no equivalent of `conv.py:145`'s
  `alpha.view(-1, heads, 1)`.
- **`select_assign` on cubecl is duplicate-safe by construction.** One thread per non-indexed
  coordinate, looping serially over the indexed axis
  (`burn-cubecl-0.21.0/src/kernel/index/select_assign.rs:48-57`), no atomics. The design note's
  argument for preferring it to `scatter_nd` holds against the shipped kernel.
- **The conv does not depend on destination-sortedness.** `select` and `select_assign` are
  order-agnostic; only `colptr` needs sorted edges. The "You can assume sorted by dest" comment at
  `graph.rs:5` is unenforced, but nothing currently relies on it either.
- **`MultiLayerPreceptron` builds the right shape.** With `n_extra_layers = 0`,
  `layer_norm = false`, `final_activation = false`, `common.rs:64-81` yields `Linear → GELU →
  Linear`, matching the `nn.Sequential` at `layers/block.py:455-459` and the `node_dst_mlp.{0,2}`
  checkpoint keys. The `n_extra_layers` bug the design note describes (§5 Step 1) is already fixed
  at `common.rs:36-41`.
- **LayerNorm epsilon matches.** Burn's `LayerNormConfig` defaults `epsilon = 1e-5`
  (`burn-nn-0.21.0/src/modules/norm/layer.rs:20`), the same as `torch.nn.LayerNorm`.
- **Batch expansion is consistent between edges and edge attributes.** `repeat_dim` tiles
  (`burn-tensor-0.21.0/src/tensor/api/base.rs:1975-1988`), which is what einops
  `"e f -> (repeat e) f"` means, so `TrainableTensor::forward` (`common.rs:249-259`) and
  `expand_edges` (`graph.rs:84-96`) agree on batch-major ordering. Concatenating destination-sorted
  per-batch blocks with `dst += i·N_dst` stays globally destination-sorted.
- **The residual asymmetry is right.** `x_skip` is taken before the norms (`encoder.rs:173`);
  `x_r = lin_self(...)` consumes the **post**-norm `x_dst` (`encoder.rs:182`) while the residual
  added after `projection` is the **pre**-norm `x_skip.1` (`encoder.rs:212`); then
  `node_dst_mlp(layer_norm_mlp_dst(out)) + out`. Matches `layers/block.py:713-722`.
- **`lin_edge` bias.** anemoi's `bias=False` at `layers/block.py:445` is commented out, and the
  checkpoint has `lin_edge.bias [1024]`. The default-true bias at `encoder.rs:90-91` is correct.

---

## 5. The change set

§1 and §2 together are one small, self-contained change:

| File         | Change                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------- |
| `encoder.rs` | `get_qkve` drops `h`/`c` and reads `num_heads` / `out_channels_conv()` (§1.1)                |
| `encoder.rs` | forward mapper returns the pre-embedding `x.0`; `post_process` deleted (§1.2)                |
| `graph.rs`   | `colptr` field and the `colptrs` / `colptrs_max` machinery in `cat` deleted (§1.3)           |
| `common.rs`  | `+ 1e-16` on the softmax denominator (§1.4)                                                  |
| `common.rs`  | assert the trailing extent of `x` is 1; assert `key` / `value` rows against `num_src` (§2.2) |
| `common.rs`  | delete the post-softmax shape check; `graph.rs` delete the `b == 1 \|\| b == e` arm (§2.3)   |

The §3 items are separate pieces of work and should not be folded in.

**Verification.** `cargo check && cargo clippy -p aifsv2 --all-targets`, then a
`GraphTransformerConv` fixture generated from the `quiet_grub` venv — a small destination-sorted
bipartite graph with one zero-degree destination and one of degree 1 — compared at `1e-5` on `wgpu`,
plus the duplicate-index aggregation test. §1.1 and §1.2 both fall out of a shape-level test alone;
the fixture is what pins the arithmetic.
