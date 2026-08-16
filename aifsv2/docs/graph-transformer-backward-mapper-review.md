# Review: `GraphTransformerBackwardMapper`, as implemented

Companion to [`graph-transformer-forward-mapper-review.md`](./graph-transformer-forward-mapper-review.md),
which reviews the encoder side, and to
[`graph-transformer-explained.md`](./graph-transformer-explained.md), which explains the operator.
This one reviews `src/decoder.rs` against upstream: what is right, what is wrong, and what is
missing.

Upstream is pinned to **anemoi-core `b666d5bf`** (`models-0.9.3`) — the checkout at
`~/Documents/projects/anemoi-core`, and the version the checkpoint was trained with. Anemoi paths
are relative to `models/src/anemoi/models/`; repo paths to `aifsv2/`.

Corroborating evidence: the safetensors header read directly (§1), and
`data/aifs-single-mse-2.0_metadata.json` (`num_channels: 1024`, `num_heads: 16`,
`mlp_hidden_ratio: 4`, `qk_norm: false`, `trainable_size: 8`, `num_chunks: 4`).

The headline: **the port is arithmetically faithful.** Everything in §2 checks out line by line
against upstream. The findings in §3 are dead config, visibility, and one trap laid for the
checkpoint-loading work — not wrong numbers. That is a different situation from the encoder review,
which opened with four confirmed bugs.

---

## 1. Checkpoint ground truth

`data/aifs-single-mse-2.0.safetensors` carries **32 `model.decoder.*` keys** (the encoder has 30 —
the forward-mapper review's "31" is off by one):

```
model.decoder.emb_nodes_dst.{weight,bias}            [1024, 224], [1024]
model.decoder.trainable.trainable                    [1626240, 8]
model.decoder.edge_inc                               [2, 1]
model.decoder.node_data_extractor.0.{weight,bias}    [1024], [1024]        <- LayerNorm
model.decoder.node_data_extractor.1.{weight,bias}    [120, 1024], [120]    <- Linear
model.decoder.proc.lin_{query,key,value,self}.{w,b}  [1024, 1024], [1024]
model.decoder.proc.lin_edge.{weight,bias}            [1024, 11], [1024]
model.decoder.proc.projection.{weight,bias}          [1024, 1024], [1024]
model.decoder.proc.layer_norm_attention.{w,b}        [1024]   <- alias of _dest, see below
model.decoder.proc.layer_norm_attention_dest.{w,b}   [1024]
model.decoder.proc.layer_norm_attention_src.{w,b}    [1024]
model.decoder.proc.layer_norm_mlp_dst.{w,b}          [1024]
model.decoder.proc.node_dst_mlp.{0,2}.{weight,bias}  [4096,1024]/[4096], [1024,4096]/[1024]
```

Derived: `in_channels_dst = 224`, `hidden_dim = 1024`, `out_channels_dst = 120`, `edge_dim = 11`
(1 `edge_length` + 2 `edge_dirs` + 8 trainable), `num_heads = 16`, `out_channels_conv = 64`, MLP
hidden `= 4096`. `N_src = 40,320` (hidden grid), `N_dst = 542,080` (data grid), `E = 1,626,240`,
mean destination degree exactly **3.0**.

**There is no `emb_nodes_src` key**, which is correct: the backward mapper does not have one, and
its absence is the single structural asymmetry between the two mappers.

`layer_norm_attention` and `layer_norm_attention_dest` are the same module —
`self.layer_norm_attention_dest = self.layer_norm_attention` at `block.py:644` — so `state_dict()`
emits both keys with identical values. One Burn field, one remap, one ignored duplicate.

---

## 2. Checked and correct

Recorded so it does not get re-litigated. Checked against `mapper.py:623-725` and
`block.py:386-757`.

- **`pre_process` embeds only `x_dst`** (`decoder.rs:99-103`), matching
  `GraphTransformerBackwardMapper.pre_process` (`mapper.py:716-725`). `x_src` passes through
  untouched because it arrives from the processor already at `hidden_dim`. Contrast
  `ForwardMapperPreProcessMixin` (`mapper.py:124-139`), which embeds both.
- **`post_process` is `Linear(LayerNorm(x_dst))`** (`decoder.rs:106-109`), matching
  `nn.Sequential(nn.LayerNorm, nn.Linear)` at `mapper.py:706-708` and the `.0`/`.1` key split in §1.
  Applied after the block, not before.
- **`forward` returns `x_dst` alone**, not a pair. `GraphTransformerBackwardMapper` inherits
  `GraphTransformerBaseMapper.forward` (`mapper.py:492-518`), which dispatches to
  `forward_with_edge_sharding` and returns `out_dst` (`mapper.py:454`) — no `x_src` component. Only
  the *forward* mapper overrides this to re-attach `x[0]` (`mapper.py:620`).
- **`update_src_nodes` is left at its default false.** Matches `GraphTransformerMapperBlock`'s
  default (`block.py:594`) and the fact that `GraphTransformerBaseMapper` never passes it
  (`mapper.py:276-285`). The `node_src_mlp` / `layer_norm_mlp_src` branch is therefore unreachable
  from this mapper, exactly as upstream.
- **`trainable` before `expand_edges`** (`decoder.rs:85-86`) matches `prepare_edges`
  (`mapper.py:304-305`). The two operations are independent, so the order is immaterial either way.
- **MLP hidden width** is `hidden_dim * mlp_hidden_ratio = 4096`, matching
  `node_dst_mlp.0.weight [4096, 1024]`.
- **The block is shared with the encoder and needs no decoder-specific variant.** Upstream reaches
  the same conclusion: both mappers construct the same `GraphTransformerMapperBlock`
  (`mapper.py:276-285`) and differ only in the pre/post-process mixins.

---

## 3. Findings

Ordered by how much they will hurt.

### 3.1 `node_data_extractor`'s field names collide with the checkpoint prefix

The Burn fields are `node_data_extractor_norm: LayerNorm` and `node_data_extractor: Linear`
(`decoder.rs:37-38`), which `collect()` renders as:

```
node_data_extractor.weight        [1024, 120]     <- the Linear
node_data_extractor.bias          [120]
node_data_extractor_norm.gamma    [1024]          <- the LayerNorm
node_data_extractor_norm.beta     [1024]
```

The checkpoint spells the same two modules `node_data_extractor.0.*` (LayerNorm) and
`node_data_extractor.1.*` (Linear). So the Burn `Linear` field is named **exactly the prefix of both
checkpoint keys**, and the remap `node_data_extractor.1.` → `node_data_extractor.` is not
order-independent with `node_data_extractor.0.` → `node_data_extractor_norm.`: whichever rule
a `KeyRemapper` applies first changes what the second one sees.

The encoder has the analogous `node_dst_mlp.{0,2}` → `node_dst_mlp.layers.{0,1}` remap, but no name
collision — `node_dst_mlp` is never itself a leaf. This is decoder-only.

Not a runtime bug today, because `load_from` is not wired (forward-mapper review §3.1). It is a trap
set for that work, and the cheapest time to defuse it is now.

**Fix.** Rename to `node_data_extractor_norm` / `node_data_extractor_linear` so neither Burn path is
a prefix of the other, or wrap the pair in a small struct so the Burn paths become
`node_data_extractor.{norm,linear}`. Either way the remap becomes two independent, anchored rules.

### 3.2 `in_channels_src` is accepted and never read

`decoder.rs:14` is a config field that `init` never touches. Upstream stores it on `BaseMapper`
(`mapper.py:67`) and only `GraphTransformerForwardMapper` spends it, on `emb_nodes_src`
(`mapper.py:597`) — so an unused field on the backward mapper is faithful to upstream in the narrow
sense.

But it is not free-floating here. The block is constructed with `in_channels = hidden_dim`
(`decoder.rs:53`), so `lin_key` and `lin_value` are `Linear(1024, ·)` and `x_src` **must** arrive
1024-wide. A caller who sets `in_channels_src` to anything else gets no complaint at construction
and a matmul shape error several layers deep inside `lin_key.forward`.

**Fix.** Drop the field, or keep it and `assert_eq!(in_channels_src, hidden_dim)` in `init`. This is
the mirror image of forward-mapper review §3.7, where `out_channels_dst` is the unused one.

### 3.3 `attn_channels` has no upstream counterpart and exactly one correct value

Anemoi derives the per-head width from the output width — `out_channels_conv = out_channels //
num_heads`, `projection = Linear(out_channels, out_channels)` (`block.py:432-447`). The port
introduces a separate `attn_channels`, computing `out_channels_conv = attn_channels / num_heads`
(`block.rs:64-67`) and `projection = Linear(attn_channels, out_channels)` (`block.rs:108`).

The generalisation is internally coherent: `lin_self` emits `num_heads * out_channels_conv =
attn_channels`, the flattened conv output is the same width, so `msg + res` broadcasts and the
projection maps down to `out_channels`. But it equals upstream **only** when `attn_channels ==
out_channels`, which the checkpoint requires (`lin_query.weight [1024, 1024]`,
`projection.weight [1024, 1024]`). Nothing enforces it, in either mapper.

Same shape of defect as forward-mapper review §3.5 (`edge_dim` free but determined): a config field
with one correct value is a way to be inconsistent. Derive it, rather than accept it and hope.

### 3.4 Chunking matters more here than for the encoder

With `E = 1,626,240`, `H = 16`, `C = 64`, one `[E, H, C]` f32 is **~6.66 GB**, and
`graph_tranformer_conv` holds four live at once — `q_i`, `k_j`, `v_j`, `msg` (`common.rs:196-219`) —
for **~26.6 GB**. That is 2.2x the encoder's ~12 GB (forward-mapper review §3.3). The decoder, not
the encoder, is the binding constraint on the memory follow-up.

`GraphTransformerBackwardMapperConfig::num_chunks` (`decoder.rs:27-28`) exists and is ignored, as on
the encoder.

One thing cuts the other way: mean destination degree here is **3.0**, against 18.6 for the encoder.
That makes the padded-dense `[N_dst, D_max, H, C]` formulation from design note §8 follow-up 2 far
more attractive on this side — it would get exact per-segment softmax out of stock Burn `max_dim` /
`softmax` with `-inf` masking, removing the global-max compromise (forward-mapper review §1.4)
rather than merely guarding it. Measure `D_max` before committing to it.

### 3.5 `num_chunks` and `edge_pre_mlp` are dead, as on the encoder

`decoder.rs:27-28` and `:31-32`. `edge_pre_mlp` does not exist anywhere in 0.9.3 —
`GraphTransformerBaseBlock.__init__` (`block.py:386-461`) has no such parameter on either the mapper
or the processor path — so it is threaded into the block config purely to be ignored
(forward-mapper review §3.6). Delete it from both mappers in one go.

### 3.6 Everything is private, so the module is entirely dead

`struct GraphTransformerBackwardMapperConfig` (`decoder.rs:13`) and `struct
GraphTransformerBackwardMapper` (`decoder.rs:36`) have no `pub`, while `init` and `forward` inside
them do. `main.rs` declares `mod decoder;` and nothing else references it, so the whole file is dead
code. Make both types `pub`, as the encoder's are.

### 3.7 `edge_inc` and `edge_index` are as homeless as the encoder's

`model.decoder.edge_inc [2, 1]` is a persistent buffer upstream (`mapper.py:169-171`) but is a
`forward` argument here (`decoder.rs:82`), so the key lands nowhere at load. `edge_attr` and
`edge_index_base` are registered `persistent=False` (`mapper.py:167-168`) and live in the `.ckpt`'s
`graph_data` under `('hidden','to','data')` — confirmed absent from the safetensors header.

**The decoder cannot run on real weights until `scripts/ckpt_to_safetensors.py` dumps them.** Same
blocker as forward-mapper review §3.1/§3.2, and the same extension unblocks both sides at once.

### 3.8 Minor

- `x.clone()` at `decoder.rs:91` is dead — `x` is not used again. The encoder needs its clone
  because it returns `x.0`; the decoder does not.
- `let hidden_dim = ...` at `decoder.rs:51` shadows `self.hidden_dim` with the MLP width. Correct,
  but easy to misread three lines later where both are in scope. `mlp_hidden_dim` says it.
- `mlp_hidden_ratio: f64` with the `+ 0.5` rounding trip reproduces an integer metadata value
  (`mlp_hidden_ratio: 4`). `usize` removes the round-off question entirely. Both mappers.
- `decoder.rs` uses `burn::prelude::*` where `encoder.rs` and `block.rs` list imports explicitly.
  Pick one.

---

## 4. Assertions

Same standard as the forward-mapper review: an assertion earns its place by catching a mistake this
code can actually make.

**Nothing new is justified inside `decoder.rs`.** The mapper is eleven lines of plumbing over the
block, and every shape claim it makes is checked downstream — `graph_tranformer_conv` asserts
`n_src` / `n_dst` against `edge_index` (`common.rs:177-186`), and the `emb_nodes_dst` /
`node_data_extractor` widths surface as Burn matmul errors at the first bad call.

The one gap worth closing is §3.2's, and it belongs in `init` rather than `forward`: asserting
`in_channels_src == hidden_dim` at construction converts a deep matmul error into a statement about
the config. That is a legibility call, not a corruption one.

---

## 5. The change set

§3.1, §3.2, §3.5, §3.6 and §3.8 are one small self-contained change:

| File                        | Change                                                                | Finding |
| --------------------------- | --------------------------------------------------------------------- | ------- |
| `decoder.rs`                | `pub` on both types                                                   | §3.6    |
| `decoder.rs`                | rename the extractor fields so neither path prefixes the other        | §3.1    |
| `decoder.rs`                | drop `in_channels_src`, or assert it equals `hidden_dim` in `init`    | §3.2    |
| `decoder.rs`                | drop the dead `clone`; rename the shadowed `hidden_dim`               | §3.8    |
| `block.rs` + both mappers   | delete `edge_pre_mlp`                                                 | §3.5    |
| `block.rs` + both mappers   | derive `attn_channels` from `out_channels` instead of accepting it    | §3.3    |

§3.4 and §3.7 are separate pieces of work and should not be folded in.

**Verification.** `cargo check && cargo clippy -p aifsv2 --all-targets && cargo test -p aifsv2`.
`param_paths_and_shapes_match_checkpoint` in `src/decoder_test.rs` is the test that moves under §3.1
and §3.3 — it asserts the exact set of Burn parameter paths and shapes at the real model
dimensions, so any rename has to be reflected there deliberately rather than by accident.

---

## 6. Test coverage, as it now stands

`src/decoder_test.rs` (four tests, `wgpu`, no fixture):

| Test                                          | What it pins                                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `forward_maps_dst_to_out_channels_dst`        | pair in, single tensor out at `out_channels_dst` not `hidden_dim`; `x_src` consumed at `hidden_dim`                |
| `param_paths_and_shapes_match_checkpoint`     | the exact Burn parameter tree at real dimensions, against §1. An exact set, so it also catches a stray `emb_nodes_src` |
| `batch_size_two_expands_edges_and_trainable`  | `graph::expand_edges` and `TrainableTensor::forward` agreeing on batch-major ordering — untested before            |
| `zero_degree_destination_is_isolated_and_finite` | a zero-degree destination stays finite and does not move when other destinations' edge attributes change, which is what the global-max shift's exactness claim rests on |

The conv arithmetic itself is pinned separately by `common_test.rs`; these pin the mapper wiring
around it.

**Two gaps, both deliberate:**

1. **`GraphTransformerForwardMapper` has no test.** `encoder_test.rs` held one test, of the block,
   which moved to `block_test.rs` when the block moved to `block.rs`. Mirroring
   `forward_maps_dst_to_out_channels_dst` and `param_paths_and_shapes_match_checkpoint` for the
   encoder is cheap and would be worth doing — the latter would have caught forward-mapper review
   §1.1 at config level.
2. **No anemoi fixture oracle.** Design note §6 sketches one; the decoder's would need a `HeteroData`
   sub-graph to instantiate `GraphTransformerBackwardMapper`, which is substantially more work than
   the encoder's conv-level fixture. Worth filing, not worth blocking on: the arithmetic these tests
   sit on top of is already pinned at the conv level.
