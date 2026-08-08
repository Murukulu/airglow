# Design note: `GraphTransformerForwardMapper` (issue #16)

*This document describes the changes to make. Nothing has been implemented.*

## 1. Context — why the PR stalled

The branch currently models PyG's `MessagePassing` as a Rust trait
(`aifsv2/src/encoder.rs:145`), with an `Adj` struct, a `GraphTransformerConvConfig`, and knobs
for `fuse`, `flow`, `aggr_type`, `node_dim`, and `decomposed_layers`. It doesn't compile, and
it's the reason the PR is stuck.

**None of it is needed.** `MessagePassing` is a Python-reflection framework: `propagate()`
introspects `message()`'s signature to decide which tensors to gather with `_i`/`_j` suffixes,
so dozens of conv layers can share plumbing. airglow has exactly one conv, and every knob is a
fixed constant for it:

| Knob | Value here | Source |
|---|---|---|
| `aggr` | `"add"` | `kwargs.setdefault("aggr", "add")`, `conv.py:99` |
| `node_dim` | `0` | `super().__init__(node_dim=0, ...)`, `conv.py:100` |
| `flow` | `source_to_target` | PyG default → `_i` = `edge_index[1]`, `_j` = `edge_index[0]` |
| `fuse` | `False` | no `message_and_aggregate` is defined |
| `decomposed_layers` | `1` | never set |
| `dropout` | `0.0` / inference | `dropout(..., training=self.training)` is a no-op |

Stripped of the framework, the conv is ~10 lines of tensor ops with **no parameters at all** —
a free function, not a `Module`.

---

## 2. There are two reference implementations, and they compute the same thing

This is the part worth getting right before writing any Rust. `GraphTransformerBaseBlock`
picks a backend at construction time (`block.py:589-618`):

```python
graph_attention_backend: str = "triton"     # the DEFAULT
...
if not is_triton_available():               # no triton package, or no CUDA
    self.graph_attention_backend = "pyg"
if self.graph_attention_backend == "triton":
    self.conv = graph_transformer_attention_conv     # anemoi/models/triton/gt.py
else:
    self.conv = GraphTransformerConv(...)            # anemoi/models/layers/conv.py
```

`is_triton_available()` requires `torch.cuda.is_available()`, so **on a Mac the PyG path is
what actually runs** — but on ECMWF's GPUs the Triton path is what actually runs. Both need to
be understood, because they define the same function with different memory schedules.

### 2a. PyG backend — `GraphTransformerConv.message` (`conv.py:126-145`)

Materializes one value per edge, then scatters:

```
k_j   = key[src]  + edge_attr                      # [E, H, C]
v_j   = value[src] + edge_attr                     # [E, H, C]
alpha = (query[dst] * k_j).sum(-1) / sqrt(C)       # [E, H]
alpha = segment_softmax(alpha, index=dst)          # normalize within each dst group
out   = scatter_add(v_j * alpha, dst)              # [N_dst, H, C]
```

### 2b. Triton backend — `_gt_fwd` (`triton/gt.py:81-180`)

One GPU program per **destination node**, walking that node's incoming edges in CSC order,
with a flash-attention online softmax — running max `m_i`, running denominator `l_i`, and the
`exp(m_i - m_ij)` rescaling correction:

```
neigh_start, neigh_end = colptr[dst], colptr[dst+1]
if num_edges == 0: out[dst] = 0; return          # explicit zero-degree branch
q = Q[dst]; acc = 0; l_i = 0; m_i = -inf
for e_idx in neigh_start..neigh_end:
    e   = E[e_idx]                                # edge attrs, already in CSC order
    src = ROW[e_idx]
    k_e = K[src] + e ;  v_e = V[src] + e
    qk    = sum(q * k_e) / sqrt(C)
    m_ij  = max(m_i, qk)
    alpha = exp(qk - m_ij) ;  corr = exp(m_i - m_ij)
    acc   = acc * corr + alpha * v_e ;  l_i = l_i * corr + alpha ;  m_i = m_ij
out[dst] = acc / l_i
```

**The two are algebraically identical** — same `k + e`, same `v + e`, same `1/sqrt(C)`, same
per-destination normalization. Implementing the PyG form is therefore *not* "implementing the
slow fallback"; it is implementing the same math with a different schedule. Two incidental
differences worth recording: the Triton kernel accumulates in fp32 regardless of input dtype
(matters only if airglow ever runs f16), and it stores `m_i + log(l_i)` for the backward pass
(irrelevant to an inference engine).

### 2c. What Triton tells us that PyG doesn't

Three things the kernel makes obvious, all of which shape the Burn design:

1. **CSC, not COO, is the natural layout.** `edge_index_to_csc` (`triton/utils.py:25`) reduces
   to `colptr = index2ptr(dst, N_dst)` — a prefix sum over destination degrees — when
   `edges_are_dst_sorted=True`, which is the case for graphs from anemoi's graph provider
   (`perm` comes back `None`, so `edges` needs no permutation). The `reverse` tuple
   (`rowptr`, `edge_ids`, `edge_dst`) exists **only for the backward pass**; an inference
   engine can skip it entirely.
2. **The memory blowup is not incidental.** With `E = 748,348`, `H = 16`, `C = 64`, one
   `[E, H, C]` f32 tensor is **~3.06 GB**, and the PyG form holds ~3 of them live. The Triton
   kernel never materializes anything per-edge — its working set is `O(N_dst · H · C)`. This
   is *why* anemoi also wraps the PyG path in `num_chunks = 4` destination-range chunking
   (`block.py:804-824`).
3. **Numerical stability is per-segment.** `m_i` is a max over *one destination's* edges.
   Burn 0.21 cannot express that directly (see §4), so the interim implementation uses a
   global max, and the online-softmax form is the principled endpoint.

**Recommendation:** implement the vectorized PyG form now (§4), but carry the CSC metadata
from day one so the eventual kernel port is additive rather than a rewrite.

---

## 3. Checkpoint ground truth

`uv run scripts/parse_safetensors.py --query "encoder"`, run from `aifsv2/` — 20 tensors:

```
model.encoder.emb_nodes_src.{weight,bias}              [1024, 224], [1024]
model.encoder.emb_nodes_dst.{weight,bias}              [1024, 12],  [1024]
model.encoder.trainable.trainable                      [748348, 8]
model.encoder.proc.lin_{query,key,value,self}.{w,b}    [1024, 1024], [1024]
model.encoder.proc.lin_edge.{weight,bias}              [1024, 11],  [1024]
model.encoder.proc.projection.{weight,bias}            [1024, 1024], [1024]
model.encoder.proc.layer_norm_attention.{w,b}          [1024]   <- alias, see below
model.encoder.proc.layer_norm_attention_dest.{w,b}     [1024]
model.encoder.proc.layer_norm_attention_src.{w,b}      [1024]
model.encoder.proc.layer_norm_mlp_dst.{w,b}            [1024]
model.encoder.proc.node_dst_mlp.0.{weight,bias}        [4096, 1024], [4096]
model.encoder.proc.node_dst_mlp.2.{weight,bias}        [1024, 4096], [1024]
```

`data/aifs-single-mse-2.0_metadata.json` gives `num_heads: 16`, `mlp_hidden_ratio: 4`,
`qk_norm: false`, `trainable_size: 8`,
`sub_graph_edge_attributes: [edge_length, edge_dirs]`.

Derived: `hidden_dim = 1024`, `out_channels_conv = 1024/16 = 64`, `edge_dim = 11`
(1 `edge_length` + 2 `edge_dirs` + 8 trainable), MLP hidden `= 4096`,
`N_src = 542,080`, `N_dst = 40,320`, `E = 748,348` (mean destination degree ≈ 18.6).

Three notes that shape the design:

1. **No `q_norm`/`k_norm`, no `edge_pre_mlp`, no `node_src_mlp`/`layer_norm_mlp_src` keys.**
   Per your decision these stay as config-gated `Option` fields for API fidelity; they are
   simply `None` for this checkpoint.
2. **`layer_norm_attention` and `layer_norm_attention_dest` are the same module.** anemoi does
   `self.layer_norm_attention_dest = self.layer_norm_attention` (`block.py:940`), so
   `state_dict()` emits both keys with identical values. Define one Burn field
   (`layer_norm_attention_dst`), remap `layer_norm_attention_dest` onto it, and let the
   duplicate `layer_norm_attention` key be ignored at load time.
3. **`node_dst_mlp.0` / `.2`** confirms `MLP = [Linear, GELU, Linear]`, `n_extra_layers = 0`,
   `layer_norm = False`.

---

## 4. The changes

### Step 1 — Fix and relocate the MLP

Move `MultiLayerPreceptron` out of `encoder.rs` into a new `aifsv2/src/common.rs` — this is the
existing `TODO(saiputravu)` at `encoder.rs:13`, and the decoder (#18) and processor (#17) both
need it. Rename to `MultiLayerPerceptron` while it's still cheap.

- **Drop the `Activation` enum.** `use ...nn::activation::Activation::{self, Gelu}` collides
  with `use ...nn::Gelu` (`encoder.rs:5-6`) — that is the compile error. The MLP is fixed to
  GELU, so hold a plain `Gelu` field (unit struct, `Gelu::new()`).
- **Delete `build_hidden_layers`** (`encoder.rs:67`) — dead code.
- **Fix `n_extra_layers`.** `encoder.rs:43-45` pushes at most one hidden layer regardless of
  the count. anemoi builds `Linear(in,hidden)` + `n_extra_layers ×` `Linear(hidden,hidden)` +
  `Linear(hidden,out)`; make it a loop.
- **Delete the `PairTensor` alias** (`encoder.rs:15`) — bounds on type aliases are ignored with
  a `type_alias_bounds` lint. Write `(Tensor<B, 2>, Tensor<B, 2>)` inline.

`forward` (`encoder.rs:78-95`) is already correct — keep as-is.

### Step 2 — Delete the `MessagePassing` machinery

**Delete `encoder.rs:98-182` outright**: `GraphTransformerConvConfig`, the
`GraphTransformerConv` module, `Adj`, the `MessagePassing` trait, `sum_forward`, and the impl.

### Step 3 — The graph structure (`common.rs`)

Carry both COO and CSC from the start; CSC costs one CPU prefix sum and is what the future
kernel needs.

```rust
/// Bipartite edge list for one sub-graph, assumed sorted by destination node
/// (true for anemoi graph-provider edges — see `edge_index_to_csc`).
/// Under PyG's `source_to_target` flow, `src` is `_j` and `dst` is `_i`.
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>,  // [E]  — `row` in CSC terms
    pub dst: Tensor<B, 1, Int>,  // [E]
    pub colptr: Vec<i64>,        // [N_dst + 1] — host-side, `index2ptr(dst, N_dst)`
    pub num_src: usize,
    pub num_dst: usize,
}
```

`colptr` is built once at graph load in plain Rust (bincount + prefix sum over `dst`). It is
unused by the Step 4 implementation but is the whole input to the chunking and kernel
follow-ups, and building it is where a `debug_assert` that the edges really are dst-sorted
belongs.

### Step 4 — `graph_transformer_conv` (`common.rs`)

```rust
/// anemoi `GraphTransformerConv` — parameterless attention over a bipartite graph.
/// query [N_dst, H, C], key/value [N_src, H, C], edges [E, H, C] -> [N_dst, H, C]
pub fn graph_transformer_conv<B: Backend>(
    query: Tensor<B, 3>,
    key: Tensor<B, 3>,
    value: Tensor<B, 3>,
    edges: Tensor<B, 3>,
    edge_index: &EdgeIndex<B>,
) -> Tensor<B, 3>
```

Body, mirroring `conv.py:126-145`:

1. `q_i = query.select(0, dst)` → `[E, H, C]`
2. `k_j = key.select(0, src) + edges.clone()` — anemoi adds `edge_attr` into the **key** as
   well as the value; easy to miss, and both Triton (`k_e = k + e`) and PyG agree
3. `v_j = value.select(0, src) + edges`
4. `alpha = (q_i * k_j).sum_dim(2) / (C as f32).sqrt()` → `[E, H, 1]`.
   Burn's `sum_dim` keeps the reduced dim; leave it at `[E, H, 1]` so it broadcasts against
   `[E, H, C]` with no squeeze/unsqueeze round-trip
5. `alpha = segment_softmax(alpha, dst, n_dst)`
6. `msg = v_j * alpha`
7. `Tensor::zeros([n_dst, h, c], dev).select_assign(0, dst, msg, IndexingUpdateOp::Add)`

`select_assign(dim, indices: Tensor<B,1,Int>, values, IndexingUpdateOp::Add)` is exactly
`index_add_` along dim 0 and takes a **1-D** index tensor — no index broadcasting, unlike
`scatter`. (burn-tensor 0.21, `src/tensor/api/base.rs:1678`; note `scatter` **panics** on any
update op other than `Add`, `base.rs:1803`.)

#### `segment_softmax` — and the one real compromise

Burn 0.21 has no scatter-max: `IndexingUpdateOp` implements only `Add`. The per-destination
running max that `_gt_fwd` computes for free (`m_i`) is not directly expressible. Subtract a
**single global max** instead:

```rust
let m     = alpha.clone().max();                                   // scalar
let e     = (alpha - m).exp();                                     // [E, H, 1]
let denom = Tensor::zeros([n_dst, h, 1], dev)
    .select_assign(0, dst.clone(), e.clone(), IndexingUpdateOp::Add);
e / denom.select(0, dst)                                           // [E, H, 1]
```

This is **mathematically exact** — a constant subtracted inside a group cancels in the ratio.
Comment the caveat: it underflows only if some destination's logits sit more than ~88 below
the *global* max in f32, which post-`LayerNorm` attention logits do not. Division is per-edge
against the gathered denominator, so zero-degree destinations never divide — they stay 0 from
the scatter, matching both PyG and `_gt_fwd`'s explicit zero branch.

### Step 5 — `GraphTransformerMapperBlock`

Keep the struct at `encoder.rs:201` with these corrections:

- Split `layer_norm_attention` into `layer_norm_attention_src` and `layer_norm_attention_dst`
  (both `LayerNorm<B>`, `normalized_shape = in_channels`), per §3 note 2. The struct literal at
  `encoder.rs:289-291` already names fields (`layer_norm_attention_src`,
  `layer_norm_attention_dst`, `layer_norm_mlp_src`) that the declaration doesn't have.
- `layer_norm_mlp_dst` is declared `Option` but built unconditionally (`encoder.rs:259`) — make
  it a plain `LayerNorm<B>`.
- Add `layer_norm_mlp_src: Option<LayerNorm<B>>`; it and `node_src_mlp` are `Some` only when
  `update_src_nodes`.
- `query_norm` / `key_norm` stay `Option`, gated on `qk_norm`. `edge_pre_mlp` stays `Option`.
- **Remove the `conv` field** — the conv is a free function. Store `num_heads` and
  `out_channels_conv` as `usize` instead.
- Fix the init bugs: `node_dst_mlp` (`encoder.rs:260`) never calls `.init(device)`;
  `node_src_mlp = ()` (`:267`); `conv` (`:278`) never calls `.init`.
- `hidden_dim` for `node_dst_mlp` is `mlp_hidden_ratio × in_channels` = 4096, passed in by the
  caller — don't recompute it inside the block.

`forward((x_src, x_dst), edge_attr, edge_index) -> ((Tensor, Tensor), Tensor)`, following
`block.py:963-1023` with sharding, chunking and `cond` removed:

```
x_skip = (x_src, x_dst)
x_src  = layer_norm_attention_src(x_src)
x_dst  = layer_norm_attention_dst(x_dst)

x_r    = lin_self(x_dst)                                     // [N_dst, H*C]
query  = lin_query(x_dst)                                    // query from dst
key    = lin_key(x_src)
value  = lin_value(x_src)
edges  = lin_edge(edge_pre_mlp.map_or(edge_attr, |m| m.forward(edge_attr)))

// [N, H*C] -> [N, H, C]  (einops "nodes (heads vars) -> nodes heads vars")
reshape query/key/value/edges to [.., num_heads, out_channels_conv]
if qk_norm { query = query_norm(query); key = key_norm(key) }

out = graph_transformer_conv(query, key, value, edges, edge_index)   // [N_dst, H, C]
out = out.reshape([N_dst, H*C])
out = projection(out + x_r)                                  // [N_dst, out_channels]
out = out + x_skip.1

dst_new = node_dst_mlp(layer_norm_mlp_dst(out)) + out
src_new = if update_src_nodes { node_src_mlp(layer_norm_mlp_src(x_skip.0)) + x_skip.0 }
          else { x_skip.0 }
return ((src_new, dst_new), edge_attr)
```

Note the residual asymmetry: the attention residual `x_skip.1` is the **pre**-LayerNorm
`x_dst`, while `x_r = lin_self(...)` consumes the **post**-LayerNorm one.

### Step 6 — `GraphTransformerForwardMapper`

`encoder.rs:300-329`. Fix `init` (`emb_nodes_src` / `emb_nodes_dst` never call `.init(device)`;
`proc: ()`), and add `trainable: Param<Tensor<B, 2>>` to hold
`model.encoder.trainable.trainable` `[E, 8]` — that key needs a home for `load_from` to
succeed, and the checkpoint's anemoi version owns the trainable edge attributes inside the
mapper rather than in the graph provider.

```
forward((x_src, x_dst), edge_attr /* [E, 3] */, edge_index):
    edge_attr = cat([edge_attr, trainable], dim = -1)        // [E, 11]
    x_src_emb = emb_nodes_src(x_src)                         // 224 -> 1024
    x_dst_emb = emb_nodes_dst(x_dst)                         //  12 -> 1024
    ((_, x_dst_out), _) = proc((x_src_emb, x_dst_emb), edge_attr, edge_index)
    return (x_src, x_dst_out)     // NB: the ORIGINAL x_src, not the embedded one
```

The last line is deliberate — `mapper.py:590` returns `x[0]`, un-embedded. Also assert
`out_channels_dst.is_none()`, matching `mapper.py:544`.

---

## 5. Verification

`aifsv2` currently enables only the `wgpu` backend. Add
`burn = { version = "0.21.0", features = ["ndarray"] }` under `[dev-dependencies]` so tests run
on CPU deterministically.

**Fixture generator** — `aifsv2/scripts/gen_conv_fixture.py`, a `uv` inline-script matching the
style of `parse_safetensors.py` / `ckpt_to_safetensors.py` (`# /// script` header). Builds a
tiny dst-sorted bipartite graph (`N_src = 7`, `N_dst = 4`, `E = 15`, `H = 2`, `C = 3`, with one
destination of **zero** degree and one of degree 1), runs the real PyG `GraphTransformerConv`
on random inputs, and writes inputs + expected output to
`aifsv2/data/fixtures/graph_transformer_conv.safetensors`. Commit the fixture (a few KB) so the
test needs no PyG install. Because §2 establishes the two backends are equivalent, this fixture
is a valid oracle for the Triton path too.

**Rust tests** in `common.rs`:
1. `graph_transformer_conv` vs. the fixture, approx-equal at `1e-5`.
2. `segment_softmax` alone: weights grouped by `dst` sum to 1; the zero-degree destination
   yields an all-zero output row; adding a large constant to all logits changes nothing.
3. `MultiLayerPerceptron`: `n_extra_layers = 2` yields 4 `Linear` layers;
   `final_activation = false` leaves the last unactivated.
4. `EdgeIndex::colptr` matches a hand-computed prefix sum, and `colptr[N_dst] == E`.

**Block/mapper test** in `encoder.rs`: the same fixture approach if the anemoi block imports
cleanly in the `quiet_grub` venv; otherwise a shape-and-finiteness smoke test
(`[N_src, 224]`, `[N_dst, 12]`, `[E, 3]` → `[N_dst, 1024]`, all finite) plus an assertion that
`GraphTransformerForwardMapperConfig::init` yields parameter shapes matching the 20 checkpoint
keys in §3.

Then `cargo test -p aifsv2` and `cargo clippy -p aifsv2 --all-targets`.

---

## 6. Follow-ups to file

1. **Export the encoder sub-graph.** `edge_index` and the base edge attributes
   (`edge_length`, `edge_dirs`) are **not** in the safetensors — only
   `encoder.trainable.trainable` is. They live in the `.ckpt`'s `graph_data` `HeteroData` under
   `('data','to','hidden')`. Extend `scripts/ckpt_to_safetensors.py` to dump them. **Until this
   lands the encoder cannot run on real weights**, so #16's end-to-end acceptance is blocked on
   it even once the layer code is done. Worth splitting out of #16 explicitly.
2. **Make it fit in memory** — the ~3 GB-per-`[E,H,C]`-tensor problem from §2c. Two options,
   in increasing order of effort:
   - *Destination-range chunking*, as anemoi does with `num_chunks = 4`. `colptr` from Step 3
     makes chunk boundaries a `Vec` slice, so it's a loop over contiguous edge ranges — none of
     the `GraphPartition` machinery is needed.
   - *A CubeCL port of `_gt_fwd`* on the `wgpu` backend: one workgroup per destination node,
     online softmax, working set `O(N_dst · H · C)`. This also removes the global-max
     compromise in Step 4 and is the only form that matches what runs on ECMWF's GPUs.
     Worth measuring the destination degree distribution first (mean is 18.6; if the max is
     small, a padded-dense `[N_dst, D_max, H, C]` formulation gets exact per-segment softmax
     out of stock Burn `max_dim`/`softmax` with `-inf` masking, at a fraction of the effort).
3. **`GraphTransformerBackwardMapper` (#18)** reuses `MultiLayerPerceptron`,
   `graph_transformer_conv`, `EdgeIndex` and `GraphTransformerMapperBlock` verbatim, adding
   only `node_data_extractor = Sequential(LayerNorm, Linear)`. Putting the shared pieces in
   `common.rs` now is what makes #18 small.
