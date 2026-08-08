# The graph transformer, explained

Companion to [`graph-transformer-forward-mapper.md`](./graph-transformer-forward-mapper.md). That
document decides **what to build and why**; it assumes you already know what a bipartite
message-passing graph is. This one explains the object itself: where it comes from, what the
tensors mean, and why the softmax is the awkward part.

Read this first if any of the following is unclear: which paper this is, why `src` and `dst` have
the same length, why the head axis sits in the middle, or what `alpha` actually contains.

Every claim below cites a file and line. Upstream references are pinned to **anemoi-core
`0fa84c1`** and **PyTorch Geometric `cc678a3`** — the same commits the design note uses. Repo
paths are relative to `aifsv2/`.

---

## 1. The paper

**Shi, Huang, Sun, Xie, Zhang, Deng (2020) — _Masked Label Prediction: Unified Message Passing
Model for Semi-Supervised Classification_, [arXiv:2009.03509](https://arxiv.org/abs/2009.03509).**
Usually called **UniMP**.

Both upstreams name it in the class docstring, so this is not an inference:

- anemoi, `models/src/anemoi/models/layers/conv.py:84-88`:

  ```python
  class GraphTransformerConv(MessagePassing):
      """Message passing part of graph transformer operator.

      Adapted from 'Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification'
      (https://arxiv.org/abs/2009.03509)
      """
  ```

- PyG, `torch_geometric/nn/conv/transformer_conv.py:27-30`: _"The graph transformer operator from
  the `"Masked Label Prediction: ..."` paper."_

### 1a. The two equations, mapped to identifiers

UniMP's operator, as PyG transcribes it (`transformer_conv.py:31-41`):

$$
alpha_{i,j} = softmax_{j in N(i)} ( (W3 x_i)^T (W4 x_j) / sqrt(d) )
$$

$$
x'_i = W1 x_i + sum_{j in N(i)} alpha_{i,j} W2 x_j
$$

Read `i` as the **destination** node and `j` as a **source** node feeding it. `N(i)` is the set of
sources with an edge into `i` — the sum runs over that set, not over all nodes. This is the entire
idea: standard scaled dot-product attention ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) where each query attends to its graph neighbours instead
of to every position.

| Paper             | anemoi                                   | This repo (`src/common.rs`)       |
| ----------------- | ---------------------------------------- | --------------------------------- |
| `W3 x_i`          | `lin_query(x_dst)` (`block.py:630`)      | `query`, `[N_dst, H, C]`          |
| `W4 x_j`          | `lin_key(x_src)` (`block.py:631`)        | `key`, `[N_src, H, C]`            |
| `W2 x_j`          | `lin_value(x_src)` (`block.py:632`)      | `value`, `[N_src, H, C]`          |
| `W1 x_i`          | `lin_self(x_dst)` — **outside** the conv | not an argument, see below        |
| `sqrt(d)`         | `self.out_channels**0.5` (`conv.py:142`) | `norm = 1/sqrt(C)`                |
| `alpha_{i,j}`     | `alpha` after `softmax` (`conv.py:144`)  | `alpha`, `[E, H, 1]`              |
| `sum_{j in N(i)}` | `aggr="add"` (`conv.py:97`)              | the final `select_assign(…, Add)` |

Two deltas between the paper and what `graph_transformer_conv` computes, both deliberate:

1. **The `W1 x_i` self term is not in the conv.** anemoi hoists it to the enclosing block, so the
   conv is a pure neighbour-aggregation with no residual. That is why the Rust signature takes no
   `x_dst` — see §5 Step 5 of the design note.
2. **Edge features are added to _both_ key and value.** `conv.py:139-147`:

   ```python
   if edge_attr is not None:
       key_j = key_j + edge_attr
   alpha = (query_i * key_j).sum(dim=-1) / self.out_channels**0.5
   alpha = softmax(alpha, index, ptr, size_i)
   return (value_j + edge_attr) * alpha.view(-1, heads, 1)
   ```

   The same projected `edges` tensor lands in the attention logit _and_ in the message being
   weighted. UniMP's formulation has no edge term at all; PyG's optional `edge_dim` adds it to the
   key and value separately. Easy to miss when reading, and both anemoi backends do it.

### 1b. Disambiguation: this is not the other "Graph Transformer"

Searching for "graph transformer paper" returns **Dwivedi & Bresson (2020), _A Generalization of
Transformer Networks to Graphs_, [arXiv:2012.09699](https://arxiv.org/abs/2012.09699)** first.
Same year, same name, **different operator** — it uses Laplacian eigenvector positional encodings
and does not condition values on edge features. It is not what anemoi implements. If you find
yourself reading about spectral positional encodings, you are in the wrong paper.

### 1c. Where the rest fits

| Reference                                                                 | Why it matters here                                                                                                                                                  |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vaswani et al. 2017, [1706.03762](https://arxiv.org/abs/1706.03762)       | `softmax(QKᵀ/√d)V`. Make the graph complete and §1a collapses to exactly this.                                                                                       |
| Battaglia et al. 2018, [1806.01261](https://arxiv.org/abs/1806.01261)     | Origin of the message → aggregate → update decomposition that PyG's `MessagePassing` encodes as a class. §2 of the design note is about dissolving that abstraction. |
| Fey & Lenssen 2019, [1903.02428](https://arxiv.org/abs/1903.02428)        | PyG's design paper — why `gather`/`scatter` on an edge list is the chosen primitive rather than sparse matrix products.                                              |
| Lang et al. 2024, [2406.01465](https://arxiv.org/abs/2406.01465)          | AIFS itself. Why there is an encoder mapping a 542,080-point weather grid onto a 40,320-point one at all.                                                            |
| Lam et al. 2023, [2212.12794](https://arxiv.org/abs/2212.12794)           | GraphCast — the encode-process-decode pattern AIFS follows.                                                                                                          |
| Milakov & Gimelshein 2018, [1805.02867](https://arxiv.org/abs/1805.02867) | Online (single-pass) softmax: the running max and denominator. This is what the Triton path computes and what §7 below cannot express in stock Burn.                 |
| Dao et al. 2022, [2205.14135](https://arxiv.org/abs/2205.14135)           | FlashAttention. `_gt_fwd` in anemoi is this structure applied to a graph.                                                                                            |

---

## 2. The actual graph in this checkpoint

Two point clouds on a sphere, both **reduced Gaussian grids** — despite the name, the "hidden"
side of AIFS-single-mse-2.0 is not an icosahedral mesh.

From `data/aifs-single-mse-2.0_metadata.json`, under the key
`quiet_grub/anemoi-metadata/ai-models.json` → `config.graph`:

```json
"nodes": {
  "data":   { "node_builder": { "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
                                "dataset": ".../...-n320-2016-2025-6h-v1-for-single-v2.zarr" } },
  "hidden": { "node_builder": { "_target_": "anemoi.graphs.nodes.ReducedGaussianGridNodes",
                                "grid": "o96" } }
}
```

- **`data`** — the N320 reduced Gaussian grid, **542,080** points (~28 km spacing). This is where
  observations live.
- **`hidden`** — the O96 _octahedral_ reduced Gaussian grid. Its point count is exact and
  derivable: `4·96² + 36·96 = 36,864 + 3,456 =` **40,320** (~110 km spacing).

```mermaid
flowchart LR
    D0["data<br/>N320 grid<br/>542,080 nodes"]
    H0["hidden<br/>O96 grid<br/>40,320 nodes"]
    H1["hidden<br/>40,320 nodes"]
    D1["data<br/>542,080 nodes"]

    D0 -->|"<b>encoder</b><br/>CutOffEdges(0.6)<br/>748,348 edges<br/>mean in-degree 18.6"| H0
    H0 -->|"processor<br/>(hidden to hidden)"| H1
    H1 -->|"<b>decoder</b><br/>KNNEdges(k=3)<br/>1,626,240 edges<br/>in-degree exactly 3"| D1

    classDef enc fill:#1f4e79,stroke:#12314b,color:#fff
    class D0,H0 enc
```

`GraphTransformerForwardMapper` — the subject of the design note — is the **encoder** arrow.

### 2a. The edge builders, and why they produce what they produce

From the same file, `config.graph.edges`:

```json
[
  {
    "source_name": "data",
    "target_name": "hidden",
    "edge_builders": [
      { "_target_": "anemoi.graphs.edges.CutOffEdges", "cutoff_factor": 0.6 }
    ]
  },
  {
    "source_name": "hidden",
    "target_name": "data",
    "edge_builders": [
      {
        "_target_": "anemoi.graphs.edges.KNNEdges",
        "num_nearest_neighbours": 3
      }
    ]
  }
]
```

**Encoder — `CutOffEdges`.** For each _hidden_ node, draw a ball of radius
`0.6 × (reference spacing)` and connect every data node inside it. A **radius query**: the number
of edges per hidden node is whatever geometry hands you, and it varies with latitude and grid
alignment.

**Decoder — `KNNEdges(k=3)`.** Each _data_ node takes its 3 nearest hidden nodes. Fixed degree.

That difference is the whole of §3 and §6 below. One produces variable-length groups; the other
produces uniform ones.

### 2b. Every number is confirmed by a checkpoint tensor

These are not estimates. `uv run scripts/parse_safetensors.py --query trainable`, run from
`aifsv2/`:

```
model.node_attributes.trainable_tensors.data.trainable     [542080, 8]
model.node_attributes.trainable_tensors.hidden.trainable   [40320, 8]
model.encoder.trainable.trainable                          [748348, 8]
model.decoder.trainable.trainable                          [1626240, 8]
```

Each row is 8 learned features attached to one node or one edge, so the leading dimension _is_ the
count:

| Symbol        | Value     | Source                                                |
| ------------- | --------- | ----------------------------------------------------- |
| `N_src`       | 542,080   | `...trainable_tensors.data` — matches N320            |
| `N_dst`       | 40,320    | `...trainable_tensors.hidden` — matches `4·96²+36·96` |
| `E` (encoder) | 748,348   | `model.encoder.trainable`                             |
| `E` (decoder) | 1,626,240 | `model.decoder.trainable`                             |

The decoder row is the useful one: **1,626,240 = 3 × 542,080 exactly**. A checkpoint tensor whose
first dimension is precisely `k × N_data` for the configured `k = 3` confirms the whole reading —
edges are per-(destination, neighbour) pairs, and `KNNEdges(k=3)` means each of the 542,080
destinations has exactly 3 of them.

Derived encoder degrees:

```
748,348 / 40,320  = 18.56   edges per hidden node   (destination in-degree)
748,348 / 542,080 =  1.38   edges per data node     (source out-degree)
```

The second number says most data points fall inside exactly one hidden node's ball, and a minority
fall inside two — consistent with balls of radius `0.6 ×` spacing overlapping slightly.

### 2c. What the `8` is, and why the feature counts close

Each of those four tensors is a **lookup table, not a projection**: an `nn.Parameter` of shape
`[count, 8]` indexed by node id or edge id, so entity `n` owns row `n` — a learned embedding for
that specific grid point or that specific edge. The width is configured, not derived:
`config.model.trainable_parameters = {"data": 8, "hidden": 8}`, plus `trainable_size: 8` for edges.

Those 8 channels are concatenated with the physical inputs before the embedding, which is why the
`emb_nodes_*` shapes in §4 of the design note come out as they do. Both sides reconcile exactly:

```
model.node_attributes.latlons_hidden   [40320, 4]     geometry
model.node_attributes.…hidden.trainable[40320, 8]     learned
                                              --
                                              12   =  emb_nodes_dst.weight [1024, 12]  ✓

  106 model input variables  ×  2 timesteps        =  212      (multistep_input: 2)
+ latlons_data       [542080, 4]                   +   4
+ …data.trainable    [542080, 8]                   +   8
                                                     ---
                                                     224   =  emb_nodes_src.weight [1024, 224]  ✓
```

`106` is `data_indices.model.input.full` (92 prognostic + 14 forcing; the 28 diagnostic variables
are output-only, so they are not inputs). The hidden side has **no** physical variables at all —
only geometry and learned state, which is what makes it "hidden."

The `latlons_*` tensors are a trigonometric encoding of position, not raw degrees: every row of
`latlons_hidden` has `sum(x²) == 2.0` exactly, and channels `(0, 2)` and `(1, 3)` each form a unit
vector — two angles, interleaved.

The same pattern holds one level down, for edges: `model.encoder.trainable.trainable [748348, 8]`
is per-edge, and `8 + 1 (edge_length) + 2 (edge_dirs) = 11 = lin_edge.weight [1024, 11]`. Nodes and
edges are treated identically — geometry plus a learned per-entity vector, embedded to
`hidden_dim`.

> **`N_src` and `N_dst` are roles, not identities.** They name a position in the encoder's
> bipartite graph. The `data` node set is the source there and the **destination** in the decoder
> (`KNNEdges`, `hidden → data`), where the counts swap. `542,080` is a property of the N320 grid;
> being "source" is a property of which mapper you are looking at.

> **What we cannot state.** `edge_index` itself is **not** in the checkpoint —
> `parse_safetensors.py --query index` returns nothing. It is a non-persistent buffer loaded from
> `graph_enc_proc_dec_n320.pt` (`config.hardware.files.graph`), which this repo does not have. So
> **only the mean degree is derivable; the maximum is unknown.** Follow-up 2 in the design note
> (padded-dense softmax) depends on that maximum, and cannot be evaluated until the graph file is
> in hand.

---

## 3. Why `src` and `dst` are both length `E`

This is the question that unlocks everything else, and the answer is a reframing.

> **`src` and `dst` are not node lists. They are _edge_ attributes.**

There are `E` edges. Every edge has exactly one source and exactly one destination. So "the source
of each edge" is a list of `E` numbers, and "the destination of each edge" is a list of `E`
numbers. They are the same length because they are indexed by the same thing — the edge — not
because the two node sets are the same size.

What differs is not their **length** but their **value range**:

```
src : length E = 748,348     values in [0,  542,080)   <- indexes into `key`/`value`
dst : length E = 748,348     values in [0,   40,320)   <- indexes into `query`, and into the output
```

`src` is drawn from a set 13× larger than `dst`'s, so `dst` must repeat far more often. That
repetition is not a defect — see §6.

### 3a. It is one table, stored column-wise

The natural way to write an edge list is a row per edge. COO ("coordinate") format is that table
transposed into two parallel arrays:

```
         e=0    e=1    e=2    e=3    e=4    e=5    e=6   ...   e=E-1
       +------+------+------+------+------+------+------+     +--------+
src    | 4127 |  886 | 4127 |31005 |  886 |  205 |31005 | ... | 541990 |   <- in [0, N_src)
       +------+------+------+------+------+------+------+     +--------+
dst    |    0 |    0 |    0 |    1 |    2 |    2 |    2 | ... |  40319 |   <- in [0, N_dst)
       +------+------+------+------+------+------+------+     +--------+
          |                                                       
          +--> edge 0 runs from data node 4127 to hidden node 0
```

Read **down a column** to get one edge. Reading _along_ a row is what causes the confusion: `src`
in isolation looks like a list of nodes, but node `4127` appearing at both `e=0` and `e=2` is one
node with two outgoing edges, not two nodes.

Three sizes, three meanings, no arithmetic relationship forced between them:

|                   | Range of values                           | Length of the array                                           |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------- |
| `N_src = 542,080` | how many distinct source nodes exist      | —                                                             |
| `N_dst = 40,320`  | how many distinct destination nodes exist | —                                                             |
| `E = 748,348`     | —                                         | how many (source, destination) pairs the cutoff rule produced |

`E` is set by geometry. It could legally be anything from `0` to `N_src × N_dst`; the cutoff radius
happens to yield 748,348.

### 3b. `E` is a count; `e` is a name

`748,348` is **not an edge**. It is how many edges there are. Every edge is still a pair — that
never stops being true. Two distinct things are easy to conflate here:

|                           | What it is                                     | Example                                            |
| ------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **Definition** of an edge | a pair `(src[e], dst[e])`                      | `(4127, 0)` — from data node 4127 to hidden node 0 |
| **Identity** of an edge   | a single integer `e`, its position in the list | `0`                                                |

Every edge is _defined_ by two numbers and _named_ by one. The name is just its column index in
the §3a table — an arbitrary label handed out by build order, `e ∈ [0, 748348)`. Exactly like a row
number in a spreadsheet of `(from, to)` pairs: "row 5" identifies the row, `(4127, 0)` is what the
row says.

That is what makes `[748348, 8]` sensible. Anything attached _to an edge_ is stored as a table with
one row per edge, addressed by `e`:

```
model.encoder.trainable.trainable  [748348, 8]

     e        8 learned features
  ------   ----------------------
     0     [ ................ ]     <- belongs to edge (src[0], dst[0]) = (4127, 0)
     1     [ ................ ]     <- belongs to edge (886, 0)
    ...
  748347   [ ................ ]
```

`src`, `dst` and this feature table are three columns of one conceptual table, all keyed by `e`.

### 3c. Where `E` comes from: it counts nonzeros

The other way to see it. A bipartite graph is a boolean matrix `A` of shape `[N_src, N_dst]`, with
`A[s, d] = 1` when an edge exists:

```
          d=0    d=1    d=2   ...    d=40319
    s=0  [  1      0      1   ...       0   ]
    s=1  [  1      0      0   ...       0   ]     542,080 x 40,320
    ...                                          = 21,856,665,600 cells
s=542079 [  0      0      0   ...       1   ]
```

`E` is the number of `1`s. For this graph that is 748,348 out of 21.86 billion — a density of
**0.0034%**, so 99.9966% of the matrix is zero. Storing it densely would cost ~87 GB in f32;
storing only the coordinates of the nonzeros costs 12 MB for `src` + `dst` in i64.

That is what COO — **coordinate** format — means, and it is why `E` is an independent number.
`N_src` and `N_dst` fix the matrix's _shape_; `E` is how many cells the cutoff rule happened to
switch on. Nothing forces a relationship between the three.

---

## 4. `EdgeIndex`: COO and CSC, and why `colptr` is there

`src/graph.rs:7-14`:

```rust
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>, // Shape [E]
    pub dst: Tensor<B, 1, Int>, // [E]
    pub colptr: Vec<i64>,       // [N_ds+ 1]

    pub num_src: usize,
    pub num_dst: usize,
}
```

The struct holds **two representations of the same graph**: COO (`src`, `dst`) and the compressed
half of CSC (`colptr`). It is deliberately redundant.

### 4a. A worked example

Four source nodes, three destination nodes, seven edges. Say the builder emits them grouped by
source, which is the natural order for a "for each source, find its targets" loop:

```
edges:  (s0→d0) (s0→d2) (s1→d0) (s2→d1) (s2→d2) (s3→d0) (s3→d2)
```

**(a) Unsorted COO** — as built:

```
       e=  0    1    2    3    4    5    6
src      [ 0    0    1    2    2    3    3 ]
dst      [ 0    2    0    1    2    0    2 ]
```

**(b) Destination-sorted COO** — stable sort by `dst`. The permutation that achieves it is
`perm = [0, 2, 5, 3, 1, 4, 6]`:

```
       e=  0    1    2  |  3  |  4    5    6
src      [ 0    1    3  |  2  |  0    2    3 ]
dst      [ 0    0    0  |  1  |  2    2    2 ]
           \___________/   \_/   \___________/
            dst 0, deg 3   d1     dst 2, deg 3
```

The destination-sorted form has a property the unsorted one lacks: **each destination's edges are
one contiguous run.** Everything below depends on that.

**(c) CSC** — replace the run of repeated `dst` values with the offsets where each run starts.
Count edges per destination (`[3, 1, 3]`), then take the exclusive prefix sum with a trailing
total:

```
dst value      :   0    1    2
count          :   3    1    3
                   |    |    |
colptr         : [ 0,   3,   4,   7 ]        length N_dst + 1 = 4
                   ^         ^    ^
                   |         |    +-- colptr[N_dst] == E, always
                   |         +------- dst 2 owns edges [4, 7)
                   +----------------- dst 0 owns edges [0, 3)
```

To get every edge into destination `d`, slice `colptr[d] .. colptr[d+1]`. No search, no scan.

### 4b. `colptr` is not "the third part of CSC" — `dst` is the redundant one

This is the crux of the confusion. Strictly:

```
COO  =  (src, dst)              two arrays, both length E
CSC  =  (colptr, src)           colptr length N_dst+1, src length E
```

`src` is shared — in CSC terminology it is the **row index** array. Given a destination-sorted
edge list, `colptr` and `dst` carry **identical information** and each reconstructs the other:

```
dst → colptr :  bincount(dst, N_dst) then exclusive prefix sum
colptr → dst :  repeat_interleave(0..N_dst, colptr[1:] - colptr[:-1])
```

So the struct stores three arrays for two arrays' worth of information. That is a considered
choice, not an oversight:

- **Step 4 needs `dst` as a tensor.** `query.select(0, dst)` and
  `select_assign(0, dst, msg, Add)` both take a `Tensor<B, 1, Int>` of gather/scatter indices.
  `colptr` cannot be passed to either.
- **The CubeCL follow-up needs `colptr`.** A segmented kernel assigns one workgroup per
  destination and reads `colptr[d]..colptr[d+1]` to find its edges. With only `dst` it would have
  to search.
- **`colptr` is host-side (`Vec<i64>`) and cheap.** 40,321 × 8 bytes ≈ 320 KB, built once at graph
  load by a bincount and a prefix sum. Against 6 MB for `dst` on device, it rounds to nothing.

### 4c. `colptr` is only meaningful if the edges are destination-sorted

`graph.rs:5` says _"You can assume sorted by dest"_, and anemoi backs that up rather than assuming
it. `block.py:779-786`, the Triton path:

```python
csc, perm, reverse = edge_index_to_csc(
    edge_index,
    num_nodes=conv_size,
    reverse=True,
    edges_are_dst_sorted=edges_are_dst_sorted,
)
if perm is not None:
    edges = edges.index_select(0, perm)
```

If the edges are not already sorted, `edge_index_to_csc` returns a permutation and **the edge
feature tensor is physically reordered to match**. That is the cost of the CSC view, and the reason
`colptr` and an unsorted `dst` cannot coexist. Building `colptr` in Rust is where a
`debug_assert!` on sortedness belongs — see §5 Step 3 of the design note.

---

## 5. Why `[N, H, C]` and not `[H, N, C]`

Grouping by head sounds right — heads are independent, so why not make the head axis outermost?
The answer is that **anemoi uses both layouts**, and the choice tracks what the next operation is.

### 5a. Four reasons for node-major here

**1. The node axis must be axis 0 for the framework to gather it.** `conv.py:98`:

```python
super().__init__(node_dim=0, **kwargs)
```

PyG's `_lift` does `index_select(node_dim, index)`. With `[H, N, C]` you would need `node_dim=1`,
and then every scatter, every `softmax(…, dim=…)` and every size check would have to follow. The
port inherits the same constraint for a simpler reason: `Tensor::select(0, idx)` and
`select_assign(0, …)` are the operations, and axis 0 is where the node lives.

**2. The head split is free only in node-major order.** `block.py:645-651`:

```python
einops.rearrange(
    t,
    "nodes (heads vars) -> nodes heads vars",
    heads=self.num_heads,
    vars=self.out_channels_conv,
)
```

`lin_query` is a `Linear(1024, 1024)` producing `[N, 1024]`. Splitting the **last** axis is a pure
reinterpretation of the same bytes — no copy, no kernel:

```
one node's 1024 contiguous floats, as stored:

  offset:  0        64       128              960      1024
           |--------|--------|-- ... ---------|--------|
             head 0   head 1                    head 15
           \________________ view as [16, 64] ________/

  [N, 1024]  ->  [N, 16, 64]     same memory, stride (1024, 64, 1)
```

`block.py:687` reverses it just as cheaply. Reaching `[H, N, C]` instead means element `(h, n, c)`
must move to offset `h·N·C + n·C + c` — a real permutation of 542,080 × 1024 floats.

**3. The contracted axis has to be last.** `alpha = (q_i * k_j).sum_dim(-1)` reduces over `C`. In
`[E, H, C]` the `C` values of one (edge, head) pair are 64 adjacent floats — a coalesced read. Same
for the scatter: one destination's `H·C = 1024` outputs are one contiguous slab.

**4. Head-major buys nothing, because heads never interact.** They are independent all the way
through, so there is no operation that wants all of head `h` contiguous. Head-major would help if
something batched over heads — and that is exactly where anemoi does use it.

### 5b. The counterexample: anemoi _does_ go head-major, for one purpose

`block.py:706-712`, inside `shard_qkve_heads`:

```python
einops.rearrange(
    t,
    "(batch grid) (heads vars) -> batch heads grid vars",
    heads=self.num_heads,
    vars=self.out_channels_conv,
    batch=batch_size,
)
```

then splits across GPUs on the head axis, and immediately at `:724`:

```python
einops.rearrange(t, "batch heads grid vars -> (batch grid) heads vars")
```

Head-major exists **only** so `all_to_all_transpose` has heads as the outermost partitionable axis.
It is undone before the conv is ever called. The rule that emerges:

| Layout                       | Used when                                                                 | Example                                                  |
| ---------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------- |
| `[nodes, heads, vars]`       | the op indexes **nodes** — gather, scatter, segment softmax               | `block.py:645`, and every graph conv                     |
| `[batch, heads, grid, vars]` | the op partitions or batches over **heads** — collectives, batched matmul | `block.py:706` (sharding), `:847` (dense self-attention) |

Node-major is right here because every op in `graph_transformer_conv` is node- or edge-addressed.
Nothing batches over heads.

---

## 6. Duplicates, worked end to end

Take the seven-edge graph from §4a, with `H = 1` and `C = 2` so the arithmetic fits on a page.
Heads never interact (§5a), so `H > 1` changes only the shapes. Set `edges = 0` throughout to keep
the numbers clean; §1a explains where it would otherwise enter.

Inputs:

```
query  [N_dst=3, 1, 2]        key  [N_src=4, 1, 2]        value  [N_src=4, 1, 2]
  q[0] = (1, 0)                 k[0] = (1, 0)               v[0] = (10,  0)
  q[1] = (0, 1)                 k[1] = (0, 1)               v[1] = ( 0, 10)
  q[2] = (1, 1)                 k[2] = (1, 1)               v[2] = ( 5,  5)
                                k[3] = (2, 0)               v[3] = ( 1,  1)

edge_index (destination-sorted, from §4a(b)):
  src = [0, 1, 3, 2, 0, 2, 3]
  dst = [0, 0, 0, 1, 2, 2, 2]
  colptr = [0, 3, 4, 7]
```

### 6a. The domains, and which step moves between them

```mermaid
flowchart TD
    subgraph SRC["source domain (N_src = 4)"]
        K["key"]:::d
        V["value"]:::d
    end
    subgraph DST1["destination domain (N_dst = 3)"]
        Q["query"]:::d
    end
    subgraph EDG["edge domain (E = 7)"]
        QI["q_i"]:::e
        KJ["k_j"]:::e
        VJ["v_j"]:::e
        AL["alpha (logits)"]:::e
        AL2["alpha (normalised)"]:::e
        MS["msg"]:::e
    end
    subgraph DST2["destination domain (N_dst = 3)"]
        OUT["out"]:::d
    end

    Q -->|"select(0, dst)  — lift"| QI
    K -->|"select(0, src)  — lift"| KJ
    V -->|"select(0, src)  — lift"| VJ
    QI --> AL
    KJ --> AL
    AL -->|"segment_softmax(·, dst)"| AL2
    AL2 --> MS
    VJ --> MS
    MS -->|"select_assign(0, dst, Add)  — lower"| OUT

    classDef d fill:#1f4e79,stroke:#12314b,color:#fff
    classDef e fill:#7a4a00,stroke:#4a2c00,color:#fff
```

The design note's §2d diagram shows the same computation as a sequence of _operations_. This one
shows it as movement between _domains_ — three lifts in, one lower out. Every tensor in the middle
band is length 7.

### 6b. Step by step

**Gather.** `select(0, dst)` and `select(0, src)`. Note `q_i` repeats `q[0]` three times: rows 0-2
all come from destination 0.

| e | `src[e]` | `dst[e]` | `q_i = q[dst[e]]` | `k_j = k[src[e]]` | `v_j = v[src[e]]` |
| - | -------- | -------- | ----------------- | ----------------- | ----------------- |
| 0 | 0        | 0        | (1, 0)            | (1, 0)            | (10, 0)           |
| 1 | 1        | 0        | (1, 0)            | (0, 1)            | (0, 10)           |
| 2 | 3        | 0        | (1, 0)            | (2, 0)            | (1, 1)            |
| 3 | 2        | 1        | (0, 1)            | (1, 1)            | (5, 5)            |
| 4 | 0        | 2        | (1, 1)            | (1, 0)            | (10, 0)           |
| 5 | 2        | 2        | (1, 1)            | (1, 1)            | (5, 5)            |
| 6 | 3        | 2        | (1, 1)            | (2, 0)            | (1, 1)            |

Shapes: `[3,1,2]` and `[4,1,2]` both become `[7,1,2]`. Reads with duplicate indices are harmless.

**Logits.** `alpha = (q_i * k_j).sum_dim(-1) / sqrt(C)`, with `sqrt(2) = 1.4142`:

| e | `q_i · k_j` | `alpha` |
| - | ----------- | ------- |
| 0 | 1           | 0.7071  |
| 1 | 0           | 0.0000  |
| 2 | 2           | 1.4142  |
| 3 | 1           | 0.7071  |
| 4 | 1           | 0.7071  |
| 5 | 2           | 1.4142  |
| 6 | 2           | 1.4142  |

`[7, 1, 2] → [7, 1, 1]`. The channel axis is gone; one scalar per edge per head.

**Segment softmax.** Normalise **within each destination's run**, using `colptr = [0, 3, 4, 7]`:

```
segment for dst 0 = edges [0,3)   logits 0.7071, 0.0000, 1.4142
segment for dst 1 = edges [3,4)   logits 0.7071
segment for dst 2 = edges [4,7)   logits 0.7071, 1.4142, 1.4142
```

| e | dst | logit  | `exp(x − max_seg)` | `/ sum_seg` |
| - | --- | ------ | ------------------ | ----------- |
| 0 | 0   | 0.7071 | 0.4931             | **0.284**   |
| 1 | 0   | 0.0000 | 0.2431             | **0.140**   |
| 2 | 0   | 1.4142 | 1.0000             | **0.576**   |
|   |     |        | Σ = 1.7362         | Σ = 1.000   |
| 3 | 1   | 0.7071 | 1.0000             | **1.000**   |
|   |     |        | Σ = 1.0000         | Σ = 1.000   |
| 4 | 2   | 0.7071 | 0.4931             | **0.198**   |
| 5 | 2   | 1.4142 | 1.0000             | **0.401**   |
| 6 | 2   | 1.4142 | 1.0000             | **0.401**   |
|   |     |        | Σ = 2.4931         | Σ = 1.000   |

Three separate sums, each normalising to 1. Edge 3 gets weight exactly `1.000` because it is the
only edge into destination 1 — a one-element softmax is always 1.

**Weight and scatter.** `msg = v_j * alpha`, then add into the output row named by `dst`:

```
 msg[0] = (10, 0)  × 0.284 = ( 2.840, 0.000 )  ──┐
 msg[1] = ( 0,10)  × 0.140 = ( 0.000, 1.400 )  ──┼──> out[0] = ( 3.416, 1.976 )
 msg[2] = ( 1, 1)  × 0.576 = ( 0.576, 0.576 )  ──┘

 msg[3] = ( 5, 5)  × 1.000 = ( 5.000, 5.000 )  ─────> out[1] = ( 5.000, 5.000 )

 msg[4] = (10, 0)  × 0.198 = ( 1.980, 0.000 )  ──┐
 msg[5] = ( 5, 5)  × 0.401 = ( 2.005, 2.005 )  ──┼──> out[2] = ( 4.386, 2.406 )
 msg[6] = ( 1, 1)  × 0.401 = ( 0.401, 0.401 )  ──┘
```

`[7, 1, 2] → [3, 1, 2]` = `[N_dst, H, C]`. Because the weights in each segment sum to 1, every
output row is a **convex combination** of its incoming `v_j` rows — a weighted average of the
source values, never an extrapolation.

### 6c. The collision, made explicit

The last step is where duplicate indices stop being a curiosity:

```
                   dst = [0, 0, 0, 1, 2, 2, 2]
                          \  |  /
msg[0] ──┐                 \ | /
msg[1] ──┼──> += ──> out[ 0 ]     three writes, one address
msg[2] ──┘
```

Three edges name output row 0. At the toy scale that is three; at production scale it is **18.6 on
average**, because `CutOffEdges` puts ~18.6 data points inside each hidden node's ball (§2a).

The duplicates **are** the segments. They are not an artifact to be designed around:

- Without them, every segment would have size 1, every softmax weight would be `1.000`, and the
  layer would reduce to a per-edge copy — no attention at all.
- The decoder is the controlled comparison: `KNNEdges(k=3)` gives every destination exactly 3
  duplicates, so its segments are uniformly sized (§2b). Same mechanism, fixed group size.

If those three writes are issued by three threads with a non-atomic read-modify-write, two of them
are lost. That is the entire subject of §8-§9, and of §5 Step 4 of the design note.

---

## 7. The attention computation, step by step

§6 ran numbers through the layer. This section is about the tensor algebra: what each axis is for,
which one gets contracted, and why the one operation that looks like it should be a matrix product
is not one.

### 7a. The six steps at a glance

| # | Operation                               | In                   | Out             | Axis that changes         |
| - | --------------------------------------- | -------------------- | --------------- | ------------------------- |
| 1 | `query.select(0, dst)`                  | `[N_dst, H, C]`      | `[E, H, C]`     | node → edge               |
| 2 | `key.select(0, src) + edges`            | `[N_src, H, C]`      | `[E, H, C]`     | node → edge               |
| 3 | `value.select(0, src) + edges`          | `[N_src, H, C]`      | `[E, H, C]`     | node → edge               |
| 4 | `(q_i * k_j).sum_dim(-1) * norm`        | `[E, H, C]` ×2       | `[E, H, 1]`     | **`C` contracted away**   |
| 5 | `segment_softmax(alpha, dst)`           | `[E, H, 1]`          | `[E, H, 1]`     | none — values only        |
| 6 | `v_j * alpha`                           | `[E,H,C]`, `[E,H,1]` | `[E, H, C]`     | `C` restored by broadcast |
| 7 | `zeros.select_assign(0, dst, msg, Add)` | `[E, H, C]`          | `[N_dst, H, C]` | edge → node               |

Steps 1-3 and 7 move data between domains and are covered in §8. Steps 4-6 are the arithmetic.

### 7b. Steps 1-3 — what the gathers buy

The gathers do all the pairing work. After step 3, row `e` of `q_i`, `k_j` and `v_j` all describe
**the same edge** — `q_i[e]` is the query of that edge's destination, `k_j[e]` the key of its
source. Three tensors that started in two different domains are now aligned index-for-index.

That alignment is what makes step 4 trivial. There is no "which query goes with which key" left to
work out; it was decided by the edge list. Everything downstream is elementwise along `E`.

### 7c. Step 4 — an elementwise product, then a sum over `C`

```rust
let alpha = (q_i * k_j).sum_dim(-1) * norm;   // [E,H,C] -> [E,H,C] -> [E,H,1]
```

`q_i * k_j` is the **Hadamard (elementwise) product**, not a matrix product. Two `[E, H, C]`
tensors go in, one `[E, H, C]` tensor comes out, with `out[e,h,c] = q_i[e,h,c] * k_j[e,h,c]`. No
axis is contracted; nothing is summed yet.

The contraction is the separate `sum_dim(-1)`. Together the two operations are a **dot product**,
because that is all a dot product is:

```
a · b  =  sum over c of  a[c] * b[c]        multiply elementwise, then add up
```

Written as scalar loops, the whole of step 4 is:

```
for e in 0..E:                     // 748,348 edges   — independent
    for h in 0..H:                 //      16 heads   — independent
        s = 0.0
        for c in 0..C:             //      64 channels — SUMMED
            s += q_i[e,h,c] * k_j[e,h,c]
        alpha[e,h,0] = s * norm
```

Three axes, two distinct roles:

| Axis | Size    | Role           | Why                                                    |
| ---- | ------- | -------------- | ------------------------------------------------------ |
| `E`  | 748,348 | **batch**      | each edge's score is independent of every other edge's |
| `H`  | 16      | **batch**      | heads are separate subspaces and never interact        |
| `C`  | 64      | **contracted** | it is the feature dimension _inside_ one head          |

#### Why `C` is the one that gets summed

`q_i[e,h,:]` is a 64-number vector describing what the destination of edge `e` is looking for, in
head `h`. `k_j[e,h,:]` describes what the source offers, in the same head. The dot product measures
how well they line up — geometrically `q · k = |q| |k| cos θ`, large and positive when the two
vectors point the same way.

Summing over `C` collapses "how aligned are these two 64-dimensional vectors" into a single number.
That number is the attention logit.

It **has** to be a single number, for two reasons that both come from downstream:

- **Softmax needs a scalar per candidate.** Step 5 normalises the ~18.6 edges into one destination
  so their weights sum to 1. "Sum to 1" is only meaningful for scalars.
- **Step 6 weights the whole message.** `msg = v_j * alpha` scales all `C` channels of `v_j` by the
  same factor. If `alpha` kept its `C` axis you would have a per-channel gate, not attention — a
  different operator, and one for which "weights over a destination's edges sum to 1" has no
  meaning.

So the `C` axis serves its purpose during the comparison and is then deliberately destroyed. It
reappears in step 6 only because `v_j` still has it.

#### The paper writes `qᵀk`, and that is the same operation

UniMP's Eq. 3 (p. 4) does write a transpose, which reads like a matrix product:

```
                     ⟨ q_{c,i}, k_{c,j} + e_{c,ij} ⟩
   α_{c,ij}  =  ────────────────────────────────────────
                Σ_{u ∈ N(i)}  ⟨ q_{c,i}, k_{c,u} + e_{c,iu} ⟩

   where  ⟨q, k⟩ = exp( qᵀk / √d )
```

It is not a matrix product, and the paper says so two sentences later:

> For the `c`-th head attention, we firstly transform the source feature `h_i` and distant feature
> `h_j` into **query vector `q_{c,i} ∈ R^d`** and **key vector `k_{c,j} ∈ R^d`** respectively […]

`q` and `k` are **vectors in `R^d`**, not matrices. Under the usual convention that an unadorned
vector is a column, `qᵀ` is `1 × d`, `k` is `d × 1`, and `qᵀk` is `1 × 1` — a scalar. `qᵀk` is
simply how linear algebra spells a dot product; the transpose is shape bookkeeping, not an
instruction to call a matmul. The paper's own `⟨·,·⟩` — standard inner-product notation — says the
same thing.

Three tells that no matrix is involved, all visible in the equation itself:

| Tell                              | What it means                                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `q, k ∈ R^d`                      | operands are vectors; `qᵀk` is `1×1`                                                                                        |
| `α` is subscripted `c,ij`         | **one scalar per (head `c`, edge `(i,j)`)** — the paper indexes a specific pair, never a matrix of all pairs                |
| the denominator is `Σ_{u ∈ N(i)}` | normalisation runs over `i`'s **neighbours**, not over all nodes — this is §7i's segment softmax, written out in the source |

Compare with Vaswani et al., which writes `softmax(QKᵀ/√d)V` with **capital** `Q, K`. Those really
are matrices (`Q ∈ R^{n×d}`), `QKᵀ` really is a matmul, and it produces an `n × m` matrix of
scores — every query against every key. **The capitalisation is the tell.** UniMP uses lowercase
because it is describing one pair at a time.

So `qᵀk` and `(q_i * k_j).sum_dim(-1)` compute the identical quantity. The code differs from the
formula only in that it evaluates that formula `E · H` times at once: in code there is no single
`q`, there is `q_i` of shape `[E, H, C]` — a _batch_ of `E · H` vectors — and elementwise-multiply
then sum-the-last-axis is how you apply a vector formula across a batch.

The trap is treating `matmul` as "the operation that computes a dot product." It is not. It is the
operation that computes a **whole matrix of dot products**, every row of the left against every
column of the right. That is the right tool when you want all pairs, and the wrong one when the
pairs have already been chosen — §7d.

#### PyG does the same, for the same reason

`torch_geometric/nn/conv/transformer_conv.py:273`, character for character what anemoi has at
`conv.py:142`:

```python
alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
```

The `_i` and `_j` suffixes are the whole explanation. By the time `message()` runs, PyG's `_collect`
has already gathered along the edge list (§8a), so `query_i[e]` and `key_j[e]` are the query and
key **of the same edge**. The pairing is finished. An all-pairs operation has nothing left to
contribute, and `matmul` is an all-pairs operation.

Put differently: `MessagePassing` exists precisely to turn "attention over a graph" into "a batch
of `E` independent scalar problems." Having done that, the arithmetic that remains is elementwise.

### 7d. The `[E, H, H]` that never gets built

If step 4 were a matrix product, what would it produce? This is worth working out, because the
answer explains why it is not one.

Burn's `matmul` contracts the **last** axis of the left operand with the **second-to-last** of the
right, treating all leading axes as batch (`burn-tensor-0.21.0/src/tensor/api/numeric.rs:915`; the
inner-dimension check is `check.rs:534-566`). `q_i` and `k_j` are both `[E, H, C]`, so
`q_i.matmul(k_j)` does not even typecheck — `C ≠ H`. You would have to transpose:

```rust
q_i.matmul(k_j.swap_dims(1, 2))     // [E,H,C] @ [E,C,H]  ->  [E,H,H]
```

That **does** produce `E` matrices of shape `[H, H]` — the thing the shapes seem to invite. Its
entries are:

```
out[e, h1, h2] = sum over c of  q_i[e, h1, c] * k_j[e, h2, c]

                 = "head h1's query, dotted against head h2's key"
```

```
        for one edge e:              h2=0   h2=1   h2=2  ...  h2=15
                             h1=0  [  ✓      ✗      ✗    ...    ✗  ]
                             h1=1  [  ✗      ✓      ✗    ...    ✗  ]
                             h1=2  [  ✗      ✗      ✓    ...    ✗  ]
                              ...
                             h1=15 [  ✗      ✗      ✗    ...    ✓  ]

        ✓ = the diagonal, h1 == h2  -> exactly alpha[e, h]
        ✗ = a query from one head meeting a key from another
```

**The diagonal is precisely what we want, and everything else is meaningless.** Multi-head
attention keeps the heads in separate subspaces on purpose: head 0's query has no relationship to
head 7's key, and mixing them would defeat the point of having heads at all. `H` is a batch axis,
not something to contract over — which is also why no `[H, H]` object appears anywhere in
multi-head attention, dense or sparse.

Computing it and taking the diagonal would work, and would waste a factor of `H`:

|                                  | MACs                           | Materialised tensor (f32) |
| -------------------------------- | ------------------------------ | ------------------------- |
| `(q_i * k_j).sum_dim(-1)`        | `E·H·C` = **766,308,352**      | `[E, H, 1]` = **47.9 MB** |
| `q_i.matmul(k_jᵀ)` then diagonal | `E·H·H·C` = **12,260,933,632** | `[E, H, H]` = **0.77 GB** |

16× the arithmetic and 16× the memory to compute 15/16 garbage. The elementwise-then-sum form
computes the diagonal directly.

#### Where a matmul _is_ the right tool

Dense self-attention. There you have `Q` of shape `[H, N, C]` and `K` of shape `[H, M, C]`, and you
genuinely want **every** query against **every** key: `Q @ Kᵀ` → `[H, N, M]`, a real matrix product
with `C` contracted and `H` batched. That is why anemoi's self-attention path uses head-major
layout while the graph conv does not (§5b).

The difference is which pairs you want:

|                                  | Pairs scored                | Score tensor                               |
| -------------------------------- | --------------------------- | ------------------------------------------ |
| Dense attention over these grids | all `N_dst × N_src`         | `[H, 40320, 542080]` ≈ 350 billion entries |
| This graph conv                  | only the `E` that are edges | `[E, H, 1]` = 12 million entries           |

The gathers in steps 1-3 already picked out the pairs. Once the pairing is decided, no
all-against-all operation remains — only `E · H` independent dot products, which is exactly what
elementwise-multiply-then-sum expresses.

### 7e. Why divide by `√C`

If `q` and `k` had independent components with mean 0 and variance 1, then `q · k` — a sum of `C`
such products — would have variance `C`, so a standard deviation of `√64 = 8`. Logits would grow
with the head width, pushing softmax into its saturated region where one weight is ~1, the rest ~0,
and the gradient vanishes. Dividing by `√C` restores unit scale. This is Vaswani et al. §3.2.1
([1706.03762](https://arxiv.org/abs/1706.03762)); anemoi spells it `/ self.out_channels**0.5`
(`conv.py:142`).

The port precomputes the reciprocal and multiplies (`src/common.rs:131`): `1/√C` is one rounding of
a constant, versus `E · H` divisions each carrying their own error.

### 7f. Broadcasting: how `[E, H, 1]` meets `[E, H, C]`

Step 6 multiplies tensors of different shapes. Burn's rules are stricter than NumPy's, and they are
enforced at two different times:

**Rank must match, and it is a compile-time constraint.** `Tensor<B, D, K>` carries `D` as a const
generic, and the elementwise check takes both operands at the _same_ `D`
(`burn-tensor-0.21.0/src/tensor/api/check.rs:43-51`). Burn will not left-pad a rank-2 tensor with a
leading `1` the way NumPy does; a rank mismatch fails to compile.

**Sizes broadcast equal-or-1, checked at runtime.** `binary_ops_ew_shape` walks every axis and
accepts a pair only if the sizes are equal or one of them is `1` (`check.rs:1329-1339`).

Applied to step 6:

```
   v_j    [E, H, C]        rank 3
   alpha  [E, H, 1]        rank 3          ✓ ranks match — compiles
                ^
   axis 0:  E vs E   equal      ✓
   axis 1:  H vs H   equal      ✓
   axis 2:  1 vs C   one is 1   ✓ broadcast — the single value is reused C times

   result [E, H, C]
```

This is why §7g's rank detail matters in practice: `[E, H]` would be rank 2 and **would not
compile** against `[E, H, C]`. Keeping the reduced axis is not cosmetic.

#### What the type actually guarantees: rank, and nothing else

Burn 0.21 has no shape-typed tensors. `Tensor<B, 3>` carries `D = 3` as a const generic and that is
the entire contract — `[E, H, 1]`, `[E, H, C]`, `[E, 1, H]` and `[N_dst, H, 1]` are all **the same
type**. Extents are runtime data, so every `[E, …]` in this document is a convention held up by the
code that produced the tensor, not a fact the compiler knows.

Nor do the runtime checks close the gap evenly. They are per-operation and narrow:

| Operation                   | What it verifies                                                    | What it lets through                                                            |
| --------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| elementwise (`*`, `-`, `/`) | every axis equal-or-1 (`check.rs:1329-1339`)                        | a size-1 axis silently broadcasting where a real size was meant                 |
| `select_assign`             | axis in range; `values.shape[dim] == indices.shape[0]` (`check.rs`) | **any mismatch between `values` and the destination on the other axes**         |
| `matmul`                    | `lhs.shape[D-1] == rhs.shape[D-2]` (`check.rs:534-566`)             | nothing much — this one is tight                                                |
| `select`                    | axis in range                                                       | out-of-range index _values_, on backends without bounds checks (`base.rs:1764`) |

The `select_assign` row is the sharp edge: the axes carrying `H` and `C` are never compared against
the destination buffer, so a tensor with the right dim 0 and the wrong trailing axes reaches the
kernel and is indexed with mismatched strides. Nothing panics.

The practical consequence is that shape correctness in this layer is an **assertion discipline**,
not a type-system property, and it is cheapest to enforce where the invariant is born rather than
at every use — which is why `EdgeIndex` gets a checked constructor. §5 Steps 3-4 of the design note
carry the specific assertions.

### 7g. Step 5 — what `alpha` is

Following `conv.py:139-147` against the Rust in `src/common.rs:138-148`:

| After                         | anemoi                | Rust (Burn)       | Meaning                                |
| ----------------------------- | --------------------- | ----------------- | -------------------------------------- |
| gather                        | `query_i` `[E, H, C]` | `q_i` `[E, H, C]` | per edge, per head, a `C`-vector       |
| `sum(dim=-1)` / `sum_dim(-1)` | `[E, H]`              | `[E, H, 1]`       | per edge, per head, **one scalar**     |
| `softmax(…)`                  | `[E, H]`              | `[E, H, 1]`       | same, now normalised within segments   |
| `.view(-1, heads, 1)`         | `[E, H, 1]`           | — not needed      | ready to broadcast against `[E, H, C]` |

The rank difference is a real API difference, not a bug: **PyTorch's `sum` drops the reduced axis;
Burn's `sum_dim` keeps it.** So anemoi has to put the axis back by hand at `conv.py:147` before it
can broadcast, exactly as §7f requires; the port never loses it. Verified: `sum_dim` returns `Self`,
preserving the const-generic rank (`burn-tensor-0.21.0/src/tensor/api/numeric.rs:451`).

So `alpha` is **one number per (edge, head)**. Not per channel — the channel axis was contracted
away by the dot product. It answers: _how much should this edge's message count, for this head?_

### 7h. Step 5 — three softmaxes on the same numbers

Take the seven logits from §6b and normalise them three ways.

**(i) Over the channel axis** — `activation::softmax(alpha, 2)`. `alpha` is `[7, 1, 1]`; the last
axis has extent 1. Softmax of a one-element vector is `1.0`. Every entry becomes `1.000`. A no-op.
The channel information is already gone; there is nothing to normalise.

**(ii) Over all edges** — `activation::softmax(alpha, 0)`. One denominator for all seven:

| e | dst   | segment softmax | global softmax |
| - | ----- | --------------- | -------------- |
| 0 | 0     | 0.284           | 0.104          |
| 1 | 0     | 0.140           | 0.051          |
| 2 | 0     | 0.576           | 0.212          |
| 3 | **1** | **1.000**       | **0.104**      |
| 4 | 2     | 0.198           | 0.104          |
| 5 | 2     | 0.401           | 0.212          |
| 6 | 2     | 0.401           | 0.212          |

Look at edge 3. It is the sole edge into destination 1, so its message must pass through
undiminished — weight `1.000`. The global softmax gives it `0.104`, scaling destination 1's output
down by ~10× for no reason other than that six unrelated edges exist elsewhere in the graph. The
global version also makes every output depend on every other destination's logits, which destroys
locality: adding one edge anywhere changes every weight.

**(iii) Segment softmax** — the correct one. Column 3 above. Each destination's weights sum to
`1.000` independently.

### 7i. Step 5 — how it is actually computed: down, then back up

Segment softmax is a **scatter followed by a gather**. Every segmented reduction has this shape:
reduce into the segment domain, then broadcast the per-segment result back out to its members.

```
num      [E, H, 1]            edge domain        num = exp(alpha - m)
   │
   │  select_assign(0, dst, Add)         lower:  E -> N_dst   (sum within each segment)
   ▼
denom    [N_dst, H, 1]        destination domain
   │
   │  select(0, dst)                     lift:   N_dst -> E   (copy back to each member)
   ▼
denom_e  [E, H, 1]            edge domain
   │
   ▼
num / denom_e  ->  [E, H, 1]
```

The gather back up is not optional bookkeeping — it is forced twice over:

- **Softmax's output is per-edge.** Edge `e` needs _its own_ destination's denominator, i.e.
  `denom[dst[e]]`. That is a lookup by `dst`, the same operation as `query.select(0, dst)` in
  step 1.
- **The shapes do not otherwise meet.** `[E, H, 1] / [N_dst, H, 1]` is `748348` against `40320` on
  axis 0 with neither equal to 1, which the equal-or-1 rule rejects (§7f).

On the §6 toy, `denom = [1.7362, 1.0000, 2.4931]` — three rows — gathered by
`dst = [0,0,0,1,2,2,2]` becomes seven:

```
[1.7362, 1.7362, 1.7362, 1.0000, 2.4931, 2.4931, 2.4931]
```

and `0.4931 / 1.7362 = 0.284`, matching §6b. Note the numerator is **not** gathered: it is already
edge-indexed and correctly aligned. Only `denom` is in the wrong domain.

PyG writes the identical pair, `_softmax.py:87-88`:

```python
out_sum = scatter(out, index, dim, dim_size=N, reduce="sum") + 1e-16
out_sum = out_sum.index_select(dim, index)
```

`index_select(dim, index)` _is_ `select(0, dst)`, and line 85 does the same round trip a second time
for the max. Its segment-pointer branch (`_softmax.py:69-80`) performs the same broadcast with
`repeat_interleave` over run lengths instead of an index gather — cheaper, because with `colptr` you
already know each segment's length. That is one of the things a `colptr`-driven CubeCL kernel gets
for free (§4b).

#### Why subtract a max at all

Softmax is shift-invariant — for any constant `c`, `exp(xᵢ−c) / Σⱼ exp(xⱼ−c)` equals
`exp(xᵢ) / Σⱼ exp(xⱼ)`, because the `exp(−c)` factors cancel. So `m` is chosen purely for floating
point.

In f32, `exp` overflows to `inf` above **88.72** and flushes to zero below **−103.28**. Without the
shift, logits of `[90, 89, 88]` give `[inf, inf, 1.65e38]` and then `[nan, nan, 0.0]`; with it,
`[1.0, 0.3679, 0.1353]` → `[0.665, 0.245, 0.090]`. Not a precision loss — a destroyed tensor, and
`NaN` propagates.

The **max** is the unique constant that puts every argument in `(−∞, 0]`, so `exp(x−m) ∈ (0, 1]` and
overflow becomes structurally unreachable rather than merely unlikely. Anything smaller leaves
overflow headroom; anything larger drags the group toward underflow. It pairs with the `1/√C` of
§7e, which fixes the logits' _scale_ where this fixes their _absolute position_.

Using a **global** max rather than a per-segment one keeps one guarantee and loses the other:

|                     | Per-segment max (PyG)                   | Global max (this port)                           |
| ------------------- | --------------------------------------- | ------------------------------------------------ |
| Overflow impossible | yes                                     | yes — every `x − m ≤ 0` regardless               |
| Denominator `≥ 1`   | yes — each segment contains its own max | **no** — only the segment holding the global max |

A segment whose logits all sit more than 103.3 below the global max would flush to zero entirely and
divide `0/0`. With post-LayerNorm logits of roughly unit variance (§7e) that spread is far outside
anything realistic, but it is where PyG's `+ 1e-16` earns its place in this variant: it turns a
remote `NaN` into a `0`, which is the right answer for a segment that has genuinely vanished.

### 7j. Step 5 — why no `dim` argument can express it

`activation::softmax(tensor, dim)` normalises over **the full extent of one axis**. Segment
boundaries here are `colptr[d]..colptr[d+1]` — **variable-length runs inside axis 0**:

```
axis 0, length 7:

  [ e0  e1  e2 | e3 | e4  e5  e6 ]
    \________/   \/   \________/
      len 3     len 1    len 3      <- not an axis; extents differ per group
```

An axis has one extent. These runs have three. No `dim` names them, which is precisely why PyG
ships `torch_geometric.utils.softmax(src, index, ptr, num_nodes)` as a separate function rather
than reusing `torch.softmax`.

**The one escape.** If every destination had the _same_ degree `D`, the runs would be an axis: view
`[E, H, 1]` as `[N_dst, D, H, 1]` and call the builtin on axis 1. The decoder's `KNNEdges(k=3)`
graph satisfies this exactly. The encoder's `CutOffEdges` graph does not — and its maximum degree
is unknown (§2b), so the padded-dense variant of this trick cannot even be costed yet. That is
follow-up 2 in the design note.

### 7k. Steps 6-7 — weight, then aggregate

```rust
let msg = v_j * alpha;    // [E,H,C] * [E,H,1] -> [E,H,C]   (§7f)
```

One scalar scales all `C` channels of an edge's message, per head. Nothing is contracted; the
shapes come back to `[E, H, C]` purely by broadcast.

Step 7 sums the messages into their destinations — the scatter, §8. Because §7h's weights sum to
`1.000` within each destination, the result is a **convex combination** of that destination's
incoming `v_j` rows: a weighted average, never an extrapolation beyond the range of its inputs. A
softmax that normalised over the wrong axis would silently break that property, which is another
way of saying why §7j matters.

Two things worth noticing about the shape trajectory `[E,H,C] → [E,H,1] → [E,H,C]`: the `C` axis is
destroyed and restored, and the `H` axis is never touched by anything. Heads enter as 16
independent copies of the same computation and leave the same way — the only place they meet is the
final reshape back to `[N_dst, 1024]` outside this function (§5a).

---

## 8. How scatter works

Everything in §6 is **three gathers and one scatter**. Gather and scatter are inverses of each
other, and the difference between them is the source of every difficulty in this port — so it is
worth being precise about what each one does.

### 8a. Gather reads at indices

A gather takes a list of positions and produces the values found there. `Tensor::select(dim,
indices)`, `burn-tensor-0.21.0/src/tensor/api/base.rs:1641`, with a **1-D** index list:

```
output[i, j, k] = input[indices[i], j, k]        // dim = 0
```

Using `src = [0, 1, 3, 2, 0, 2, 3]` from §6 to gather rows of `value`:

```
value  [N_src = 4]         v_j  [E = 7]   =   value.select(0, src)
------------------         -------------------------------------------
 row 0 : (10,  0)  <----    e=0   src[0]=0  ->  (10,  0)
 row 1 : ( 0, 10)  <----    e=1   src[1]=1  ->  ( 0, 10)
 row 2 : ( 5,  5)           e=2   src[2]=3  ->  ( 1,  1)
 row 3 : ( 1,  1)           e=3   src[3]=2  ->  ( 5,  5)
                            e=4   src[4]=0  ->  (10,  0)   <- row 0, read again
                            e=5   src[5]=2  ->  ( 5,  5)   <- row 2, read again
                            e=6   src[6]=3  ->  ( 1,  1)   <- row 3, read again
```

Two properties make gather easy:

- **Every output cell is written exactly once.** The output has one row per index, in order.
- **Duplicate indices are free.** `src[0]` and `src[4]` are both `0`; row 0 of `value` is simply
  read twice. Concurrent reads of the same address never conflict.

**The output length is the _index_ length, not the source length.** `output.shape[0] ==
indices.len()`; the source's dim 0 never appears in the output shape — it only bounds what the
index values may legally be. So a gather can shrink a tensor, keep it the same, or **grow** it:

```
  value                   [      4, H, C]      4 rows
  src                     [      7]            7 indices
  value.select(0, src)  = [      7, H, C]      7 rows      <- grew, because src repeats

  denom                   [ 40320, H, 1]       40,320 rows
  dst                     [748348]             748,348 indices
  denom.select(0, dst)  = [748348, H, 1]       748,348 rows   <- each row copied ~18.6x
```

`select` is a **lookup, not a filter** — `[source[i] for i in indices]`. Reading it as "pick out a
subset" is what makes the growing case surprising.

Shape goes `[N_src, …] → [E, …]`. It moves data _into_ the edge domain.

### 8b. Scatter writes at indices — and loses both properties

A scatter is the same index list used in the opposite direction: for each element of `values`,
write it to the position named by `indices`. `Tensor::select_assign(dim, indices, values, update)`,
`base.rs:1673`:

```
input[indices[i], j, k] += values[i, j, k]       // dim = 0
```

Same `dst = [0, 0, 0, 1, 2, 2, 2]`, running the other way:

```
  msg  [E = 7]                          out  [N_dst = 3]   (initialised to zeros)
  -------------------------------       -----------------------------------------
   e=0  (2.840, 0.000)  dst[0]=0  --+
   e=1  (0.000, 1.400)  dst[1]=0  --+-->  out[0] = (3.416, 1.976)    3 writes
   e=2  (0.576, 0.576)  dst[2]=0  --+

   e=3  (5.000, 5.000)  dst[3]=1  ---->  out[1] = (5.000, 5.000)     1 write

   e=4  (1.980, 0.000)  dst[4]=2  --+
   e=5  (2.005, 2.005)  dst[5]=2  --+-->  out[2] = (4.386, 2.406)    3 writes
   e=6  (0.401, 0.401)  dst[6]=2  --+
```

Neither property survives:

| How many times a cell is written                     | Consequence                                                                           |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **0** — no edge names it (a zero-degree destination) | the output needs a defined starting value; scatter cannot invent one                  |
| **1**                                                | the easy case                                                                         |
| **many** — 18.6 on average here                      | the writes must be _combined_, and the combining rule has to be part of the operation |

That is why the API is `Tensor::zeros(…).select_assign(…, Add)` rather than a bare scatter. The
`zeros` supplies the answer for degree 0; `Add` supplies it for degree > 1.

### 8c. Why the update op cannot be `Assign`

The combining rule is not a stylistic choice. For a scatter to be **well defined** it must not
depend on the order the writes happen to arrive, and for cells nobody writes it must still produce
something. Formally the rule has to be a commutative monoid — an associative, commutative operator
with an identity element:

| Op       | Identity | Associative & commutative? | Well defined on duplicates?                        |
| -------- | -------- | -------------------------- | -------------------------------------------------- |
| `Add`    | `0`      | yes                        | **yes**                                            |
| `Mul`    | `1`      | yes                        | yes                                                |
| `Min`    | `+inf`   | yes                        | yes                                                |
| `Max`    | `−inf`   | yes                        | yes                                                |
| `Assign` | — none   | **no**                     | **no** — the result is whichever write landed last |

`Assign` with duplicate indices has no defined answer at all: three edges naming `out[0]` would
each claim it, and nothing in the problem statement says which wins. It is only meaningful when the
indices are unique.

This is also why aggregation in message passing is _always_ sum, mean, min or max, never
assignment — anemoi's `aggr="add"` (`conv.py:97`) is picking an element from that table. §7's
softmax is what makes `add` the right pick: the weights already sum to 1, so summation produces a
weighted average.

### 8d. Burn 0.21 has three gather/scatter pairs, differing only in how indices address

| Gather                | Scatter                   | Addressing                                           | Index shape                       | Index shape _for our case_ |
| --------------------- | ------------------------- | ---------------------------------------------------- | --------------------------------- | -------------------------- |
| `select` (`:1641`)    | `select_assign` (`:1673`) | **axis** — one index per slice along `dim`           | `[n]`, always 1-D                 | `[E]` → **6 MB**           |
| `gather` (`:1766`)    | `scatter` (`:1804`)       | **element** — one index per element                  | same rank _and_ shape as `values` | `[E, H, C]` → **6.1 GB**   |
| `gather_nd` (`:1883`) | `scatter_nd` (`:1853`)    | **slice** — `K` indices address the leading `K` dims | `[…, K]`                          | `[E, 1]` → **6 MB**        |

All three express our aggregation. They differ in what they cost and what they permit:

- **`select_assign`** — what the port uses. `indices` is `dst` itself, `[E]`, and `values` is the
  full-rank `[E, H, C]` message tensor. The 1-D index is the cheap part.
- **`scatter`** — element-addressed, so scattering `[E, H, C]` messages requires an `[E, H, C]`
  i64 index tensor: the same index repeated `H·C = 1024` times per edge, **≈ 6.1 GB**. Ruled out on
  memory alone.
- **`scatter_nd`** — `K = 1` indexes only dim 0, so `[E, 1]` suffices and `values` keeps the
  trailing dims. Cheap, and the only one of the three that accepts an update op other than `Add`
  (`scatter` panics otherwise, per its own docstring at `base.rs:1802-1803`). It is nonetheless the
  wrong choice here, for a reason that has nothing to do with shapes — §9.

### 8e. The call this port makes

```rust
Tensor::zeros([n_dst, h, c], dev)
    .select_assign(0, dst, msg, IndexingUpdateOp::Add)
```

Read against Burn's own specification at `base.rs:1664`:

```
input[indices[i], j, k] += values[i, j, k]        // dim = 0

  input   = zeros [N_dst, H, C]      the accumulator
  indices = dst   [E]                indices[e] is the destination of edge e
  values  = msg   [E, H, C]          values[e] is the message on edge e
  ⇒  out[dst[e], h, c] += msg[e, h, c]     for every e, h, c
```

`i` ranges over `E`, and `dst[e]` collides for ~18.6 values of `e`. That collision is not
incidental — it is what performs the sum over `j ∈ N(i)` in §1a's equation.

### 8f. Burn documents the duplicate case, and it is not reassuring

From `scatter_nd`'s docstring, `base.rs:1840-1847`, verbatim:

> When `indices` contains duplicate entries, behavior varies by operation:
>
> - For `Add`, accumulation is supported, though results may be non-deterministic on GPU backends.
> - For other operations (`Assign`, `Mul`, `Min`, `Max`), duplicate indices result in undefined
>   behavior for both the forward result and the backward gradients.
>
> For deterministic results and correct gradient calculation across all operations, `indices` should
> contain unique entries.

Duplicate indices are our _entire workload_ — the mean destination has 18.6 of them (§6c) — so
"indices should contain unique entries" is advice this layer cannot take. The upstream warning is
therefore load-bearing, and §9 works out what it actually means at the kernel level.

---

## 9. Does PyTorch race?

PyG computes the per-segment maximum with `scatter`, `_softmax.py:84`:

```python
src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
out = src - src_max.index_select(dim, index)
out = out.exp()
out_sum = scatter(out, index, dim, dim_size=N, reduce="sum") + 1e-16
```

`index` is `dst`, with the same ~18.6-way duplicates. So the question is fair: if cubecl's
`scatter_nd` races on exactly this pattern, why doesn't PyTorch's?

**It has the same hazard and resolves it with atomics.** The trace:

1. `_softmax.py:84` calls `torch_geometric.utils.scatter`.
2. `_scatter.py:84-114` routes `reduce in ['min','max','amin','amax']` either to
   `Tensor.scatter_reduce_(dim, index, src, reduce='amax', include_self=False)` or to
   `torch_scatter.scatter`, depending on build and device.
3. The CUDA implementation is `aten/src/ATen/native/cuda/ScatterGatherKernel.cu`. Its reduction
   functors are one line each:

   ```cpp
   class ReduceMaximum {
     ...
     constexpr C10_DEVICE void operator() (scalar_t* self_data_start, int64_t index,
                                           int64_t numel, const scalar_t * src_data) const {
       gpuAtomicMax(self_data_start + index, *src_data);      // line 64
     }
   };
   ```

   `gpuAtomicMax` at `:64`, `gpuAtomicMin` at `:54`, `fastAtomicAdd` at `:35` and `:44`,
   `gpuAtomicMul` at `:26`. Every scatter reduction PyTorch offers is atomic.

That is the missing half of the design note's argument. The hazard is identical; the resolution is
not.

### 9a. Correct versus reproducible

Two different properties, and they come apart:

| Implementation                      | Correct? | Bit-reproducible? | Why                                                                                                                                                       |
| ----------------------------------- | -------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| torch `scatter_add_` (CUDA)         | yes      | **no**            | `fastAtomicAdd` serialises the updates, so nothing is lost — but float addition is not associative, so a different arrival order gives different rounding |
| torch `scatter_reduce_` amax (CUDA) | yes      | yes               | `gpuAtomicMax`. Max is commutative, associative **and idempotent**, so order cannot change the result at all                                              |
| cubecl `scatter_nd`, any op         | **no**   | no                | plain non-atomic load/store, `burn-cubecl-0.21.0/src/kernel/index/scatter_nd.rs:69-73`                                                                    |
| cubecl `select_assign`, `Add`       | yes      | yes               | one thread per output column, serial loop over the scattered axis, `select_assign.rs:49`                                                                  |

The cubecl `scatter_nd` kernel says what it does in its own doc comment (`scatter_nd.rs:16-17`):
_"Each thread handles one element across all update slices. Work items = num_updates \*
slice_size."_ With one thread per element and

```rust
let result = Op::BinaryOp::<T, Const<1>>::execute(
    Vector::cast_from(data[data_idx]),
    Vector::cast_from(values[val_offset]),
);
data[data_idx] = result[0];
```

there is nothing to serialise the eighteen threads that share a `data_idx`. `select_assign`
partitions the other way — `working_units = value.num_elements() / value.shape(axis)` — and then
loops `for i in 0..value.shape(axis)` **inside one thread**, so duplicates arrive in sequence.

The fourth row is worth stating plainly: **Burn's `select_assign` is _more_ reproducible than
PyTorch's `scatter_add_`**, and pays for it in parallelism — 1024 threads each walking all 748,348
edges. That is the exact trade §5 Step 4 of the design note makes.

### 9b. What this means for the port

- **Duplicates are not the problem. Non-atomicity is.** PyG does not avoid duplicate indices; it
  relies on a kernel that handles them.
- **This is why PyG can afford a true per-segment max and the port cannot.** `_softmax.py:84`
  works because `gpuAtomicMax` exists. Burn 0.21's `IndexingUpdateOp::Max` exists too, but only on
  `scatter_nd`, the primitive that races. The duplicate-safe primitives (`select_assign`,
  `scatter`) are `Add`-only, and that restriction sits in the kind layer
  (`burn-backend/src/tensor/ops/float.rs:114`) above the `Backend` trait, so no backend swap lifts
  it. Hence the global-max substitution in §5 Step 4 — mathematically exact, just not fused.
- **The reference fixture is a correctness oracle only.** PyG reaches `softmax` with `ptr = None`
  (§3b note 4 of the design note) and so takes this scatter path rather than the segment-pointer
  path. It is not a performance baseline.
- **A CPU-only test would hide all of this.** `burn-ndarray` accumulates correctly under every
  primitive — one sequential host loop, `burn-ndarray-0.21.0/src/ops/base.rs:236` — so a
  `scatter_nd`-based implementation passes on CPU and is silently wrong on `wgpu`. That is why §6
  of the design note tests on `wgpu`.

---

## 10. Glossary

| Symbol   | Domain                         | Meaning                                                                                  |
| -------- | ------------------------------ | ---------------------------------------------------------------------------------------- |
| `i`      | destination                    | The node receiving. PyG suffix `_i`; anemoi's `query_i`.                                 |
| `j`      | source                         | The node sending. PyG suffix `_j`; anemoi's `key_j`, `value_j`.                          |
| `N_src`  | —                              | Source node count. 542,080 (N320 grid).                                                  |
| `N_dst`  | —                              | Destination node count. 40,320 (O96 grid).                                               |
| `E`      | —                              | Edge count. 748,348 encoder, 1,626,240 decoder.                                          |
| `H`      | —                              | Attention heads. 16. Independent throughout.                                             |
| `C`      | —                              | Channels per head, `out_channels_conv`. 64. `H·C = 1024 = hidden_dim`.                   |
| `src`    | edge (length `E`)              | `src[e]` = source node of edge `e`, in `[0, N_src)`. CSC "row index".                    |
| `dst`    | edge (length `E`)              | `dst[e]` = destination node of edge `e`, in `[0, N_dst)`.                                |
| `colptr` | destination (length `N_dst+1`) | `colptr[d]..colptr[d+1]` = edge range owned by `d`. `colptr[N_dst] == E`.                |
| `alpha`  | edge                           | `[E, H, 1]`. One attention weight per edge per head, post-softmax.                       |
| segment  | —                              | The set of edges sharing a destination. Variable-length here; fixed at 3 in the decoder. |
| COO      | —                              | `(src, dst)` — two parallel per-edge arrays.                                             |
| CSC      | —                              | `(colptr, src)` — compressed destinations. Requires destination-sorted edges.            |

## Where to read next

[`graph-transformer-forward-mapper.md`](./graph-transformer-forward-mapper.md):

- **§2** — how PyG's 1035-line `MessagePassing` reduces to three gathers and one scatter, branch by
  branch.
- **§3** — the two anemoi reference implementations (PyG and Triton) and what each reveals.
- **§4** — checkpoint tensor inventory and the parameter shapes.
- **§5 Step 3-4** — `EdgeIndex` and `graph_transformer_conv`, with the full `scatter_nd` /
  `select_assign` / `segment_softmax` argument this document's §8-§9 supply the mechanics and
  the PyTorch half of.
- **§8** — the CubeCL segmented kernel follow-up, which removes both the global-max compromise and
  the 1024-thread bottleneck.

> **Current state of the code.** `src/common.rs:117-155` is mid-edit and does not compile: line 149
> calls `alpha.scatter_nd(indices, values, update)` with three undefined bindings and discards the
> result, and line 154 returns `alpha` at `[E, H, 1]` rather than the declared `[N_dst, H, C]`. The
> gather and logit steps (`:138-148`) match §6b above. Nothing from the design note's six
> implementation steps has landed yet.
