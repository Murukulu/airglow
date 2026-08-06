use burn::prelude::*;

// Implementing CSC Sparse Graph.
// This structure stores the bipartite edge list for one sub-graph.
// You can assume sorted by dest
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>, // Shape [E]
    pub dst: Tensor<B, 1, Int>, // [E]
    pub colptr: Vec<i64>,       // [N_ds+ 1]

    pub num_src: usize,
    pub num_dst: usize,
}
