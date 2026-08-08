use burn::prelude::*;

// This structure stores the bipartite edge list for one sub-graph.
// This connects source nodes to dest nodes.
// You can assume sorted by dest.
#[derive(Clone)]
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>, // Shape [E]
    pub dst: Tensor<B, 1, Int>, // [E]
    pub colptr: Vec<i64>,       // [N_ds+ 1]

    pub num_src: usize,
    pub num_dst: usize,
}

// TODO(putravu): functions related to this impl.
