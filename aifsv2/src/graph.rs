use burn::prelude::*;

// This structure stores the bipartite edge list for one sub-graph.
// This connects source nodes to dest nodes.
// You can assume sorted by dest.
//
// attr carry the edge attributes we care about.
#[derive(Clone, Debug)]
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>,    // Shape [E]
    pub dst: Tensor<B, 1, Int>,    // [E]
    pub colptr: Tensor<B, 1, Int>, // [N_ds + 1]

    pub num_src: usize,
    pub num_dst: usize,
}

// TODO(putravu): functions related to this impl

// Cat combines the src, dst, col_ptr of all the indices
pub fn cat<B: Backend>(edge_indices: Vec<EdgeIndex<B>>) -> EdgeIndex<B> {
    let mut srcs = Vec::default();
    let mut dsts = Vec::default();
    let mut colptrs = Vec::default();
    let mut num_srcs = 0 as usize;
    let mut num_dsts = 0 as usize;

    let mut colptrs_max = Tensor::from_ints([0], &edge_indices[0].src.device());
    for e in edge_indices.iter() {
        srcs.push(e.src.clone());
        dsts.push(e.dst.clone());

        // Add colptrs the max of the previous. Track the max.
        // TODO(saiputravu): Is this correct + there has to be a nicer way of
        // doing this + do we even need colptrs, its unused.
        colptrs.push(e.colptr.clone() + colptrs_max.clone());
        colptrs_max = colptrs_max + e.colptr.clone().max_dim(0);

        num_srcs += e.num_src;
        num_dsts += e.num_dst;
    }

    EdgeIndex {
        src: Tensor::cat(srcs, 0),
        dst: Tensor::cat(dsts, 0),
        colptr: Tensor::cat(colptrs, 0),
        num_src: num_srcs,
        num_dst: num_dsts,
    }
}

impl<B: Backend> EdgeIndex<B> {
    pub fn add(&self, rhs: Tensor<B, 2, Int>) -> EdgeIndex<B> {
        let [a, b] = rhs.shape().dims();
        let [e] = self.src.shape().dims();
        assert_eq!(a, 2, "found dim 0 in EdgeIndex::add {}, expected {}", a, 2);
        assert!(
            b == 1 || b == e,
            "found dim 1 in EdgeIndex::add {}, expected {} or {}",
            b,
            1,
            e
        );

        // TODO(saiputravu): Clean this code up.
        //
        // Also think about what to do with colptr. I think colptr doesn't change here, as the
        // number of values is not changing?
        let mut lhs = self.clone();
        let other = rhs.flatten(0, 1);
        let top = other.clone().slice(0..b);
        let bot = other.slice(b..);

        lhs.src = lhs.src + top;
        lhs.dst = lhs.dst + bot;
        lhs
    }
}

// From a starting edge_index, we increment known edges by some edge_inc. We expand
// each edge batch_size number of times.
//
// This is so that each node in each sub-graph has unique identifiers.
pub fn expand_edges<B: Backend>(
    edge_idx: EdgeIndex<B>,
    edge_inc: Tensor<B, 2, Int>,
    batch_size: usize,
) -> EdgeIndex<B> {
    let mut edge_indices = Vec::default();
    for i in 0..batch_size {
        // For each batch, increment the node identifiers.
        // new batch <- (edge_idx + i*inc)
        edge_indices.push(edge_idx.add(edge_inc.clone().mul_scalar(i as i32)));
    }
    cat(edge_indices)
}
