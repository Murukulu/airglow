use burn::prelude::*;
use burn_store::{ModuleStore, SafetensorsStore};

// The geometric half of the graph, as extracted from the .ckpt into safetensors.
//
// The 8 trainable columns per node and per edge are Parameters and live in the weights file
// instead; edge attributes here are already unit-std normalised.
#[derive(Debug)]
pub struct GraphData<B: Backend> {
    pub data_x: Tensor<B, 2>,   // [N_data, 2]; [lat, lon] radians, lon in 0..2pi
    pub hidden_x: Tensor<B, 2>, // [N_hidden, 2]; same encoding

    pub data_to_hidden_edge_index: Tensor<B, 2, Int>, // [2, E_enc]; row 0 src (data), row 1 dst (hidden)
    pub data_to_hidden_edge_direction: Tensor<B, 2>,  // [E_enc, 2]; per edge direction feature.
    pub data_to_hidden_edge_length: Tensor<B, 2>,     // [E_enc, 1]; per edge length feature.

    pub hidden_to_data_edge_index: Tensor<B, 2, Int>, // [2, E_dec]; row 0 src (hidden), row 1 dst (data)
    pub hidden_to_data_edge_direction: Tensor<B, 2>,  // [E_dec, 2]; per edge direction feature.
    pub hidden_to_data_edge_length: Tensor<B, 2>,     // [E_dec, 1]; per edge length feature.

    pub data_area_weight: Tensor<B, 2>, // [N_data, 1]; training-loss weight, unused at inference. Ignore.
}

// Snapshots are lazy — to_data is where the bytes are actually read.
fn snapshot(store: &mut SafetensorsStore, name: &str) -> Result<TensorData, String> {
    store
        .get_snapshot(name)
        .map_err(|e| format!("reading {name}: {e}"))?
        .ok_or_else(|| format!("missing {name}"))?
        .to_data()
        .map_err(|e| format!("reading {name}: {e}"))
}

impl<B: Backend> GraphData<B> {
    pub fn from_safetensors_store(
        store: &mut SafetensorsStore,
        device: &B::Device,
    ) -> Result<GraphData<B>, String> {
        Ok(GraphData {
            data_area_weight: Tensor::from_data(snapshot(store, "data.area_weight")?, device),
            data_x: Tensor::from_data(snapshot(store, "data.x")?, device),
            hidden_x: Tensor::from_data(snapshot(store, "hidden.x")?, device),
            data_to_hidden_edge_index: Tensor::from_data(
                snapshot(store, "data_to_hidden.edge_index")?,
                device,
            ),
            data_to_hidden_edge_direction: Tensor::from_data(
                snapshot(store, "data_to_hidden.edge_dirs")?,
                device,
            ),
            data_to_hidden_edge_length: Tensor::from_data(
                snapshot(store, "data_to_hidden.edge_length")?,
                device,
            ),
            hidden_to_data_edge_index: Tensor::from_data(
                snapshot(store, "hidden_to_data.edge_index")?,
                device,
            ),
            hidden_to_data_edge_direction: Tensor::from_data(
                snapshot(store, "hidden_to_data.edge_dirs")?,
                device,
            ),
            hidden_to_data_edge_length: Tensor::from_data(
                snapshot(store, "hidden_to_data.edge_length")?,
                device,
            ),
        })
    }
}

// This structure stores the bipartite edge list for one sub-graph.
// This connects source nodes to dest nodes.
// You can assume sorted by dest.
//
// attr carry the edge attributes we care about.
#[derive(Clone, Debug)]
pub struct EdgeIndex<B: Backend> {
    pub src: Tensor<B, 1, Int>, // Shape [E]
    pub dst: Tensor<B, 1, Int>, // [E]

    pub num_src: usize,
    pub num_dst: usize,
}

// TODO(putravu): functions related to this impl

// Cat combines the src, dst, col_ptr of all the indices
pub fn cat<B: Backend>(edge_indices: Vec<EdgeIndex<B>>) -> EdgeIndex<B> {
    let mut srcs = Vec::default();
    let mut dsts = Vec::default();
    let mut num_srcs = 0 as usize;
    let mut num_dsts = 0 as usize;

    for e in edge_indices.iter() {
        srcs.push(e.src.clone());
        dsts.push(e.dst.clone());

        num_srcs += e.num_src;
        num_dsts += e.num_dst;
    }

    EdgeIndex {
        src: Tensor::cat(srcs, 0),
        dst: Tensor::cat(dsts, 0),
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
