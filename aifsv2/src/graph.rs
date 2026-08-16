use burn::prelude::*;
use burn_store::{ModuleStore, SafetensorsStore};

// The geometric half of the graph, as extracted from the .ckpt into safetensors.
//
// The 8 trainable columns per node and per edge are Parameters and live in the weights file
// instead; edge attributes here are already unit-std normalised.
//
// Example:
//   9 tensors
//    data.area_weight                                             [542080, 1]          F32
//    data.x                                                       [542080, 2]          F32
//    data_to_hidden.edge_dirs                                     [748348, 2]          F32
//    data_to_hidden.edge_index                                    [2, 748348]          I32
//    data_to_hidden.edge_length                                   [748348, 1]          F32
//    hidden.x                                                     [40320, 2]           F32
//    hidden_to_data.edge_dirs                                     [1626240, 2]         F32
//    hidden_to_data.edge_index                                    [2, 1626240]         I32
//    hidden_to_data.edge_length                                   [1626240, 1]         F32
//
#[derive(Debug)]
pub struct GraphData<B: Backend> {
    pub data_x: Tensor<B, 2>,   // [N_data, 2]; [lat, lon] radians, lon in 0..2pi
    pub hidden_x: Tensor<B, 2>, // [N_hidden, 2]; same encoding

    // Each edge_index is the bipartite edge list for one sub-graph, connecting source nodes to
    // destination nodes. Row 0 is the source, row 1 the destination -- anemoi bakes in the flip
    // from PyG's [target, source] convention when it builds the graph, so no flip is needed here.
    //
    // As extracted, both are sorted by destination and neither is sorted by source. Nothing in the
    // convolution depends on that -- it addresses by index, not position (see common.rs) -- but a
    // CSR-style reduction would, so the property is recorded rather than relied upon.
    pub data_to_hidden_edge_index: Tensor<B, 2, Int>, // [2, E_enc]; row 0 src (data), row 1 dst (hidden)
    pub data_to_hidden_edge_direction: Tensor<B, 2>,  // [E_enc, 2]; per edge direction feature.
    pub data_to_hidden_edge_length: Tensor<B, 2>,     // [E_enc, 1]; per edge length feature.

    pub hidden_to_data_edge_index: Tensor<B, 2, Int>, // [2, E_dec]; row 0 src (hidden), row 1 dst (data)
    pub hidden_to_data_edge_direction: Tensor<B, 2>,  // [E_dec, 2]; per edge direction feature.
    pub hidden_to_data_edge_length: Tensor<B, 2>,     // [E_dec, 1]; per edge length feature.

    pub data_area_weight: Tensor<B, 2>, // [N_data, 1]; training-loss weight, unused at inference. Ignore.

    // Information about sizes that makes it easier.
    pub num_data_nodes: usize,
    pub num_data_attr: usize,
    pub num_hidden_nodes: usize,
    pub num_hidden_attr: usize,
}

// Snapshots are lazy — to_data is where the bytes are actually read.
pub(crate) fn snapshot(store: &mut SafetensorsStore, name: &str) -> Result<TensorData, String> {
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
        let data_x = Tensor::from_data(snapshot(store, "data.x")?, device);
        let hidden_x = Tensor::from_data(snapshot(store, "hidden.x")?, device);
        let [num_data_nodes, num_data_attr] = data_x.shape().dims();
        let [num_hidden_nodes, num_hidden_attr] = hidden_x.shape().dims();
        Ok(GraphData {
            // TODO
            num_data_nodes,
            num_data_attr,
            num_hidden_nodes,
            num_hidden_attr,
            data_area_weight: Tensor::from_data(snapshot(store, "data.area_weight")?, device),
            data_x,
            hidden_x,
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

    /// A small graph with the real feature widths but a token number of nodes and edges.
    ///
    /// The real graph is unusable for a smoke test: 1,626,240 decoder edges projected to 1024
    /// channels is a 6.7 GB activation, and graph_tranformer_conv holds four of those at once.
    /// This keeps the geometry (2 coordinate columns, 2 direction columns, 1 length column, so
    /// edge_dim comes out 11 exactly as in the checkpoint) and shrinks only the counts.
    ///
    /// Each hidden node is connected to every data node in both directions, which is dense but
    /// harmless at this size, and guarantees every node has at least one incident edge -- an
    /// isolated destination node would take the zero row out of the scatter and hide a bug.
    ///
    /// Attributes ramp rather than being constant: a LayerNorm over a constant row is 0 whatever
    /// the weights are, so constants would mask a mis-loaded norm.
    pub fn synthetic(num_data_nodes: usize, num_hidden_nodes: usize, device: &B::Device) -> Self {
        let ramp = |rows: usize, cols: usize| {
            Tensor::<B, 1, Int>::arange(0..(rows * cols) as i64, device)
                .float()
                .reshape([rows, cols])
                * 0.1
        };
        // A complete bipartite edge list, [2, E], src in row 0 and dst in row 1.
        let edges = |num_src: usize, num_dst: usize| {
            let src: Vec<i64> = (0..num_src)
                .flat_map(|s| std::iter::repeat_n(s as i64, num_dst))
                .collect();
            let dst: Vec<i64> = (0..num_src).flat_map(|_| 0..num_dst as i64).collect();
            let ints = |v: Vec<i64>| {
                let n = v.len();
                Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), device)
            };
            Tensor::stack::<2>(vec![ints(src), ints(dst)], 0)
        };

        let num_encoder_edges = num_data_nodes * num_hidden_nodes;
        let num_decoder_edges = num_hidden_nodes * num_data_nodes;

        GraphData {
            data_x: ramp(num_data_nodes, 2),
            hidden_x: ramp(num_hidden_nodes, 2),

            data_to_hidden_edge_index: edges(num_data_nodes, num_hidden_nodes),
            data_to_hidden_edge_direction: ramp(num_encoder_edges, 2),
            data_to_hidden_edge_length: ramp(num_encoder_edges, 1),

            hidden_to_data_edge_index: edges(num_hidden_nodes, num_data_nodes),
            hidden_to_data_edge_direction: ramp(num_decoder_edges, 2),
            hidden_to_data_edge_length: ramp(num_decoder_edges, 1),

            data_area_weight: Tensor::ones([num_data_nodes, 1], device),
            num_data_nodes,
            num_hidden_nodes,
            // Coordinate width of data_x / hidden_x above: [lat, lon], as in the real graph.
            num_data_attr: 2,
            num_hidden_attr: 2,
        }
    }
}

// From a starting edge_index, we increment known edges by some edge_inc. We expand
// each edge batch_size number of times.
//
// This is so that each node in each sub-graph has unique identifiers.
//
// Takes the [2, E] block layout the graph is stored in and returns the two [E * batch_size] index
// arrays the convolution consumes, batch-major: copy i names nodes offset by i * edge_inc, where
// edge_inc is [[num_src], [num_dst]] for this sub-graph. TrainableTensor::forward tiles the edge
// attributes in the same order, and the two must agree.
//
// Destination-sortedness (see GraphData) survives this: each copy is offset by num_dst and the
// copies are concatenated in order, so dst stays non-decreasing across the whole result.
pub fn expand_edges<B: Backend>(
    edge_index: Tensor<B, 2, Int>,
    edge_inc: Tensor<B, 2, Int>,
    batch_size: usize,
) -> (Tensor<B, 1, Int>, Tensor<B, 1, Int>) {
    let mut edge_indices = Vec::default();
    for i in 0..batch_size {
        // For each batch, increment the node identifiers.
        // new batch <- (edge_idx + i*inc)
        let new_edge_index = edge_index.clone() + (edge_inc.clone() * (i as i64));
        edge_indices.push(new_edge_index);
    }

    // Combine indices and cut out rows.
    let edge_indices = Tensor::cat(edge_indices, 1);
    let src = edge_indices
        .clone()
        .select(0, Tensor::from_ints([0], &edge_index.device()))
        .flatten(0, 1);
    let dst = edge_indices
        .clone()
        .select(0, Tensor::from_ints([1], &edge_index.device()))
        .flatten(0, 1);
    (src, dst)
}
