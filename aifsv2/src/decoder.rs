use burn::{
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::{
    block::{GraphTransformerProcessorBlock, GraphTransformerProcessorBlockConfig},
    common::{PairTensor, TrainableTensor, TrainableTensorConfig},
    graph::{self, GraphData},
};

#[derive(Config, Debug)]
pub struct GraphTransformerBackwardMapperConfig {
    in_channels_src: usize,
    in_channels_dst: usize,
    out_channels_dst: usize,
    hidden_dim: usize,

    mlp_hidden_ratio: f64,
    num_heads: usize,

    trainable_size: usize,

    #[config(default = 1)] // Not expecting to do sharding.
    num_chunks: usize,
    #[config(default = false)]
    qk_norm: bool,
    #[config(default = false)]
    edge_pre_mlp: bool,
}

#[derive(Module, Debug)]
pub struct GraphTransformerBackwardMapper<B: Backend> {
    node_data_extractor_norm: LayerNorm<B>,
    node_data_extractor: Linear<B>,
    emb_nodes_dst: Linear<B>,
    proc: GraphTransformerProcessorBlock<B>,
    trainable: TrainableTensor<B, 2>,

    edge_index: Tensor<B, 2, Int>,
    edge_inc: Tensor<B, 2, Int>,
    edge_attr: Tensor<B, 2>,
    n_src_base: usize,
    n_dst_base: usize,
}

impl GraphTransformerBackwardMapperConfig {
    pub fn init<B: Backend>(
        &self,
        graph_data: &GraphData<B>,
        device: &B::Device,
    ) -> GraphTransformerBackwardMapper<B> {
        let node_data_extractor_norm = LayerNormConfig::new(self.hidden_dim).init(device);
        let node_data_extractor =
            LinearConfig::new(self.hidden_dim, self.out_channels_dst).init(device);
        let emb_nodes_dst = LinearConfig::new(self.in_channels_dst, self.hidden_dim).init(device);

        let edge_index = graph_data.hidden_to_data_edge_index.clone();
        // NOTE: Order is important here.
        // TODO(saiputravu): Have this order read from the metadata, rather than hardcoding.
        let edge_attr = Tensor::cat(
            vec![
                graph_data.hidden_to_data_edge_length.clone(),
                graph_data.hidden_to_data_edge_direction.clone(),
            ],
            1,
        );
        let edge_inc = Tensor::from_ints(
            [
                [graph_data.num_hidden_nodes as i64],
                [graph_data.num_data_nodes as i64],
            ],
            device,
        );

        let edge_attr_shape = edge_attr.shape().dims::<2>();
        let trainable =
            TrainableTensorConfig::new(edge_attr_shape[0], self.trainable_size).init(device);

        let edge_dim = edge_attr_shape[1] + self.trainable_size;
        let hidden_dim = ((self.hidden_dim as f64 * self.mlp_hidden_ratio) + 0.5) as usize;
        let proc = GraphTransformerProcessorBlockConfig::new(
            self.hidden_dim, // in
            self.hidden_dim, // out
            hidden_dim,      // hidden
            self.num_heads,
            edge_dim, // Number of edge attributes + trainable.
            self.qk_norm,
            self.edge_pre_mlp,
        )
        .init(device);

        GraphTransformerBackwardMapper {
            node_data_extractor_norm,
            node_data_extractor,
            emb_nodes_dst,
            proc,
            trainable,
            edge_index,
            edge_inc,
            edge_attr,

            n_src_base: graph_data.num_hidden_nodes,
            n_dst_base: graph_data.num_data_nodes,
        }
    }
}

impl<B: Backend> GraphTransformerBackwardMapper<B> {
    pub fn forward(&self, x: PairTensor<B, 2>, batch_size: usize) -> Tensor<B, 2> {
        let edge_attr = self.trainable.forward(self.edge_attr.clone(), batch_size);
        let (edge_index_src, edge_index_dst) =
            graph::expand_edges(self.edge_index.clone(), self.edge_inc.clone(), batch_size);

        // Apply pre-processing then processing.
        let (_, x_dst) = self.proc.forward(
            self.pre_process(x.clone()),
            edge_attr,
            edge_index_src,
            edge_index_dst,
            self.n_src_base * batch_size,
            self.n_dst_base * batch_size,
        );

        // Return the result of post-processing.
        self.post_process(x_dst)
    }

    // Embedding linear projection of the destination-domain input features.
    // Leave source-domain input features alone.
    fn pre_process(&self, x: PairTensor<B, 2>) -> PairTensor<B, 2> {
        let (x_src, x_dst) = x;
        let x_dst = self.emb_nodes_dst.forward(x_dst);
        (x_src, x_dst)
    }

    // Linear projection of the destination-domain input features.
    fn post_process(&self, x_dst: Tensor<B, 2>) -> Tensor<B, 2> {
        self.node_data_extractor
            .forward(self.node_data_extractor_norm.forward(x_dst))
    }
}

#[cfg(test)]
#[path = "decoder_test.rs"]
mod tests;
