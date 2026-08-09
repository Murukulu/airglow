use burn::{
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::{
    block::{GraphTransformerProcessorBlock, GraphTransformerProcessorBlockConfig},
    common::{PairTensor, TrainableTensor, TrainableTensorConfig},
    graph::{self, EdgeIndex},
};

#[derive(Config, Debug)]
struct GraphTransformerBackwardMapperConfig {
    in_channels_src: usize,
    in_channels_dst: usize,
    out_channels_dst: usize,
    hidden_dim: usize,

    mlp_hidden_ratio: f64,
    num_heads: usize,
    attn_channels: usize,
    edge_dim: usize,

    edge_attr_shape: usize,
    trainable_size: usize,

    #[config(default = 1)] // Not expecting to do sharding.
    num_chunks: usize,
    #[config(default = false)]
    qk_norm: bool,
    #[config(default = false)]
    edge_pre_mlp: bool,
}

#[derive(Module, Debug)]
struct GraphTransformerBackwardMapper<B: Backend> {
    node_data_extractor_norm: LayerNorm<B>,
    node_data_extractor: Linear<B>,
    emb_nodes_dst: Linear<B>,
    proc: GraphTransformerProcessorBlock<B>,
    trainable: TrainableTensor<B, 2>,
}

impl GraphTransformerBackwardMapperConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerBackwardMapper<B> {
        let node_data_extractor_norm = LayerNormConfig::new(self.hidden_dim).init(device);
        let node_data_extractor =
            LinearConfig::new(self.hidden_dim, self.out_channels_dst).init(device);
        let emb_nodes_dst = LinearConfig::new(self.in_channels_dst, self.hidden_dim).init(device);

        let hidden_dim = ((self.hidden_dim as f64 * self.mlp_hidden_ratio) + 0.5) as usize;
        let proc = GraphTransformerProcessorBlockConfig::new(
            self.hidden_dim, // in
            self.hidden_dim, // out
            hidden_dim,      // hidden
            self.num_heads,
            self.attn_channels,
            self.edge_dim,
            self.qk_norm,
            self.edge_pre_mlp,
        )
        .init(device);
        let trainable =
            TrainableTensorConfig::new(self.edge_attr_shape, self.trainable_size).init(device);
        GraphTransformerBackwardMapper {
            node_data_extractor_norm,
            node_data_extractor,
            emb_nodes_dst,
            proc,
            trainable,
        }
    }
}

impl<B: Backend> GraphTransformerBackwardMapper<B> {
    pub fn forward(
        &self,
        x: PairTensor<B, 2>,
        // TODO(saiputravu): Ingest heterograph? and avoid passing this information at forward.
        edge_attr: Tensor<B, 2>,
        edge_idx: EdgeIndex<B>,
        edge_inc: Tensor<B, 2, Int>,
        batch_size: usize,
    ) -> Tensor<B, 2> {
        let edge_attr = self.trainable.forward(edge_attr, batch_size);
        let edge_idx = graph::expand_edges(edge_idx, edge_inc, batch_size);

        // Apply pre-processing then processing.
        let (_, x_dst) = self
            .proc
            .forward(self.pre_process(x.clone()), edge_attr, edge_idx);

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
