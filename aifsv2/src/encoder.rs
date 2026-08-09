use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Int, backend::Backend},
};

use crate::{
    block::{GraphTransformerProcessorBlock, GraphTransformerProcessorBlockConfig},
    common::{PairTensor, TrainableTensor, TrainableTensorConfig},
    graph::{self, EdgeIndex},
};

#[derive(Config, Debug)]
pub struct GraphTransformerForwardMapperConfig {
    in_channels_src: usize,
    in_channels_dst: usize,
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
pub struct GraphTransformerForwardMapper<B: Backend> {
    emb_nodes_src: Linear<B>,
    emb_nodes_dst: Linear<B>,
    trainable: TrainableTensor<B, 2>,
    proc: GraphTransformerProcessorBlock<B>,
}

impl GraphTransformerForwardMapperConfig {
    // TODO(saiputravu): Think about how we want to construct edge_attr.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerForwardMapper<B> {
        let emb_nodes_src = LinearConfig::new(self.in_channels_src, self.hidden_dim).init(device);
        let emb_nodes_dst = LinearConfig::new(self.in_channels_dst, self.hidden_dim).init(device);

        let trainable =
            TrainableTensorConfig::new(self.edge_attr_shape, self.trainable_size).init(device);

        let hidden_dim = ((self.hidden_dim as f64 * self.mlp_hidden_ratio) + 0.5) as usize;
        let proc = GraphTransformerProcessorBlockConfig::new(
            self.hidden_dim, // in shape
            self.hidden_dim, // out shape
            hidden_dim,      // hidden dim
            self.num_heads,
            self.attn_channels,
            self.edge_dim,
            self.qk_norm,
            self.edge_pre_mlp,
        )
        .init(device);
        GraphTransformerForwardMapper {
            emb_nodes_src,
            emb_nodes_dst,
            trainable,
            proc,
        }
    }
}

impl<B: Backend> GraphTransformerForwardMapper<B> {
    pub fn forward(
        &self,
        x: PairTensor<B, 2>,
        // TODO(saiputravu): Ingest heterograph? and avoid passing this information at forward.
        edge_attr: Tensor<B, 2>,
        edge_idx: EdgeIndex<B>,
        edge_inc: Tensor<B, 2, Int>,
        batch_size: usize,
    ) -> PairTensor<B, 2> {
        let edge_attr = self.trainable.forward(edge_attr.clone(), batch_size);
        let edge_idx = graph::expand_edges(edge_idx, edge_inc, batch_size);

        let (_, x_dst) = self
            .proc
            .forward(self.pre_process(x.clone()), edge_attr, edge_idx);
        self.post_process();

        // Anemoi drops the source embedding on return. Only the destination node embedding changes.
        (x.0, x_dst)
    }

    fn pre_process(&self, x: PairTensor<B, 2>) -> PairTensor<B, 2> {
        let (x_src, x_dst) = x;
        let x_src = self.emb_nodes_src.forward(x_src);
        let x_dst = self.emb_nodes_dst.forward(x_dst);
        (x_src, x_dst)
    }
    fn post_process(&self) {}
}

#[cfg(test)]
#[path = "encoder_test.rs"]
mod tests;
