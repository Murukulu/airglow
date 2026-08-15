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
    graph::{self, GraphData},
};

#[derive(Config, Debug)]
pub struct GraphTransformerForwardMapperConfig {
    in_channels_src: usize,
    in_channels_dst: usize,
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
pub struct GraphTransformerForwardMapper<B: Backend> {
    emb_nodes_src: Linear<B>,
    emb_nodes_dst: Linear<B>,
    trainable: TrainableTensor<B, 2>,
    proc: GraphTransformerProcessorBlock<B>,

    edge_index: Tensor<B, 2, Int>,
    edge_inc: Tensor<B, 2, Int>,
    edge_attr: Tensor<B, 2>,
    n_src_base: usize,
    n_dst_base: usize,
}

impl GraphTransformerForwardMapperConfig {
    pub fn init<B: Backend>(
        &self,
        graph_data: &GraphData<B>,
        device: &B::Device,
    ) -> GraphTransformerForwardMapper<B> {
        let emb_nodes_src = LinearConfig::new(self.in_channels_src, self.hidden_dim).init(device);
        let emb_nodes_dst = LinearConfig::new(self.in_channels_dst, self.hidden_dim).init(device);

        let edge_index = graph_data.data_to_hidden_edge_index.clone();
        // NOTE: Order is important here.
        // TODO(saiputravu): Have this order read from the metadata, rather than hardcoding.
        let edge_attr = Tensor::cat(
            vec![
                graph_data.data_to_hidden_edge_length.clone(),
                graph_data.data_to_hidden_edge_direction.clone(),
            ],
            1,
        );
        let edge_inc = Tensor::from_ints(
            [
                [graph_data.num_data_nodes as i64],
                [graph_data.num_hidden_nodes as i64],
            ],
            device,
        );

        let edge_attr_shape = edge_attr.shape().dims::<2>();
        let trainable =
            TrainableTensorConfig::new(edge_attr_shape[0], self.trainable_size).init(device);

        let hidden_dim = ((self.hidden_dim as f64 * self.mlp_hidden_ratio) + 0.5) as usize;
        let edge_dim = edge_attr_shape[1] + self.trainable_size;

        let proc = GraphTransformerProcessorBlockConfig::new(
            self.hidden_dim, // in shape
            self.hidden_dim, // out shape
            hidden_dim,      // hidden dim
            self.num_heads,
            edge_dim, // the attributes and trainable features per edge.
            self.qk_norm,
            self.edge_pre_mlp,
        )
        .init(device);

        GraphTransformerForwardMapper {
            emb_nodes_src,
            emb_nodes_dst,
            trainable,
            proc,
            edge_index,
            edge_inc,
            edge_attr,
            // Note: This is the setup for encoder only. Decoder is different.
            n_src_base: graph_data.num_data_nodes,
            n_dst_base: graph_data.num_hidden_nodes,
        }
    }
}

impl<B: Backend> GraphTransformerForwardMapper<B> {
    pub fn forward(&self, x: PairTensor<B, 2>, batch_size: usize) -> PairTensor<B, 2> {
        let edge_attr = self.trainable.forward(self.edge_attr.clone(), batch_size);
        let (edge_index_src, edge_index_dst) =
            graph::expand_edges(self.edge_index.clone(), self.edge_inc.clone(), batch_size);

        let (_, x_dst) = self.proc.forward(
            self.pre_process(x.clone()),
            edge_attr,
            edge_index_src,
            edge_index_dst,
            self.n_src_base * batch_size,
            self.n_dst_base * batch_size,
        );
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
