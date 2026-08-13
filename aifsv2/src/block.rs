use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::backend::Backend,
};

use crate::{
    common::{MultiLayerPreceptron, MultiLayerPreceptronConfig, PairTensor, graph_tranformer_conv},
    graph::EdgeIndex,
};

// Ref:
// https://github.com/ecmwf/anemoi-core/blob/6aa1dc2a2b929211fe1c633ddbaeb68bc8fc7adf/models/src/anemoi/models/layers/block.py#L1032
#[derive(Config, Debug)]
pub struct GraphTransformerProcessorBlockConfig {
    in_channels: usize,
    out_channels: usize,
    hidden_dim: usize,
    num_heads: usize,
    attn_channels: usize,
    edge_dim: usize,
    qk_norm: bool,
    edge_pre_mlp: bool,

    #[config(default = true)]
    bias: bool,
    #[config(default = false)]
    update_src_nodes: bool,
    // We removed all dropout params, since we are not intending to train.
}

// For more context related to GraphTransformers, see paper https://arxiv.org/pdf/2403.10667.
#[derive(Module, Debug)]
pub struct GraphTransformerProcessorBlock<B: Backend> {
    // Layer norms.
    // TODO(saiputravu): Think about which parts of this can just get replaced with MHA.
    layer_norm_attention_src: LayerNorm<B>,
    layer_norm_attention_dst: LayerNorm<B>, // Also called layer_norm_attention.
    layer_norm_mlp_src: Option<LayerNorm<B>>,
    layer_norm_mlp_dst: LayerNorm<B>,

    // Attention projections.
    lin_key: Linear<B>,
    lin_query: Linear<B>,
    lin_value: Linear<B>,
    lin_self: Linear<B>, // Projection for the prevention of over-smoothing (W_r).
    lin_edge: Linear<B>,

    node_src_mlp: Option<MultiLayerPreceptron<B>>,
    node_dst_mlp: MultiLayerPreceptron<B>,
    projection: Linear<B>,

    // Unused in current upstream.
    query_norm: Option<LayerNorm<B>>,
    key_norm: Option<LayerNorm<B>>,

    // Store the configuration itself.
    conf: GraphTransformerProcessorBlockConfig,
}

impl GraphTransformerProcessorBlockConfig {
    pub fn out_channels_conv(&self) -> usize {
        let out_channels_conv = self.attn_channels / self.num_heads;
        out_channels_conv
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerProcessorBlock<B> {
        let out_channels_conv = self.out_channels_conv();
        assert_eq!(
            self.attn_channels % self.num_heads,
            0,
            "number of heads {} does not evenly divide attention channels {}",
            self.num_heads,
            self.attn_channels
        );

        // Setup the norms.
        let layer_norm_attention_src = LayerNormConfig::new(self.in_channels).init(device);
        let layer_norm_attention_dst = LayerNormConfig::new(self.in_channels).init(device);
        let layer_norm_mlp_dst = LayerNormConfig::new(self.out_channels).init(device);

        // Setup linear projections.
        let lin_key =
            LinearConfig::new(self.in_channels, self.num_heads * out_channels_conv).init(device);
        let lin_query =
            LinearConfig::new(self.in_channels, self.num_heads * out_channels_conv).init(device);
        let lin_value =
            LinearConfig::new(self.in_channels, self.num_heads * out_channels_conv).init(device);
        let lin_self = LinearConfig::new(self.in_channels, self.num_heads * out_channels_conv)
            .with_bias(self.bias)
            .init(device);
        let lin_edge =
            LinearConfig::new(self.edge_dim, self.num_heads * out_channels_conv).init(device);

        // Setup MLPs.
        let node_dst_mlp =
            MultiLayerPreceptronConfig::new(self.out_channels, self.out_channels, self.hidden_dim)
                .init(device);

        // Setup final projection.
        let projection = LinearConfig::new(self.attn_channels, self.out_channels).init(device);

        // Setup options.
        let (node_src_mlp, layer_norm_mlp_src) = if self.update_src_nodes {
            (
                Some(
                    MultiLayerPreceptronConfig::new(
                        self.out_channels,
                        self.out_channels,
                        self.hidden_dim,
                    )
                    .init(device),
                ),
                Some(LayerNormConfig::new(self.out_channels).init(device)),
            )
        } else {
            (None, None)
        };

        let (query_norm, key_norm) = if self.qk_norm {
            (
                Some(
                    LayerNormConfig::new(out_channels_conv)
                        .with_bias(false)
                        .init(device),
                ),
                Some(
                    LayerNormConfig::new(out_channels_conv)
                        .with_bias(false)
                        .init(device),
                ),
            )
        } else {
            (None, None)
        };

        GraphTransformerProcessorBlock {
            layer_norm_attention_src,
            layer_norm_attention_dst,
            layer_norm_mlp_src,
            layer_norm_mlp_dst,
            lin_key,
            lin_query,
            lin_value,
            lin_self,
            lin_edge,
            node_src_mlp,
            node_dst_mlp,
            projection,
            query_norm,
            key_norm,
            conf: self.clone(),
        }
    }
}

impl<B: Backend> GraphTransformerProcessorBlock<B> {
    // This is an implementation of Anemoi GraphTransformerProcessorBlock, which is an implementation
    // on top of UniMP, https://arxiv.org/pdf/2403.10667 -- the graph transformer paper.
    //
    // This takes a PairTensor in, which is of shape (x_src, x_dst)
    //
    // The edge_attribute is 11 = 1 length + 2 dirs + 8 trainable.
    //
    // TODO(saiputravu): Look at all the dims here and document it because I'm quite confused.
    //
    // Note: return shape
    // ([n_src, F], [n_dst, F])
    pub fn forward(
        &self,
        x: PairTensor<B, 2>,     // ([N_src, F], [N_dst, F])
        edge_attr: Tensor<B, 2>, // [E, A]
        edge_index: EdgeIndex<B>,
    ) -> PairTensor<B, 2> {
        let x_skip_connection = x.clone();

        // Apply layer norm across pair tensor. Keeps the same shape.
        let x = (
            self.layer_norm_attention_src.forward(x.clone().0),
            self.layer_norm_attention_dst.forward(x.clone().1),
        );

        // Compute residual.
        let res = self.lin_self.forward(x.clone().1); // [n_dst, F]

        // Generate projection values and reshape. A 3D tensor is required for graph conv. New shapes:
        // query: [n_dst, H, C]
        // key:   [n_src, H, C]
        // value: [n_src, H, C]
        // edges: [E, H, C]
        let [query, key, value, edges] = self.get_qkve(x.clone(), edge_attr);

        // Apply norm across query and key, if requested.
        let (query, key) = if self.conf.qk_norm
            && let (Some(query_norm), Some(key_norm)) =
                (self.query_norm.clone(), self.key_norm.clone())
        {
            (query_norm.forward(query), key_norm.forward(key))
        } else {
            (query, key)
        };

        // Apply attention message aggregation. Shape:
        // alpha: [n_dst, H, C]
        // TODO(saiputravu): In the future, this is a candidate for chunking on edge_index, as anemoi does.
        let msg = graph_tranformer_conv(query, key, value, edges, edge_index).flatten(1, 2); // [n_dst, H, C] -> [n_dst, H*C] = [n_dst, F]

        let out = self.projection.forward(msg + res); // [n_dst, F]
        let out = out + x_skip_connection.clone().1; // [n_dst, F]

        // Generate new pair tensor.
        let nodes_new_dst = self
            .node_dst_mlp
            .forward(self.layer_norm_mlp_dst.forward(out.clone()))
            + out; // Add residual. Final shape [n_dst, F]

        let nodes_new_src = if self.conf.update_src_nodes
            && let (Some(node_src_mlp), Some(layer_norm_mlp_src)) =
                (self.node_src_mlp.clone(), self.layer_norm_mlp_src.clone())
        {
            node_src_mlp.forward(layer_norm_mlp_src.forward(x_skip_connection.clone().0))
                + x_skip_connection.0
        } else {
            x_skip_connection.0
        };

        // ([n_src, F], [n_dst, F])
        (nodes_new_src, nodes_new_dst)
    }

    // Here, we expect
    // x: ([n_src, F], [n_dst, F])
    // edge_attr: [E, A]
    // Return value:
    // for F=H*C
    // [q: [n_dst, H, C]; k, v: [n_src, H, C]; e: [E, H, C]]
    fn get_qkve(&self, x: PairTensor<B, 2>, edge_attr: Tensor<B, 2>) -> [Tensor<B, 3>; 4] {
        let (x_src, x_dst) = x;
        let h = self.conf.num_heads as i64;
        let c = self.conf.out_channels_conv() as i64;

        // Project and reshape by expanding matrix into tensor..
        let q = self.lin_query.forward(x_dst).reshape([-1, h, c]);
        let k = self.lin_key.forward(x_src.clone()).reshape([-1, h, c]);
        let v = self.lin_value.forward(x_src).reshape([-1, h, c]);
        // Anemoi does not do an edge attribute projection, so we do not either.
        let e = self.lin_edge.forward(edge_attr).reshape([-1, h, c]);

        [q, k, v, e]
    }
}

#[cfg(test)]
#[path = "block_test.rs"]
mod tests;
