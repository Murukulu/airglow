use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, Linear, activation::Activation},
    tensor::backend::Backend,
};

// TODO(saiputravu): Move common modules to ./common.rs

#[derive(Config, Debug)]
pub struct MultiLayerPreceptronConfig {
    in_features: usize,
    out_features: usize,
    hidden_dim: usize,
    n_extra_layers: Option<usize>,
    final_activation: bool,
    layer_norm: bool,
}

#[derive(Module, Debug)]
pub struct MultiLayerPreceptron<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Activation<B>,
}

#[derive(Config, Debug)]
pub struct GraphTransformerMapperBlockConfig {
    in_channels: usize,
    out_channels: usize,
    hidden_dim: usize,
    num_heads: usize,
    attn_channels: usize,
    edge_dim: usize,
    update_src_nodes: bool,
    qk_norm: bool,
    edge_pre_mlp: bool,
    // TODO(saiputravu): Think about other parameters.
}

#[derive(Module, Debug)]
pub struct GraphTransformerMapperBlock<B: Backend> {
    lin_key: Linear<B>,
    lin_query: Linear<B>,
    lin_value: Linear<B>,
    lin_self: Linear<B>,
    lin_edge: Linear<B>,
    projection: Linear<B>,

    // These are equivalents of AutocastLayerNorm.
    query_norm: Option<LayerNorm<B>>,
    key_norm: Option<LayerNorm<B>>,

    // TODO(saiputravu): Think about which parts of this can just get replaced
    // with MHA.
    layer_norm_attention_src: LayerNorm<B>,
    layer_norm_attention_dst: LayerNorm<B>,
    layer_norm_mlp_src: Option<LayerNorm<B>>,

    node_dst_mlp: MultiLayerPreceptron<B>,
    node_src_mlp: Option<MultiLayerPreceptron<B>>,

    // Edge pre-processing.
    edge_pre_mlp: Option<MultiLayerPreceptron<B>>,
}

#[derive(Config, Debug)]
pub struct GraphTransformerForwardMapperConfig {
    in_channels_src: usize,
    in_channels_dst: usize,
    hidden_dim: usize,
    out_channels_dst: Option<usize>,
    num_chunks: usize,
}

#[derive(Module, Debug)]
pub struct GraphTransformerForwardMapper<B: Backend> {
    emb_nodes_src: Linear<B>,
    emb_nodes_dst: Linear<B>,
    proc: GraphTransformerMapperBlock<B>,
}
