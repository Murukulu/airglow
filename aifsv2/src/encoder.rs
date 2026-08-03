use burn::{
    config::Config,
    module::Module,
    nn::{
        Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        activation::Activation::{self, Gelu},
    },
    prelude::*,
    tensor::backend::Backend,
};

// TODO(saiputravu): Move common modules to ./common.rs

type PairTensor<B: Backend, const D: usize> = (Tensor<B, D>, Tensor<B, D>);

#[derive(Config, Debug)]
pub struct MultiLayerPreceptronConfig {
    in_features: usize,
    out_features: usize,
    hidden_dim: usize,
    n_extra_layers: usize,
    layer_norm: bool,
    #[config(default = false)]
    final_activation: bool,
}

// Note: This is a lazy implementation of the MLP implemented in Anemoi.
// We fix the activation to be GELU, as that is what it is in AIFS' MLP
// layers.
#[derive(Module, Debug)]
pub struct MultiLayerPreceptron<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Activation<B>,
    layer_norm: Option<LayerNorm<B>>,
    final_activation: bool,
}

impl MultiLayerPreceptronConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiLayerPreceptron<B> {
        // Compute the linear layers (first + hidden + last).
        let mut layer_confs = vec![LinearConfig::new(self.in_features, self.hidden_dim)];
        if self.n_extra_layers != 0 {
            layer_confs.extend(vec![LinearConfig::new(self.hidden_dim, self.hidden_dim)]);
        }
        layer_confs.push(LinearConfig::new(self.hidden_dim, self.out_features));

        // Initialise the linear layers.
        let layers = layer_confs.iter().map(|c| c.init(device)).collect();

        // Setup layer norm, if we are using it.
        let layer_norm = if self.layer_norm {
            Some(LayerNormConfig::new(self.out_features).init(device))
        } else {
            None
        };

        MultiLayerPreceptron {
            layers,
            final_activation: self.final_activation,
            // Fix GELU, which is kind of hacky.
            activation: Gelu(Gelu { approximate: false }),
            layer_norm,
        }
    }

    fn build_hidden_layers<B: Backend>(
        &self,
        in_features: usize,
        out_features: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Vec<Linear<B>> {
        vec![LinearConfig::new(in_features, out_features).init(device); n_layers]
    }
}
impl<B: Backend> MultiLayerPreceptron<B> {
    fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let mut x = input;

        // Apply all layers, with the activation method.
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(x);
            if i != self.layers.len() - 1 || self.final_activation {
                x = self.activation.forward(x);
            }
        }

        // Return the final tensor, applying the layer norm, if specified.
        if let Some(layer_norm) = &self.layer_norm {
            layer_norm.forward(x)
        } else {
            x
        }
    }
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
    bias: bool,
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
    layer_norm_attention: LayerNorm<B>,
    layer_norm_mlp_dst: Option<LayerNorm<B>>,

    node_dst_mlp: MultiLayerPreceptron<B>,
    node_src_mlp: Option<MultiLayerPreceptron<B>>,

    // Edge pre-processing.
    edge_pre_mlp: Option<MultiLayerPreceptron<B>>,
}

impl GraphTransformerMapperBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerMapperBlock<B> {
        let out_channels_conv = self.attn_channels / self.num_heads;
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
        let projection = LinearConfig::new(self.attn_channels, self.out_channels).init(device);
        let query_norm = if self.qk_norm {
            Some(LayerNormConfig::new(out_channels_conv).init(device))
        } else {
            None
        };
        let key_norm = if self.qk_norm {
            Some(LayerNormConfig::new(out_channels_conv).init(device))
        } else {
            None
        };
        let layer_norm_attention = LayerNormConfig::new(self.in_channels).init(device);
        let layer_norm_mlp_dst = LayerNormConfig::new(self.out_channels).init(device);
        let node_dst_mlp = MultiLayerPreceptronConfig::new(
            self.out_channels,
            self.out_channels,
            self.hidden_dim,
            0,
            false,
        );
        let node_src_mlp = ();
        let edge_pre_mlp = if self.edge_pre_mlp {
        } else {
            None
        };
        GraphTransformerMapperBlock {
            lin_key,
            lin_query,
            lin_value,
            lin_self,
            lin_edge,
            projection,
            query_norm,
            key_norm,
            layer_norm_attention_src,
            layer_norm_attention_dst,
            layer_norm_mlp_src,
            node_dst_mlp,
            node_src_mlp,
            edge_pre_mlp,
        }
    }
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

impl GraphTransformerForwardMapperConfig {
    pub fn init<B: Backend>(&self) -> GraphTransformerForwardMapper<B> {
        let emb_nodes_src = LinearConfig::new(self.in_channels_src, self.hidden_dim);
        let emb_nodes_dst = LinearConfig::new(self.in_channels_dst, self.hidden_dim);
        // let proc = ;
        GraphTransformerForwardMapper {
            emb_nodes_src,
            emb_nodes_dst,
            proc: (),
        }
    }
}

impl<B: Backend> GraphTransformerForwardMapper<B> {}
