use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        activation::Activation::{self, Gelu},
    },
    prelude::*,
    serde,
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

// TODO(saiputravu): Clean up these comments.
// This can be replaced with https://arxiv.org/pdf/2511.11581b
// (Triton Attention Kernel). This is also an implementation of the
// pytorch_geometric MessagePassing
// (https://github.com/pyg-team/pytorch_geometric/blob/cc678a392255a1467872f54582724b8dce434603/torch_geometric/nn/conv/message_passing.py#L39)

#[derive(Config, Debug)]
pub struct GraphTransformerConvConfig {
    out_channels: usize,
    dropout: f64,

    aggr_type: String, // TODO(saiputravu): This should be an enum...
    flow: String,      // TODO(saiputravu): This should be an enum...
    node_dim: usize,
    fuse: bool,

    // Feature decomp. paper: https://arxiv.org/abs/2104.03058
    #[config(default = 1)]
    decomposed_layers: usize,
}

#[derive(Module, Debug)]
pub struct GraphTransformerConv<B: Backend> {
    dropout: Dropout,
    // FIXME(saiputravu):
    fuse: bool,
    decomposed_layers: usize,
}

impl GraphTransformerConvConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerConv<B> {
        GraphTransformerConv {
            dropout: Dropout { prob: self.dropout },
            linear_dummy: LinearConfig::new(0, 0).init(device),
            fuse: self.fuse,
            decomposed_layers: self.decomposed_layers,
        }
    }
}

struct Adj<B: Backend, const D: usize> {
    adj_t: Tensor<B, D>,
    e_id: Option<Tensor<B, D>>,
    size: (usize, usize),
}

impl<B: Backend> GraphTransformerConv<B> {
    // https://github.com/pyg-team/pytorch_geometric/blob/cc678a392255a1467872f54582724b8dce434603/torch_geometric/nn/aggr/basic.py#L12
    fn sum_forward<const D: usize>(&self, x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
        x.sum_dim(dim)
    }

    // Message propagation.
    fn propagate<const D: usize>(
        &self,
        edge_index: Adj<B, D>,
        size: Option<usize>,
    ) -> Tensor<B, D> {
        if self.fuse {
            let out = self.message_and_aggregate(edge_index);
            let out = self.update(out);
            return out;
        } else {
            let mut decomp_out: Vec<Tensor<B, D>> = Vec::default();
            // Else run both functions in separation?
            // TODO(saiputravu): Do some reading on fused vs. non-fused.
            for i in 0..self.decomp {
                let out = self.message(..);
                let out = self.aggregate(out);
                let out = self.update(out);
                decomp_out.push(out);
            }
            // FIXME(saiputravu): Fix this sloppy mess.
            return Tensor::cat(decomp_out, decomp_out[0].shape()[-1]);
        }
    }

    fn forward() {}
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

    conv_dropout: f64,
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

    conv: GraphTransformerConv<B>,
}

impl GraphTransformerMapperBlockConfig {
    fn out_channels_conv(&self) -> usize {
        let out_channels_conv = self.attn_channels / self.num_heads;
        out_channels_conv
    }

    fn init<B: Backend>(&self, device: &B::Device) -> GraphTransformerMapperBlock<B> {
        let out_channels_conv = self.out_channels_conv();
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
            Some(
                MultiLayerPreceptronConfig::new(self.edge_dim, self.edge_dim, 0, 0, false)
                    .with_final_activation(true)
                    .init(device),
            )
        } else {
            None
        };
        // FIXME(saiputravu): Correct the dropout.
        let conv = GraphTransformerConvConfig::new(out_channels_conv, self.conv_dropout);

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
            conv,
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
