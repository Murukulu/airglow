use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

use crate::{
    decoder::{GraphTransformerBackwardMapper, GraphTransformerBackwardMapperConfig},
    encoder::{GraphTransformerForwardMapper, GraphTransformerForwardMapperConfig},
    transformer::{TransformerProcessor, TransformerProcessorConfig},
};

#[derive(Config, Debug)]
struct AifsV2Config {
    input_dim: usize,
    input_dim_latent: usize,
    num_channels: usize,

    #[config(default = 4.)]
    mlp_hidden_ratio: f64,

    #[config(default = 16)]
    num_heads: usize,

    #[config(default = 8)]
    trainable_size: usize,
}

#[derive(Module, Debug)]
struct AifsV2<B: Backend> {
    named_attribute: Linear<B>,
    encoder: GraphTransformerForwardMapper<B>,
    proc: TransformerProcessor<B>,
    decoder: GraphTransformerBackwardMapper<B>,
}

impl AifsV2Config {
    // FIXME(saiputravu): Fill these out correctly.
    pub fn init<B: Backend>(&self, device: &B::Device) -> AifsV2<B> {
        let named_attribute = LinearConfig::new(1, 1).init(device);
        let encoder = GraphTransformerForwardMapperConfig::new(
            self.input_dim,
            self.input_dim_latent,
            self.num_channels,
            self.mlp_hidden_ratio,
            self.num_heads,
            0,
            0,
            0,
            self.trainable_size,
        )
        .init(device);
        let proc = TransformerProcessorConfig::new(self.num_channels, 0, 0, 0, 0).init(device);
        let decoder = GraphTransformerBackwardMapperConfig::new(
            self.num_channels,
            self.input_dim,
            0,
            self.num_channels,
            self.mlp_hidden_ratio,
            self.num_heads,
            0,
            0,
            0,
            self.trainable_size,
        )
        .init(device);
        AifsV2 {
            named_attribute,
            encoder,
            proc,
            decoder,
        }
    }
}
