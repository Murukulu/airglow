use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

use crate::{
    decoder::{GraphTransformerBackwardMapper, GraphTransformerBackwardMapperConfig},
    encoder::{GraphTransformerForwardMapper, GraphTransformerForwardMapperConfig},
    graph::GraphData,
    named_node_attributes::{NamedNodeAttributes, NamedNodeAttributesConfig},
    transformer::{TransformerProcessor, TransformerProcessorConfig},
};

#[derive(Config, Debug)]
struct AifsV2Config {
    num_channels: usize,
    num_input_channels: usize,
    num_output_channels: usize,

    multistep: usize,

    #[config(default = 4.)]
    mlp_hidden_ratio: f64,

    #[config(default = 16)]
    num_heads: usize,

    #[config(default = 8)]
    trainable_size: usize,
}

#[derive(Module, Debug)]
struct AifsV2<B: Backend> {
    named_attribute: NamedNodeAttributes<B>,
    encoder: GraphTransformerForwardMapper<B>,
    proc: TransformerProcessor<B>,
    decoder: GraphTransformerBackwardMapper<B>,
}

impl AifsV2Config {
    // FIXME(saiputravu): Fill these out correctly.
    pub fn init<B: Backend>(&self, graph_data: &GraphData<B>, device: &B::Device) -> AifsV2<B> {
        let named_attribute =
            NamedNodeAttributesConfig::new(self.trainable_size).init(graph_data, device);

        let input_dim = (self.multistep * self.num_input_channels)
            + ((2 * graph_data.num_data_attr) + self.trainable_size);
        let input_dim_latent = (2 * graph_data.num_hidden_attr) + self.trainable_size;

        let encoder = GraphTransformerForwardMapperConfig::new(
            input_dim,
            input_dim_latent,
            self.num_channels,
            self.mlp_hidden_ratio,
            self.num_heads,
            self.trainable_size,
        )
        .init(graph_data, device);
        let proc = TransformerProcessorConfig::new(self.num_channels, 0, 0, 0, 0).init(device);
        let decoder = GraphTransformerBackwardMapperConfig::new(
            self.num_channels,
            input_dim,
            self.num_output_channels,
            self.num_channels,
            self.mlp_hidden_ratio,
            self.num_heads,
            self.trainable_size,
        )
        .init(graph_data, device);
        AifsV2 {
            named_attribute,
            encoder,
            proc,
            decoder,
        }
    }
}
