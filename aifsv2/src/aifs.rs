use std::path::PathBuf;

use burn::{prelude::*, tensor::IndexingUpdateOp};
use burn_store::{PyTorchToBurnAdapter, SafetensorsStore};

use crate::{
    common::PairTensor,
    decoder::{GraphTransformerBackwardMapper, GraphTransformerBackwardMapperConfig},
    encoder::{GraphTransformerForwardMapper, GraphTransformerForwardMapperConfig},
    graph::GraphData,
    metadata::Metadata,
    named_node_attributes::{NamedNodeAttributes, NamedNodeAttributesConfig, TensorType},
    processors::Processors,
    transformer::{TransformerProcessor, TransformerProcessorConfig},
};

#[derive(Config, Debug)]
pub struct AifsV2Config {
    num_channels: usize,
    metadata: Metadata,

    #[config(default = 4.)]
    mlp_hidden_ratio: f64,

    #[config(default = 16)]
    num_heads: usize,

    #[config(default = 8)]
    trainable_size: usize,

    #[config(default = 16)]
    num_processor_layers: usize,

    #[config(default = 2)]
    num_processor_chunks: usize,

    #[config(default = 1120)]
    window_size: usize,
}

#[derive(Module, Debug)]
pub struct AifsV2<B: Backend> {
    named_attribute: NamedNodeAttributes<B>,
    encoder: GraphTransformerForwardMapper<B>,
    proc: TransformerProcessor<B>,
    decoder: GraphTransformerBackwardMapper<B>,

    // Plain tensors, not Params: Burn records these as EmptyRecord, so they are invisible to the
    // store and never show up as missing keys.
    input_prognostic: Tensor<B, 1, Int>,
    output_prognostic: Tensor<B, 1, Int>,

    // Metadata
    metadata: Metadata,
}

impl AifsV2Config {
    /// Everything but num_channels is pinned by the dataset. num_channels comes from
    /// config.model.num_channels in the raw checkpoint JSON, which Metadata does not parse, so it
    /// is passed in (1024 for aifs-single-mse-2.0).
    pub fn from_metadata(metadata: &Metadata, num_channels: usize) -> Self {
        Self::new(num_channels, metadata.clone())
    }

    pub fn init<B: Backend>(&self, graph_data: &GraphData<B>, device: &B::Device) -> AifsV2<B> {
        let num_input_channels = self.metadata.model_input.full.len();
        let num_output_channels = self.metadata.model_output.full.len();
        let input_prognostic = &self.metadata.model_input.prognostic;
        let output_prognostic = &self.metadata.model_output.prognostic;
        let multistep = self.metadata.multistep;

        // A mismatch would otherwise surface as an opaque shape error inside select_assign, well
        // after the point where the configuration went wrong.
        assert_eq!(
            input_prognostic.len(),
            output_prognostic.len(),
            "input_prognostic has {} channels but output_prognostic has {}; they name the same variables",
            input_prognostic.len(),
            output_prognostic.len(),
        );

        let named_attribute =
            NamedNodeAttributesConfig::new(self.trainable_size).init(graph_data, device);

        let input_dim = (multistep * num_input_channels)
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
        let proc = TransformerProcessorConfig::new(
            self.num_channels,
            self.num_processor_layers,
            self.num_processor_chunks,
            self.num_heads,
            self.window_size,
        )
        .init(device);
        let decoder = GraphTransformerBackwardMapperConfig::new(
            self.num_channels,
            input_dim,
            num_output_channels,
            self.num_channels,
            self.mlp_hidden_ratio,
            self.num_heads,
            self.trainable_size,
        )
        .init(graph_data, device);

        let indices = |idx: &[usize]| {
            Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    idx.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                    [idx.len()],
                ),
                device,
            )
        };

        AifsV2 {
            named_attribute,
            encoder,
            proc,
            decoder,
            input_prognostic: indices(input_prognostic),
            output_prognostic: indices(output_prognostic),
            // TODO(saiputravu): Stop cloning this everywhere.
            metadata: self.metadata.clone(),
        }
    }
}

impl<B: Backend> AifsV2<B> {
    /// Flatten the assembled input into what the encoder consumes.
    ///
    /// Anemoi's `AnemoiModelInterface._assemble_input`, without the leading underscore: in Rust
    /// that prefix means "deliberately unused", which is the opposite of what these are.
    ///
    /// `x` is `[batch, time, grid, vars]` and already normalised. Returns the encoder's source and
    /// destination pair, plus the skip tensor that assemble_output needs -- the last timestep,
    /// `[batch * grid, vars]`, which would otherwise be unrecoverable once time is folded into the
    /// channel axis.
    pub fn assemble_input(&self, x: Tensor<B, 4>) -> (PairTensor<B, 2>, Tensor<B, 2>) {
        let [batch, time, grid, vars] = x.shape().dims();

        // [b, t, grid, vars] -> [b, grid, t, vars] -> [(b grid), (t vars)].
        // Time is the OUTER index of the channel axis: 0..vars is t-6h and vars..2*vars is t.
        // Reversing the two is silently wrong, no shape check downstream catches it.
        let x_flat = x
            .clone()
            .swap_dims(1, 2)
            .reshape([batch * grid, time * vars]);

        // The last timestep only, kept for the prognostic residual. [(b grid), vars]
        let x_skip = x
            .slice([0..batch, time - 1..time, 0..grid, 0..vars])
            .reshape([batch * grid, vars]);

        let x_data_latent = Tensor::cat(
            vec![
                x_flat,
                self.named_attribute.forward(TensorType::Data, batch),
            ],
            1,
        );
        let x_hidden_latent = self.named_attribute.forward(TensorType::Hidden, batch);

        ((x_data_latent, x_hidden_latent), x_skip)
    }

    /// Turn the decoder's output into the model's output.
    ///
    /// Anemoi's `AnemoiModelInterface._assemble_output`. `x_out` is the decoder's
    /// `[batch * grid, num_output_channels]` and `x_skip` the last timestep from assemble_input.
    ///
    /// TODO(saiputravu): Apply the boundings (ReluBounding on 26 vars, Hardtanh(0, 1) on 4, and
    /// two FractionBoundings), which run here in list order after the residual. See
    /// docs/grib-to-inference-pipeline.md §6.
    pub fn assemble_output(&self, x_out: Tensor<B, 2>, x_skip: Tensor<B, 2>) -> Tensor<B, 2> {
        // The prognostic residual, scattered across two different index spaces. grid_skip is 0,
        // so every grid point takes the residual.
        x_out.select_assign(
            1,
            self.output_prognostic.clone(),
            x_skip.select(1, self.input_prognostic.clone()),
            IndexingUpdateOp::Add,
        )
    }

    /// One model step.
    ///
    /// `x` is `[batch, time, grid, vars]`, already normalised. The return value is
    /// `[batch * grid, num_output_channels]`, still in normalised space: de-normalising belongs to
    /// the post-processing stage.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch = x.shape().dims::<4>()[0];
        let (x_latent, x_skip) = self.assemble_input(x);

        // The forward mapper passes its source through untouched: x_data_latent survives the
        // encoder and is what the decoder re-embeds with its own emb_nodes_dst.
        let (x_data_latent, x_latent) = self.encoder.forward(x_latent, batch);

        // latent_skip: true in the checkpoint config.
        let x_latent = self.proc.forward(x_latent.clone()) + x_latent;

        let x_out = self.decoder.forward((x_latent, x_data_latent), batch);

        self.assemble_output(x_out, x_skip)
    }

    /// One step end to end: the anemoi `predict_step`, pre- and post-processing included.
    ///
    /// `x` is `[batch, time, grid, vars]` in physical units, NaN where the source had no value.
    /// The return is `[batch * grid, num_output_channels]`, back in physical units.
    pub fn predict_step(&self, processors: &Processors<B>, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let pre = processors.pre(x);
        let y_hat = self.forward(pre.x.clone());
        processors.post(y_hat, &pre)
    }
}

/// A store over the AIFS checkpoint, remapped from the PyTorch key namespace onto our field paths.
///
/// PyTorchToBurnAdapter dispatches on the container type, so its [out, in] -> [in, out] transpose
/// only touches real Linear weights and its weight/bias -> gamma/beta rename only LayerNorms.
///
/// TrainableTensor.trainable and the latlons params are left alone. (This is why the *graph* store
/// must not use the adapter: those are raw arrays with no container type to dispatch on.)
///
/// allow_partial is deliberate. Without it load_from errors out on the first missing key instead
/// of returning the ApplyResult, which is the thing worth reading.
pub fn checkpoint_store(path: impl Into<PathBuf>) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        // Patterns apply in order, each to the result of the last.
        .with_key_remapping(r"^model\.node_attributes\.", "named_attribute.")
        .with_key_remapping(r"^model\.encoder\.", "encoder.")
        .with_key_remapping(r"^model\.decoder\.", "decoder.")
        .with_key_remapping(r"^model\.processor\.proc\.", "proc.proc.")
        // nn.Sequential numbers the GELU, MultiLayerPreceptron's Vec<Linear> does not. Matches
        // both `mlp.0` (processor blocks) and `node_dst_mlp.0` (mapper blocks).
        .with_key_remapping(r"(mlp)\.0\.", "${1}.layers.0.")
        .with_key_remapping(r"(mlp)\.2\.", "${1}.layers.1.")
        // anemoi aliases one module under two names; we keep the _dst spelling and let the bare
        // `layer_norm_attention` fall out as unused.
        .with_key_remapping(
            r"\.layer_norm_attention_dest\.",
            ".layer_norm_attention_dst.",
        )
        // node_data_extractor is an nn.Sequential(LayerNorm, Linear); we hold two named fields.
        .with_key_remapping(
            r"^decoder\.node_data_extractor\.0\.",
            "decoder.node_data_extractor_norm.",
        )
        .with_key_remapping(
            r"^decoder\.node_data_extractor\.1\.",
            "decoder.node_data_extractor.",
        )
        .allow_partial(true)
}

#[cfg(test)]
#[path = "aifs_test.rs"]
mod tests;
