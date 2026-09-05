use crate::{
    backend::{self, Backend},
    common::{MultiLayerPreceptron, MultiLayerPreceptronConfig},
    debug::dump,
};
use burn::{
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{module::attention, ops::AttentionModuleOptions},
};

// Writing custom MHA, since burn's nn::MultiHeadAttention does not satisfy:
// 1. No bias in the q/k/v linear projections. It biases all four.
// 2. Checkpoint-compatible field names. It calls them query/key/value/output, where the checkpoint
//    says lin_q/lin_k/lin_v/projection, and burn derives parameter paths from field names.
// 3. Windowed attention. Its only masking hook is a dense [batch, seq, seq] bool mask.
// 4. burn does not call burn::tensor::module::attention, which is its fused kernel, in MHA.
//
// Note, we do not do windowed attention.
// TODO(saiputravu): Windowed Attention.
#[derive(Config, Debug)]
struct MultiHeadSelfAttentionConfig {
    num_channels: usize,
    num_heads: usize,

    #[config(default = false)]
    qkv_bias: bool, // Whether to enable bias in q,k,v linear projections.
    // Softcap is default false and we do not allow it to be set.

    // Prefix for debug::dump of the projections and attention output; empty dumps nothing.
    #[config(default = "String::new()")]
    dump_tag: String,
}

#[derive(Module, Debug)]
struct MultiHeadSelfAttention<B: Backend> {
    lin_q: Linear<B>,
    lin_k: Linear<B>,
    lin_v: Linear<B>,
    projection: Linear<B>,

    num_heads: usize,
    head_dim: usize,
    dump_tag: String,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        // Anemoi asserts the same, at attention.py:87-89. Without it a bad config surfaces as a
        // reshape panic inside forward rather than as a statement about the config.
        assert_eq!(
            self.num_channels % self.num_heads,
            0,
            "number of heads {} does not evenly divide channels {}",
            self.num_heads,
            self.num_channels
        );

        let lin_q = LinearConfig::new(self.num_channels, self.num_channels)
            .with_bias(self.qkv_bias)
            .init(device);
        let lin_k = LinearConfig::new(self.num_channels, self.num_channels)
            .with_bias(self.qkv_bias)
            .init(device);
        let lin_v = LinearConfig::new(self.num_channels, self.num_channels)
            .with_bias(self.qkv_bias)
            .init(device);
        let projection = LinearConfig::new(self.num_channels, self.num_channels).init(device);
        MultiHeadSelfAttention {
            lin_q,
            lin_k,
            lin_v,
            projection,
            num_heads: self.num_heads,
            head_dim: self.num_channels / self.num_heads,
            dump_tag: self.dump_tag.clone(),
        }
    }
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Input tensor is of shape [batch, grid, channels]
        let [b, g, c] = x.shape().dims();
        assert_eq!(
            c,
            self.num_heads * self.head_dim,
            "found channels shape {}, expected {}",
            c,
            self.num_heads * self.head_dim
        );

        // The projections are dumped as [batch*grid, channels], before the head split, which is
        // the layout anemoi's lin_q/lin_k/lin_v produce.
        let query = self.lin_q.forward(x.clone());
        let key = self.lin_k.forward(x.clone());
        let value = self.lin_v.forward(x.clone());
        self.dump("query", &query);
        self.dump("key", &key);
        self.dump("value", &value);
        // swap_dims is a strided view and burn's flash attention kernel misreads the query in that
        // layout (attention_test.rs, Swapped rows; #31), so each is copied out to a real
        // [b, H, g, D] buffer first.
        //
        // TODO(saiputravu): Copy only the query. only_the_query_needs_to_be_contiguous shows key and
        // value are read correctly as views, bit for bit; the two copies are 165 MB each per layer.
        let split = |t: Tensor<B, 3>| {
            backend::contiguous(
                t.reshape([b, g, self.num_heads, self.head_dim])
                    .swap_dims(1, 2),
            )
        };
        let (query, key, value) = (split(query), split(key), split(value));
        self.dump_split("query_rearranged", &query);
        self.dump_split("key_rearranged", &key);
        self.dump_split("value_rearranged", &value);

        // Use fused attention.
        let attn = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            },
        ); // [b, H, g, D]

        let attn = attn.swap_dims(1, 2).reshape([b, g, c]); // [b, g, H, D] -> [b, g, c]
        self.dump("attn", &attn);
        let out = self.projection.forward(attn);
        self.dump("proj", &out);
        out
    }

    // debug::dump under this attention's tag as [batch*grid, channels]: `<tag>_<name>`, nothing
    // when the tag is empty.
    fn dump(&self, name: &str, t: &Tensor<B, 3>) {
        if !self.dump_tag.is_empty() {
            let [b, g, c] = t.shape().dims();
            dump(
                &format!("{}_{name}", self.dump_tag),
                &t.clone().reshape([b * g, c]),
            );
        }
    }

    // Like dump, but for the head-split q/k/v: their [batch, heads, grid, head_dim] shape is kept
    // as-is, since unlike the projections there is no [batch*grid, channels] layout to flatten to.
    fn dump_split(&self, name: &str, t: &Tensor<B, 4>) {
        if !self.dump_tag.is_empty() {
            dump(&format!("{}_{name}", self.dump_tag), t);
        }
    }
}

#[derive(Config, Debug)]
struct TransformerProcessorBlockConfig {
    num_channels: usize,
    hidden_dim: usize,
    window_size: usize,
    num_heads: usize,
    // We do not implement or use q_norm, k_norm.

    // Prefix for debug::dump of the block output; empty dumps nothing. With dump_intermediates
    // every stage of forward is dumped too. TransformerProcessorConfig sets both per layer.
    #[config(default = "String::new()")]
    dump_tag: String,
    #[config(default = false)]
    dump_intermediates: bool,
}

#[derive(Module, Debug)]
struct TransformerProcessorBlock<B: Backend> {
    layer_norm_attention: LayerNorm<B>,
    layer_norm_mlp: LayerNorm<B>,
    attention: MultiHeadSelfAttention<B>,
    mlp: MultiLayerPreceptron<B>,
    dump_tag: String,
    dump_intermediates: bool,
}

impl TransformerProcessorBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TransformerProcessorBlock<B> {
        let layer_norm_attention = LayerNormConfig::new(self.num_channels).init(device);
        let layer_norm_mlp = LayerNormConfig::new(self.num_channels).init(device);

        let attention = MultiHeadSelfAttentionConfig::new(self.num_channels, self.num_heads)
            .with_dump_tag(match self.dump_intermediates {
                true => self.dump_tag.clone(),
                false => String::new(),
            })
            .init(device);

        // Cast forward and back, with activation in the middle. I think this is an autoencoder.
        let mlp =
            MultiLayerPreceptronConfig::new(self.num_channels, self.num_channels, self.hidden_dim)
                .init(device);

        TransformerProcessorBlock {
            layer_norm_attention,
            layer_norm_mlp,
            attention,
            mlp,
            dump_tag: self.dump_tag.clone(),
            dump_intermediates: self.dump_intermediates,
        }
    }
}

impl<B: Backend> TransformerProcessorBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Attention takes 3D tensor where first dim is batch_size, so we unsqueeze here.
        // For this, we have a constant batch_size of 1. This is because graph batches are
        // just disjoint sub-graphs.
        let x_norm = self.layer_norm_attention.forward(x.clone());
        self.dump_intermediate("x_norm", &x_norm);
        let attn = self.attention.forward(x_norm.unsqueeze());

        // Drop the batch, since it is just one. Then add to input.
        let x = x + attn.squeeze::<2>();
        self.dump_intermediate("attn_res", &x);

        // Apply x + Lin(Activation(Lin(LayerNorm(x)))).
        let mlp_norm = self.layer_norm_mlp.forward(x.clone());
        self.dump_intermediate("mlp_norm", &mlp_norm);
        let mlp = self.mlp.forward(mlp_norm);
        self.dump_intermediate("mlp", &mlp);
        let out = x + mlp;
        self.dump("out", &out);
        out
    }

    // debug::dump under this block's tag: `<tag>_<name>`, nothing when the tag is empty.
    fn dump(&self, name: &str, t: &Tensor<B, 2>) {
        if !self.dump_tag.is_empty() {
            dump(&format!("{}_{name}", self.dump_tag), t);
        }
    }

    fn dump_intermediate(&self, name: &str, t: &Tensor<B, 2>) {
        if self.dump_intermediates {
            self.dump(name, t);
        }
    }
}

#[derive(Config, Debug)]
struct TransformerProcessorChunkConfig {
    num_channels: usize,
    num_layers: usize,
    window_size: usize,

    #[config(default = 16)]
    num_heads: usize,

    #[config(default = 4)]
    mlp_hidden_ratio: usize,
    // We do not implement or use q_norm, k_norm.

    // Global index of this chunk's first block: blocks are named proc<layer> with the layer
    // counted across chunks, as anemoi numbers them.
    #[config(default = 0)]
    layer_offset: usize,
    // TransformerProcessorConfig's knobs, passed down.
    #[config(default = false)]
    dump_outputs: bool,
    #[config(default = "Vec::new()")]
    dump_layers: Vec<usize>,
}

#[derive(Module, Debug)]
struct TransformerProcessorChunk<B: Backend> {
    blocks: Vec<TransformerProcessorBlock<B>>,
}

impl TransformerProcessorChunkConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TransformerProcessorChunk<B> {
        let blocks = (0..self.num_layers).map(|i| {
            let layer = self.layer_offset + i;
            let intermediates = self.dump_layers.contains(&layer);
            TransformerProcessorBlockConfig::new(
                self.num_channels,
                self.num_channels * self.mlp_hidden_ratio,
                self.window_size,
                self.num_heads,
            )
            .with_dump_tag(match self.dump_outputs || intermediates {
                true => format!("proc{layer}"),
                false => String::new(),
            })
            .with_dump_intermediates(intermediates)
        });
        TransformerProcessorChunk {
            blocks: blocks.map(|b| b.init(device)).collect(),
        }
    }
}

impl<B: Backend> TransformerProcessorChunk<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for b in self.blocks.iter() {
            x = b.forward(x);
        }
        x
    }
}

#[derive(Config, Debug)]
pub struct TransformerProcessorConfig {
    num_channels: usize,
    num_layers: usize, // Total number of layers, split evenly across the chunks.
    num_chunks: usize, // Number of chunks in processor.
    num_heads: usize,  // Number of heads in a transformer.

    window_size: usize, // 1/2 size of shifted window for attention computation.

    #[config(default = 4)]
    mlp_hidden_ratio: usize,

    // Debug dumps under the tag `proc<layer>`, 0-based across chunks, the names
    // scripts/ref_processor.py writes. dump_outputs writes every block's output (proc<layer>_out,
    // [40320, 1024] f32 per layer); dump_layers lists the layers that also dump every stage.
    #[config(default = false)]
    dump_outputs: bool,
    #[config(default = "Vec::new()")]
    dump_layers: Vec<usize>,
}

#[derive(Module, Debug)]
pub struct TransformerProcessor<B: Backend> {
    proc: Vec<TransformerProcessorChunk<B>>,
}

impl TransformerProcessorConfig {
    fn chunk_size(&self) -> usize {
        self.num_layers / self.num_chunks
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerProcessor<B> {
        // Anemoi asserts the same, at processor.py:57-59. An uneven split would silently drop the
        // remainder, so the processor would hold fewer layers than the checkpoint has.
        assert_eq!(
            self.num_layers % self.num_chunks,
            0,
            "number of processor layers ({}) has to be divisible by the number of processor chunks ({})",
            self.num_layers,
            self.num_chunks
        );

        // num_chunks chunks of chunk_size blocks each, for num_layers blocks in total.
        let chunk_size = self.chunk_size();
        let proc = (0..self.num_chunks).map(|c| {
            TransformerProcessorChunkConfig::new(self.num_channels, chunk_size, self.window_size)
                .with_num_heads(self.num_heads)
                .with_mlp_hidden_ratio(self.mlp_hidden_ratio)
                .with_layer_offset(c * chunk_size)
                .with_dump_outputs(self.dump_outputs)
                .with_dump_layers(self.dump_layers.clone())
        });
        TransformerProcessor {
            proc: proc.map(|p| p.init(device)).collect(),
        }
    }
}

impl<B: Backend> TransformerProcessor<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = x;
        for b in self.proc.iter() {
            x = b.forward(x);
        }
        x
    }
}

#[cfg(test)]
#[path = "transformer_test.rs"]
mod tests;
