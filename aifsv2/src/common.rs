use burn::{
    module::Param,
    nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, activation::Activation},
    prelude::*,
    tensor::IndexingUpdateOp,
};

use crate::graph::EdgeIndex;

// Note: This is a lazy implementation of the MLP implemented in Anemoi.
// We fix the activation to be GELU, as that is what it is in AIFS' MLP
// layers.
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

#[derive(Module, Debug)]
pub struct MultiLayerPreceptron<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Activation<B>,
    layer_norm: Option<LayerNorm<B>>,
    final_activation: bool,
}

impl MultiLayerPreceptronConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiLayerPreceptron<B> {
        // Compute the linear layers (first + hidden + last).
        let mut layer_confs = vec![LinearConfig::new(self.in_features, self.hidden_dim)];
        if self.n_extra_layers != 0 {
            layer_confs.extend(vec![
                LinearConfig::new(self.hidden_dim, self.hidden_dim);
                self.n_extra_layers
            ]);
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
            layer_norm,
            final_activation: self.final_activation,
            activation: Activation::Gelu(Gelu::new()),
        }
    }
}

impl<B: Backend> MultiLayerPreceptron<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
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

// TODO(putravu): Improve the comments here.
// https://github.com/pyg-team/pytorch_geometric/blob/cc678a392255a1467872f54582724b8dce434603/torch_geometric/utils/_softmax.py#L12
// We compute a sparsely evaluated softmax. This is necessary as we do not want to compute a softmax per a specifidc dim, but take
// softmaxes over a cluster of elements in a certain dimension. For example, take shape [E, H, 1]. If we compute a regular softmax,
// this will take the softmax for dim=0 over all of E; dim=1 over all of H; dim=2 over 1. This is not ideal, when we know we know
// we have X specific groups we want to take softmax over in dim=0.
//
// We write our own version of this.
//
// Note: return value [E, H, 1]
pub fn sparse_segment_softmax<B: Backend>(
    x: Tensor<B, 3>,            // [E, H, 1], one logit (0/1) per head, per edge.
    dst_idx: Tensor<B, 1, Int>, // [E], array of destination nodes, sorted by corresponding source nodes.
    n_dst: usize,               // The size of the dst_idx array after reduction.
) -> Tensor<B, 3> {
    let [e, h, _] = x.shape().dims();
    let _n = dst_idx.shape().dims::<1>()[0];
    assert_eq!(_n, e, "dst_idx has length {} but expected {}", _n, e);

    let device = x.device();

    // Burn does not have a scatter-max, where indices can be duplicate. It is easiest to just store a scalar value of
    // the global max. We unsqueeze the scalar into a tensor to operate.
    let m = x.clone().max().unsqueeze::<3>(); // [1] -> [1, 1, 1]

    // This computation should be per-segment max, but alas...
    // We shift by the max to reduce the IEEE754 error propagation due to floating point division.
    let numerator = (x - m).exp(); // [E, H, 1]

    // The softmax denominator summation. This will be of shape [n_dst, H, 1] as we want to work out the summation of
    // the [[ `sum([q_i^T(k_j+e_j) for j in neighbours(i)])` ]], for a given node i.
    //
    // Here, select_assign sum-reduces from numerator into the origination tensor (zeros), for all dst_idxs that are the
    // same. So we will get len(set(dst_idx)) elements, which will be edge-reduced.
    let denominator = Tensor::<B, 3>::zeros([n_dst, h, 1], &device).select_assign(
        0,
        dst_idx.clone(),
        numerator.clone(),
        IndexingUpdateOp::Add,
    );

    // We regather denominator and spray out to shape [E, H, 1], re-using the same denominator for source-domain nodes
    // sharing the destination-domain nodes.
    //
    // For example:
    // denom = [1.7, 1.0, 2.5]
    // dst = [0, 0, 0, 1, 2, 2, 2]
    // res = [1.7, 1.7, 1.7, 1.0, 2.5, 2.5, 2.5] <-- result of denom.select(0, dst)
    // num = [1.3, 4.3, 1.4, 9.1, 1.9, 1.6, 0.5]
    numerator / denominator.select(0, dst_idx)
}

// GraphTransformerConv, parameterless attention over a bipartite graph, where
// source is of the input data and the sink is of the hidden dim.
// Aggr = add.
//
// query: Shape [N_dst, Heads, Channels]
// key:   Shape [N_src, Heads, Channels]
// value: Shape [N_src, Heads, Channels]
// edges: Shape [E, Heads, Channels]
// edge_index: Bipartite edge list, filtering the relations we care about.
//
// return value: [N_dst, H, C]
//
// https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.nn.conv.TransformerConv.html
// Also shared by Anemoi in GraphTransformerConv implementation.
//
// The only difference here is that we don't apply a linear projection on the Q, K, V. So we remove
// all the matrix multiplication with W_{num} in the documentation.
//
// TODO(putravu): Comment formatting make everything the same line length and pick one and stick to it.
pub fn graph_tranformer_conv<B: Backend>(
    query: Tensor<B, 3>, // [N_dst, H, C]
    key: Tensor<B, 3>,   // [N_src, H, C]
    value: Tensor<B, 3>, // [N_src, H, C]
    edges: Tensor<B, 3>, // [E, H, C]
    edge_index: EdgeIndex<B>,
) -> Tensor<B, 3> {
    // We take the number of channels, inverse rooted. This is the attention normalisation
    // constant. I compute this value ahead of time, as we would prefer to keep higher
    // precision here? Multiplying offers less IEE754 error propagation vs. division of
    // small numbers.
    //
    // TODO(putravu): cite this.
    let shape = query.shape().dims::<3>();
    let norm = 1. / f64::sqrt(shape[2] as f64);
    let dst = edge_index.clone().dst; // [E]
    let src = edge_index.clone().src; // [E]
    let n_dst = shape[0];
    assert_eq!(
        n_dst, edge_index.num_dst,
        "found n_dst: {} but expected: {}",
        n_dst, edge_index.num_dst
    );

    // These tensors are now all of shape [E, H, C] since .dst and .src are of length E. See EdgeIndex comments for more
    // information.
    let q_i = query.select(0, dst.clone());
    let k_j = key.select(0, src.clone()) + edges.clone();
    let v_j = value.select(0, src) + edges.clone();

    // Here, we do element-wise multiplication. Burn here treats the last two dimensions
    // as the matrix and the former dims as batches. This means that they get ignored.
    //
    // We have already chosen all the edge values we care about. Once we do an elementwise
    // multiplication, we sum over the channels so that we end up with [E, H, 1]. In other
    // words, we get one vector for each attention head, for each edge index.
    let alpha = (q_i * k_j).sum_dim(-1) * norm;

    // Here, we compute the softmax for edges across the destination-domain. Here, alpha is [E, H, 1].
    // So we take the softmax of each alpha over the sum of groupings defined by dst indexer.
    let alpha = sparse_segment_softmax(alpha, dst.clone(), n_dst);

    // Confirm the shapes.
    let [_e, h, c] = edges.shape().dims::<3>();
    let [__e, __h, __one] = alpha.shape().dims::<3>();
    assert_eq!(
        [_e, h, 1],
        [__e, __h, __one],
        "found alpha shape: ({}, {}, {}) but expected shape ({}, {}, {})",
        __e,
        __h,
        __one,
        _e,
        h,
        1
    );

    // This is equivalent to computing the final output representation as defined in the paper. Here, the v_j
    // component has shape [E, H, C] whereas alpha has shape [E, H, 1]. This means the multiplication happens at
    // dim=-1. So every element per-edge, per-head will be scaled by alpha.
    let msg = v_j * alpha; // [E, H, C]

    // We scatter along the destination-domain indicies for message. So this convolution results in the destination
    // domain (i.e. convolve from source nodes -> destination nodes and encode features with importance).
    Tensor::zeros([n_dst, h, c], &edges.device()).select_assign(0, dst, msg, IndexingUpdateOp::Add)
}

#[derive(Config, Debug)]
pub struct TrainableTensorConfig {
    tensor_size: usize,
    trainable_size: usize,
}

#[derive(Module, Debug)]
pub struct TrainableTensor<B: Backend, const D: usize> {
    trainable: Param<Tensor<B, D>>,
}

impl TrainableTensorConfig {
    pub fn init<B: Backend, const D: usize>(&self, device: &B::Device) -> TrainableTensor<B, D> {
        assert!(
            self.trainable_size > 0,
            "trainable_size {} must be greater than 0",
            self.trainable_size
        );
        let trainable = Param::from_tensor(Tensor::zeros(
            [self.tensor_size, self.trainable_size],
            device,
        ));
        TrainableTensor { trainable }
    }
}

impl<B: Backend, const D: usize> TrainableTensor<B, D> {
    pub fn forward(&self, x: Tensor<B, D>, batch_size: usize) -> Tensor<B, D> {
        // TODO(saiputravu): Is this efficient? Can we do this?
        let trainable = self.trainable.clone().into_value().to_device(&x.device());
        // Nicely, for trainable tensors, we do not have to expand or reduce the dimensions, as these are just
        // graphs which are disconnected.
        let latent = vec![
            x.repeat_dim(0, batch_size),
            trainable.repeat_dim(0, batch_size),
        ];
        Tensor::cat(latent, D - 1)
    }
}
