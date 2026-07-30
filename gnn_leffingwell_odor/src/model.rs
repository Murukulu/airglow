use burn::{
    self, Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Shape, TensorMetadata, TensorPrimitive, ops::FloatTensor},
};

#[derive(Module, Debug)]
struct GCNLayer<B: Backend> {
    weights: Param<Tensor<B, 2>>,
}

#[derive(Config, Debug)]
struct GCNLayerConfig {
    input_shape: usize,
    output_shape: usize,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    initializer: Initializer,
}

impl GCNLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GCNLayer<B> {
        let shape = [self.input_shape, self.output_shape];
        let weights = self.initializer.init_with(
            shape,
            Some(self.input_shape),
            Some(self.output_shape),
            device,
        );
        GCNLayer { weights }
    }
}

/// Reshape a tensor by prepending size-1 dimensions until it has `target_ndims` dimensions.
fn unsqueeze_leading<B: Backend>(tensor: FloatTensor<B>, target_ndims: usize) -> FloatTensor<B> {
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    if ndims >= target_ndims {
        return tensor;
    }
    let mut new_dims = vec![1usize; target_ndims - ndims];
    for i in 0..ndims {
        new_dims.push(shape[i]);
    }
    B::float_reshape(tensor, Shape::from(new_dims))
}

impl<B: Backend> GCNLayer<B> {
    pub fn forward(&self, a_tilde: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        Tensor::new(TensorPrimitive::Float(self.forward_(
            a_tilde.into_primitive().tensor(),
            x.into_primitive().tensor(),
            self.weights.val().into_primitive().tensor(),
        )))
    }

    pub fn forward_(
        &self,
        a_tilde: FloatTensor<B>,
        x: FloatTensor<B>,
        weights: FloatTensor<B>,
    ) -> FloatTensor<B> {
        let weights = unsqueeze_leading::<B>(weights, x.shape().num_dims());
        let output = B::float_matmul(a_tilde, x);
        let output = B::float_matmul(output, weights);
        output
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    layer1: GCNLayer<B>,
    layer2: GCNLayer<B>,
    linear: Linear<B>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            layer1: GCNLayerConfig::new(10, 10).init(device),
            layer2: GCNLayerConfig::new(10, 10).init(device),
            linear: LinearConfig::new(100, 2).init(device),
        }
    }
}
