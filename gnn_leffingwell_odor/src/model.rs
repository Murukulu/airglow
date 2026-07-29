use std::sync::Arc;

use burn::{
    self, Tensor,
    module::{AutodiffModule, Module},
    nn::{Linear, LinearConfig},
    prelude::Backend,
    tensor::{
        Shape,
        activation::{self, quiet_softmax, relu, softmax},
        module::linear,
    },
};

struct GCNLayer<B: Backend> {
    activation: fn(Tensor<B, 2>) -> Tensor<B, 2>,
    weights: Tensor<B, 2>,
}
pub struct Model<B: Backend> {
    layer1: GCNLayer<B>,
    layer2: GCNLayer<B>,
    linear: Linear<B>,
}

impl<B: Backend> GCNLayer<B> {
    pub fn init(
        input_shape: usize,
        output_shape: usize,
        activation: fn(Tensor<B, 2>) -> Tensor<B, 2>,
        device: &B::Device,
    ) -> GCNLayer<B> {
        let shape = Shape::new([input_shape, output_shape]);
        let weights = Tensor::zeros(shape, device);
        GCNLayer {
            activation: activation,
            weights: weights,
        }
    }

    pub fn forward(&self, a_tilde: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        (self.activation)(a_tilde.matmul(x).matmul(self.weights.clone()))
    }
}

impl<B: Backend> Model<B> {
    pub fn init(device: &B::Device) -> Model<B> {
        Model {
            // TODO(putravu): Need to init these.
            layer1: GCNLayer::init(10, 10, relu, device),
            layer2: GCNLayer::init(10, 10, |tensor| quiet_softmax(tensor, 0), device),
            linear: LinearConfig::new(100, 2).init(device),
        }
    }

    // pub fn forward(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // }
}

fn gnn_model_forward<B: Backend>(device: &B::Device) {
    let model = Model::<B>::init(device);
}
