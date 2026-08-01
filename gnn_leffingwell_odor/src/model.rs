use burn::{
    self, Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        Initializer, Linear, LinearConfig,
        loss::{BinaryCrossEntropyLoss, BinaryCrossEntropyLossConfig, CrossEntropyLossConfig},
    },
    prelude::*,
    tensor::{
        Int, TensorData, TensorMetadata, TensorPrimitive,
        activation::{relu, sigmoid, softmax},
        ops::FloatTensor,
    },
    train::{self},
};

use crate::utils;

#[derive(Config, Debug)]
struct GCNLayerConfig {
    num_feature_channels: usize,
    output_shape: usize,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    initializer: Initializer,
}

impl GCNLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GCNLayer<B> {
        let shape = [self.num_feature_channels, self.output_shape];
        let weights = self.initializer.init_with(
            shape,
            Some(self.num_feature_channels),
            Some(self.output_shape),
            device,
        );
        GCNLayer { weights }
    }
}

#[derive(Module, Debug)]
struct GCNLayer<B: Backend> {
    weights: Param<Tensor<B, 2>>,
}

impl<B: Backend> GCNLayer<B> {
    pub fn forward(&self, a_hat: &Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        Tensor::new(TensorPrimitive::Float(self.forward_(
            // TODO(saiputravu): think about cloning here.
            a_hat.clone().into_primitive().tensor(),
            x.into_primitive().tensor(),
            self.weights.val().into_primitive().tensor(),
        )))
    }

    pub fn forward_(
        &self,
        a_hat: FloatTensor<B>,
        x: FloatTensor<B>,
        weights: FloatTensor<B>,
    ) -> FloatTensor<B> {
        let weights = utils::unsqueeze_leading::<B>(weights, x.shape().num_dims());
        let output = B::float_matmul(a_hat, x);
        let output = B::float_matmul(output, weights);
        output
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    classes: usize,
    gcn1_feature_channels: usize,
    gcn1_output_shape: usize,
    gcn2_output_shape: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            layer1: GCNLayerConfig::new(self.gcn1_feature_channels, self.gcn1_output_shape)
                .init(device),
            layer2: GCNLayerConfig::new(self.gcn1_output_shape, self.gcn2_output_shape)
                .init(device),
            linear: LinearConfig::new(self.gcn2_output_shape, self.classes).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    layer1: GCNLayer<B>,
    layer2: GCNLayer<B>,
    linear: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        edges: Tensor<B, 2, Int>,
        // edge_features: Tensor<B, 2>,
        node_features: Tensor<B, 2>,
        batch_idxs: Tensor<B, 1, Int>,
        batch_size: usize,
    ) -> Tensor<B, 2> {
        let a_hat = self.compute_a_hat(edges.clone(), node_features.shape()[0], &edges.device()); // [N, N]
        let x = self.layer1.forward(&a_hat, node_features); // [N, F1]
        let x = relu(x); // [N, F1]
        let x = self.layer2.forward(&a_hat, x); // [N, F2]

        // At this point we have NxF2, for F1=gcn1_output_shape, F2=gcn2_output_shape.
        // I am going to flatten the nodes into a single feature vector.
        let x = utils::scatter_mean(x, batch_idxs, batch_size, &edges.device()); // [batch_size, Classes]
        let x = self.linear.forward(x); // [batch_size, Classes]
        let x = sigmoid(x); // [batch_size, Classes]
        x
    }

    pub fn forward_classification(
        &self,
        edges: Tensor<B, 2, Int>,
        // edge_features: Tensor<B, 2>,
        node_features: Tensor<B, 2>,
        targets: Tensor<B, 2, Int>,
        batch_idxs: Tensor<B, 1, Int>,
        batch_size: usize,
    ) -> train::MultiLabelClassificationOutput<B> {
        let output = self.forward(edges, node_features, batch_idxs, batch_size);
        let loss = BinaryCrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        train::MultiLabelClassificationOutput::new(loss, output, targets)
    }

    // This is unique per graph but only needs to be computed once.
    // Will return a square matrix.
    //
    // A_hat_ij = d_tilde_i ^(-1/2) . A_tilde . d_tilde_j^(-1/2)
    fn compute_a_hat(
        &self,
        edges: Tensor<B, 2, Int>,
        n: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        // Compute d_tilde
        let mut d_tilde: Vec<f32> = vec![0.; n];

        // TODO(saiputravu): Is cloning a bunch here expensive?
        // Sums.
        for e in edges.clone().iter_dim(0) {
            let edge = e.into_data().to_vec::<i32>().unwrap();
            let (i, j) = (edge[0] as usize, edge[1] as usize);
            d_tilde[i] += 1.;
            d_tilde[j] += 1.;
        }

        // 1/sqrt(d_i)
        let d_tilde: Vec<f32> = d_tilde
            .iter()
            .map(|d| {
                if *d == 0. {
                    0.
                } else {
                    // The + 1. here is for the self-loop, which was not accounted for earlier.
                    1. / f32::sqrt(d.clone() + 1.)
                }
            })
            .collect();

        // Compute A_tilde
        // TODO(saiputravu): CSR maybe format? or just setup the A_hat as a matrix.
        let mut a_tilde = vec![vec![0.; n]; n];
        for e in edges.clone().iter_dim(0) {
            let edge = e.into_data().to_vec::<i32>().unwrap();
            let (i, j) = (edge[0] as usize, edge[1] as usize);
            a_tilde[i][j] = d_tilde[i] * d_tilde[j];
        }
        // Add self loops.
        for k in 0..n {
            a_tilde[k][k] = d_tilde[k] * d_tilde[k];
        }

        Tensor::from_floats(
            TensorData::new::<f32, _>(a_tilde.iter().flatten().copied().collect(), [n, n]),
            device,
        )
    }
}
