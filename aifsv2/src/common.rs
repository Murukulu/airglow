use burn::{nn::{ Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, activation::Activation}, prelude::*};

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
    fn init<B: Backend>(&self, device: &B::Device) -> MultiLayerPreceptron<B> {
        // Compute the linear layers (first + hidden + last).
        let mut layer_confs = vec![LinearConfig::new(self.in_features, self.hidden_dim)];
        if self.n_extra_layers != 0 {
            layer_confs.extend(vec![LinearConfig::new(self.hidden_dim, self.hidden_dim); self.n_extra_layers]);
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
