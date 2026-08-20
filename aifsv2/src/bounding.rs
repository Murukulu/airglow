use crate::metadata::{BoundingConfig, ChannelKind, Metadata};
use burn::{
    prelude::*,
    tensor::{IndexingUpdateOp::Assign, activation::relu},
};

#[derive(Module, Debug)]
pub struct ReluBounding<B: Backend> {
    vars_idx: Tensor<B, 1, Int>, // [N_dst, vars]
}

impl<B: Backend> ReluBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        vars: &Vec<String>,
        kind: &ChannelKind,
        device: &B::Device,
    ) -> ReluBounding<B> {
        ReluBounding {
            vars_idx: metadata.tensor_channels_of_vec(vars, kind, device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, vars_dim] = x.shape().dims::<2>();
        let r = relu(x.clone().select(vars_dim, self.vars_idx.clone()));
        x.scatter_nd(self.vars_idx.clone(), r, Assign)
    }
}

#[derive(Module, Debug)]
pub struct HardtanhBounding<B: Backend> {
    vars_idx: Tensor<B, 1, Int>, // [N_dst, vars]
    min: f64,
    max: f64,
}

impl<B: Backend> HardtanhBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        min: &f64,
        max: &f64,
        vars: &Vec<String>,
        kind: &ChannelKind,
        device: &B::Device,
    ) -> HardtanhBounding<B> {
        let vars_idx = metadata.tensor_channels_of_vec(vars, kind, device);
        HardtanhBounding {
            vars_idx,
            min: min.clone(),
            max: max.clone(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, vars_dim] = x.shape().dims::<2>();
        // Hardtanh is the piecewise function
        // (----------------------------)
        // |   max_val if x > max_val   |
        // |   min_val if x < min_val   |
        // |   x       else             |
        // (----------------------------)
        let h = x
            .clone()
            .select(vars_dim, self.vars_idx.clone())
            .clamp_max(self.max)
            .clamp_min(self.min);
        x.scatter_nd(self.vars_idx.clone(), h, Assign)
    }
}

#[derive(Module, Debug)]
pub struct FractionBounding<B: Backend> {
    total_var_idx: Tensor<B, 1, Int>, // [N_dst, 1]
    vars_idx: Tensor<B, 1, Int>,      // [N_dst, vars]
    min: f64,
    max: f64,
}

impl<B: Backend> FractionBounding<B> {
    pub fn new_init(
        metadata: &Metadata,
        min: &f64,
        max: &f64,
        total_var: &String,
        vars: &Vec<String>,
        kind: &ChannelKind,
        device: &B::Device,
    ) -> FractionBounding<B> {
        let total_var_idx = metadata.tensor_channels_of_vec(&vec![total_var.clone()], kind, device);
        let vars_idx = metadata.tensor_channels_of_vec(vars, kind, device);
        FractionBounding {
            total_var_idx,
            vars_idx,
            min: min.clone(),
            max: max.clone(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_, vars_dim] = x.shape().dims::<2>();

        // 1. Apply hardtanh.
        let h = x
            .clone()
            .select(vars_dim, self.vars_idx.clone())
            .clamp_max(self.max)
            .clamp_min(self.min);

        // 2. Multiply by total_variable; Anemoi calls "calculate fraction of the total var"
        // Here, total_var will be of shape [N, 1] and the value ranges for x will be normalised
        // as we expect this to be used in assemble_outputs.
        //
        // I guess this scales the total_var by the the hardtanh'd variable, which might be where
        // the fraction in the name comes from...
        let f = h * x.clone().select(vars_dim, self.total_var_idx.clone());

        x.scatter_nd(self.vars_idx.clone(), f, Assign)
    }
}

#[derive(Module, Debug)]
pub enum BoundingType<B: Backend> {
    Relu(ReluBounding<B>),
    Hardtanh(HardtanhBounding<B>),
    Fraction(FractionBounding<B>),
}

impl<B: Backend> BoundingType<B> {
    pub fn from_bounding_config(
        metadata: &Metadata,
        conf: &BoundingConfig,
        kind: &ChannelKind,
        device: &B::Device,
    ) -> BoundingType<B> {
        match conf {
            BoundingConfig::Relu { variables } => {
                BoundingType::Relu(ReluBounding::new_init(metadata, variables, kind, device))
            }
            BoundingConfig::Hardtanh {
                variables,
                min_val,
                max_val,
            } => BoundingType::Hardtanh(HardtanhBounding::new_init(
                metadata, min_val, max_val, variables, kind, device,
            )),
            BoundingConfig::Fraction {
                variables,
                min_val,
                max_val,
                total_var,
            } => BoundingType::Fraction(FractionBounding::new_init(
                metadata, min_val, max_val, total_var, variables, kind, device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            BoundingType::Relu(r) => r.forward(x),
            BoundingType::Hardtanh(h) => h.forward(x),
            BoundingType::Fraction(f) => f.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct Bounding<B: Backend> {
    boundings: Vec<BoundingType<B>>,
}

impl<B: Backend> Bounding<B> {
    pub fn new(boundings: Vec<BoundingType<B>>) -> Bounding<B> {
        Bounding { boundings }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut a = x;
        for b in self.boundings.iter() {
            a = b.forward(a);
        }
        a
    }
}
