use burn::{Tensor, backend::wgpu, prelude::*, tensor::backend::Backend};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

mod common;
mod encoder;
mod graph;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    inp: Tensor<B, 2>,
}

fn main() {
    type MyBackend = wgpu::Wgpu;
    let device: Device<MyBackend> = Default::default();
    let mut store = SafetensorsStore::from_file("../data/aifs-single-mse-2.0.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter);

    let mut model = Model {
        inp: Tensor::<MyBackend, 2>::zeros([1, 2], &device),
    };
    model.load_from(&mut store).unwrap();
    println!("Hello, world!");
    println!("Config: {:?}", model);
}
