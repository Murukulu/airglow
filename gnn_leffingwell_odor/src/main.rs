use std::sync::Arc;

use burn::{
    backend::Wgpu,
    data::dataloader::{DataLoader, DataLoaderBuilder},
};

mod data;
mod model;
mod smiles;

fn main() {
    let dataset = data::LeffingwellDataset::init("./data/leffingwell_data.csv").unwrap();

    // Create device where to do the computation.
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let model = model::ModelConfig::new().init::<MyBackend>(&device);

    let batcher = data::ChemicalBatcher::default();
    let train: std::sync::Arc<dyn DataLoader<MyBackend, _>> = DataLoaderBuilder::new(batcher)
        .batch_size(16)
        .num_workers(1)
        .build(dataset);

    println!("{model}");

    for el in train.iter().take(4) {
        println!("{:?}", el);
    }
}
