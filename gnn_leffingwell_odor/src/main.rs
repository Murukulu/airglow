use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    data::dataloader::{DataLoader, DataLoaderBuilder},
    optim::AdamConfig,
};

use crate::{model::ModelConfig, training::TrainingConfig};

mod data;
mod model;
mod smiles;
mod training;
mod utils;

fn debug() {
    let dataset = data::LeffingwellDataset::init("./data/leffingwell_data.csv").unwrap();

    let batch_size = 16;

    // Create device where to do the computation.
    type MyBackend = Wgpu<f32, i32>;
    let device = Default::default();
    let model = model::ModelConfig::new(dataset.classes, 100, 100, 100).init::<MyBackend>(&device);

    let batcher = data::ChemicalBatcher::default();
    let train: std::sync::Arc<dyn DataLoader<MyBackend, _>> = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .num_workers(4)
        .build(dataset);

    println!("{model}");

    // Test the dataset.
    for el in train.iter().take(1) {
        let d = el.edges.to_data();
        let n = d.shape[1];
        let collect: Vec<Vec<i32>> = d
            .to_vec::<i32>()
            .unwrap()
            .chunks(n)
            .map(|r| r.to_vec())
            .collect();

        for c in collect {
            println!("{:?}", c);
        }
        println!("{:?}", el.edge_features);
        println!("{:?}", el.node_features);
        println!("{:?}", el.targets);
    }
}

fn main() {
    // Create device where to do the computation.
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = WgpuDevice::default();
    let artifact_dir = "./artifacts/";

    // TODO(saiputravu): Fix this class dependency on dataset.
    let dataset = data::LeffingwellDataset::init("./data/leffingwell_data.csv").unwrap();
    training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(dataset.classes, smiles::ATOM_FEATURE_DIM, 10, 10),
            AdamConfig::new(),
        ),
        device,
    );
}
