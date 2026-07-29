use burn::tensor::{Tensor, backend::Backend};
use purr::{graph::Builder, read::Trace, write::Writer};

mod smiles;

fn computation<B: Backend>() {
    // Create device where to do the computation.
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);
    println!("{:}", tensor1 + tensor2);
}

fn main() {
    println!("Hello, world!");
    computation::<burn::backend::Wgpu>();

    // Building a dataset.
    let mut dataset: Vec<smiles::Chemical> = Vec::new();

    // Not safe, just doing this for poc.
    let mut rdr = csv::Reader::from_path("./data/leffingwell_data.csv").unwrap();
    for res in rdr.deserialize() {
        let rec: smiles::Chemical = res.unwrap();
        // Add this graph data to the dataset.
        dataset.push(rec);
    }

    println!("{:?}", dataset.len())
}
