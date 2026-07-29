use burn::tensor::{Tensor, backend::Backend};

mod model;
mod smiles;

fn computation<B: Backend>() {
    // Create device where to do the computation.
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);
    println!("{:}", tensor1 + tensor2);
}

// load_data, given a dataset CSV path, generates a vector of chemicals.
// These chemicals hold the graph representation, labels, and various other features.
fn load_data(path: &str) -> Result<Vec<smiles::Chemical>, csv::Error> {
    let mut dataset: Vec<smiles::Chemical> = Vec::new();
    let mut rdr = csv::Reader::from_path(path)?;
    for res in rdr.deserialize() {
        dataset.push(res?);
    }
    Ok(dataset)
}

fn main() {
    println!("Hello, world!");
    computation::<burn::backend::Wgpu>();

    let dataset = load_data("./data/leffingwell_data.csv").unwrap();
    println!("{:?}", dataset.len())
}
