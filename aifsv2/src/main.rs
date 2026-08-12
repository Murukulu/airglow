use burn::{Tensor, backend::wgpu, prelude::*, tensor::backend::Backend};
use burn_store::{PyTorchToBurnAdapter, SafetensorsStore};

use std::path::Path;

use crate::{graph::GraphData, metadata::Metadata};

mod aifs;
mod block;
mod common;
mod decoder;
mod encoder;
mod graph;
mod metadata;
mod named_node_attributes;
mod transformer;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    inp: Tensor<B, 2>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type MyBackend = wgpu::Wgpu;
    let device: Device<MyBackend> = Default::default();
    let mut store = SafetensorsStore::from_file("../data/aifs-single-mse-2.0.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter);

    let mut model = Model {
        inp: Tensor::<MyBackend, 2>::zeros([1, 2], &device),
    };
    // model.load_from(&mut store).unwrap();
    // println!("Config: {:?}", model);

    println!("Hello, world!");

    let metadata = Metadata::load(Path::new(
        "./data/aifs-single-mse-2.0/quiet_grub/anemoi-metadata",
    ))?;
    println!(
        "{} variables, {} in / {} out, {} grid points",
        metadata.variables.len(),
        metadata.model_input.full.len(),
        metadata.model_output.full.len(),
        metadata.latitudes.len(),
    );

    // No PyTorchToBurnAdapter here: these are raw arrays, not module weights, so the
    // adapter's [out, in] -> [in, out] transpose would corrupt them.
    let mut graph_store =
        SafetensorsStore::from_file("./data/aifs-single-mse-2.0_graph.safetensors");
    let graph_data = GraphData::<MyBackend>::from_safetensors_store(&mut graph_store, &device)?;

    let [num_data, _] = graph_data.data_x.shape().dims();
    let [num_hidden, _] = graph_data.hidden_x.shape().dims();
    let [_, num_encoder_edges] = graph_data.data_to_hidden_edge_index.shape().dims();
    let [_, num_decoder_edges] = graph_data.hidden_to_data_edge_index.shape().dims();
    println!("{num_data} data nodes, {num_hidden} hidden nodes");
    println!("  data -> hidden: {num_encoder_edges} edges (encoder, CutOffEdges)");
    println!("  hidden -> data: {num_decoder_edges} edges (decoder, KNNEdges)");

    Ok(())
}
