use burn::{data::dataloader::batcher::Batcher, prelude::*, tensor::TensorData};

use crate::smiles;

#[derive(Clone, Default)]
pub struct ChemicalBatcher {}

#[derive(Clone, Debug)]
pub struct ChemicalBatch<B: Backend> {
    pub edges: Tensor<B, 2, Int>,
    pub node_features: Tensor<B, 2>,
    pub edge_features: Tensor<B, 2>,
    pub labels: Tensor<B, 2, Int>,
}

pub struct LeffingwellDataset {
    dataset: Vec<smiles::Chemical>,
}

impl LeffingwellDataset {
    // init, given a dataset CSV path, generates a vector of chemicals.
    // These chemicals hold the graph representation, labels, and various other features.
    pub fn init(path: &str) -> Result<LeffingwellDataset, csv::Error> {
        let mut dataset = Vec::new();
        let mut rdr = csv::Reader::from_path(path)?;
        for res in rdr.deserialize() {
            dataset.push(res?);
        }
        Ok(LeffingwellDataset { dataset })
    }
}

impl burn::data::dataset::Dataset<smiles::Chemical> for LeffingwellDataset {
    fn get(&self, index: usize) -> Option<smiles::Chemical> {
        // Clone so that you can pass ownership.
        self.dataset.get(index).map(|chem| chem.clone())
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl<B: Backend> Batcher<B, smiles::Chemical, ChemicalBatch<B>> for ChemicalBatcher {
    fn batch(&self, chemicals: Vec<smiles::Chemical>, device: &B::Device) -> ChemicalBatch<B> {
        let mut base: usize = 0;

        // Note, this should never be empty. This tracks the max length of any elements.
        // Used for padding later.
        let max_num_edges: usize = chemicals
            .iter()
            .map(|c| c.smiles.edges.len())
            .max()
            .unwrap();

        let edges = chemicals
            .iter()
            .map(|c| {
                let num_edges = c.smiles.edges.len();
                // Flatten is needed because nested vectors are not supported.
                // Unzip goes from [(x1,y1), ... ] to [[x1,...],[y1,...]].
                let (srcs, dsts): (Vec<i64>, Vec<i64>) = c
                    .smiles
                    .edges
                    .iter()
                    // TODO(saiputravu): overflows, possible?
                    .map(|&(u, v)| ((u + base) as i64, (v + base) as i64))
                    .unzip();

                // Combine outer lists by flattening.
                let flat: Vec<i64> = srcs.into_iter().chain(dsts).collect();

                // Increment the base by the number of nodes, so that we have a batch full of
                // unique node identifiers.
                base += c.smiles.node_features.len();

                // Compute result tensor, add padding.
                let mut result = Tensor::from_data(TensorData::new(flat, [2, num_edges]), device);
                if num_edges < max_num_edges {
                    let padding = Tensor::zeros([2, max_num_edges - num_edges], device);
                    result = Tensor::cat(vec![result, padding], 1);
                }
                result
            })
            .collect::<Vec<_>>();

        let node_features = chemicals
            .iter()
            .map(|c| {
                let flat: Vec<f32> = c.smiles.node_features.iter().flatten().copied().collect();
                let num_nodes = c.smiles.node_features.len();
                Tensor::from_data(
                    TensorData::new(flat, [num_nodes, smiles::ATOM_FEATURE_DIM]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        let edge_features = chemicals
            .iter()
            .map(|c| {
                let flat: Vec<f32> = c.smiles.edge_features.iter().flatten().copied().collect();
                let num_edges = c.smiles.edge_features.len();
                Tensor::from_data(
                    TensorData::new(flat, [num_edges, smiles::BOND_FEATURE_DIM]),
                    device,
                )
            })
            .collect::<Vec<_>>();

        let labels = chemicals
            .iter()
            .map(|c| {
                let flat: Vec<i64> = c.odor_labels_filtered.iter().map(|&x| x as i64).collect();
                Tensor::from_data(TensorData::new(flat, [1, smiles::ODOR_VOCAB.len()]), device)
            })
            .collect::<Vec<_>>();

        ChemicalBatch {
            edges: Tensor::cat(edges, 0),
            node_features: Tensor::cat(node_features, 0),
            edge_features: Tensor::cat(edge_features, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}
