use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, MultiLabelClassificationOutput,
        SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, HammingScore, LossMetric},
    },
};

use crate::{
    data::{ChemicalBatch, ChemicalBatcher, LeffingwellDataset},
    model::{Model, ModelConfig},
};

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = ChemicalBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: ChemicalBatch<B>) -> TrainOutput<MultiLabelClassificationOutput<B>> {
        let item = self.forward_classification(
            batch.edges,
            batch.node_features,
            batch.targets,
            batch.batch_idxs,
            batch.batch_size,
        );
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = ChemicalBatch<B>;
    type Output = MultiLabelClassificationOutput<B>;
    fn step(&self, batch: ChemicalBatch<B>) -> MultiLabelClassificationOutput<B> {
        self.forward_classification(
            batch.edges,
            batch.node_features,
            batch.targets,
            batch.batch_idxs,
            batch.batch_size,
        )
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,

    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary.
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(&device, config.seed);

    let dataloader = DataLoaderBuilder::<B, _, _>::new(ChemicalBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(LeffingwellDataset::train("./data/leffingwell_data.csv").unwrap());

    let dataloader_test =
        DataLoaderBuilder::<B::InnerBackend, _, _>::new(ChemicalBatcher::default())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .build(LeffingwellDataset::test("./data/leffingwell_data.csv").unwrap());

    let training = SupervisedTraining::new(artifact_dir, dataloader, dataloader_test)
        .metrics((HammingScore::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully")
}
