use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

use crate::{data::DataBatcher, io_model::ModelConfig};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,

    #[config(default = 0.8)]
    pub train_test_split_ratio: f32,

    #[config(default = 0.01)]
    pub noise_amplitude_range_min: f32,
    #[config(default = 0.25)]
    pub noise_amplitude_range_max: f32,
    #[config(default = 96)]
    pub sound_block_size: usize,
    #[config(default = 1000)]
    pub sound_block_len: usize,
    pub data_dir: String,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    println!("Loading training data from: {}", config.data_dir);
    let dataset = crate::data::SoundDataset::new(
        [
            config.noise_amplitude_range_min,
            config.noise_amplitude_range_max,
        ],
        config.sound_block_len,
        config.sound_block_size,
        "train_data",
        config.seed,
    )
    .unwrap();

    println!(
        "Splitting dataset into train and test sets with ratio: {}",
        config.train_test_split_ratio
    );
    let (dataset_train, dataset_test) = dataset.split_by_ratio(config.train_test_split_ratio);

    let batcher = DataBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
