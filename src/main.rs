use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
};
use noise_reduction::{i_model, i_train, io_model, io_train};
use rand::rand_core::block;

const ARTIFACT_DIR: &str = "/artifact";
const TRAIN_DATA: &str = "train_data";

fn main() {
    type WgpuBackend = Wgpu<f32, i32>;
    type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

    type CpuBackend = NdArray<f32, i32>;
    type CpuAutodiffBackend = Autodiff<CpuBackend>;

    let wgpu_device = burn::backend::wgpu::WgpuDevice::default();
    let cpu_device = burn::backend::ndarray::NdArrayDevice::default();

    type AutodiffBackend = WgpuAutodiffBackend;
    let device = wgpu_device;

    // Train the input-output model
    // train_io_model::<AutodiffBackend>(ARTIFACT_DIR, TRAIN_DATA, device.clone());

    let block_size = 960;

    let i_model_config: i_model::ModelConfig = i_model::ModelConfig {
        sound_block_size: block_size,
        input_recurrent_hidden: vec![960, 640],
        input_recurrent_bias: true,
        linear_hidden: vec![1280, 960],
        linear_bias: true,
    };

    let train_config = i_train::TrainingConfig {
        model: i_model_config,
        optimizer: AdamConfig::new(),
        num_epochs: 1,
        batch_size: 32,
        num_workers: 24,
        seed: 42,
        learning_rate: 1.0e-4,
        train_test_split_ratio: 0.8,
        noise_amplitude_range_min: 0.01,
        noise_amplitude_range_max: 0.25,
        sound_block_size: block_size,
        sound_block_len: 10,
        data_dir: TRAIN_DATA.to_string(),
    };

    i_train::train::<AutodiffBackend>(ARTIFACT_DIR, train_config, device.clone());
}
