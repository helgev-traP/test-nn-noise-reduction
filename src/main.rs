use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    optim::AdamConfig,
};
use noise_reduction::{model, train};

const MODEL_CONFIG: model::ModelConfig = model::ModelConfig {
    sound_block_size: 96,
    input_recurrent_hidden: 96,
    input_recurrent_bias: true,
    output_recurrent_hidden: 64,
    output_recurrent_bias: true,
    linear_bias: true,
    dropout: 0.1,
};

fn main() {
    type WgpuBackend = Wgpu<f32, i32>;
    type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

    type CpuBackend = NdArray<f32, i32>;
    type CpuAutodiffBackend = Autodiff<CpuBackend>;

    let wgpu_device = burn::backend::wgpu::WgpuDevice::default();
    let cpu_device = burn::backend::ndarray::NdArrayDevice::default();

    type AutodiffBackend = WgpuAutodiffBackend;
    let device = wgpu_device;

    let artifact_dir = "/artifact";

    let train_config = train::TrainingConfig {
        model: MODEL_CONFIG,
        optimizer: AdamConfig::new(),
        num_epochs: 10,
        batch_size: 24,
        num_workers: 24,
        seed: 42,
        learning_rate: 1.0e-4,
        train_test_split_ratio: 0.8,
        noise_amplitude_range_min: 0.01,
        noise_amplitude_range_max: 0.25,
        sound_block_size: MODEL_CONFIG.sound_block_size,
        sound_block_len: 1000,
        data_dir: "train_data".to_string(),
    };

    train::train::<AutodiffBackend>(artifact_dir, train_config, device.clone());
}
