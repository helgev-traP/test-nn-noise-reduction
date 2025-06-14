use burn::{
    nn::{
        Linear, LinearConfig,
        loss::{self, MseLoss},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::DataBatch;

// config

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub sound_block_size: usize,
    pub input_recurrent_hidden: Vec<usize>,
    pub input_recurrent_bias: bool,
    pub linear_hidden: Vec<usize>,
    pub linear_bias: bool,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_lstm_sizes = [self.sound_block_size]
            .iter()
            .chain(self.input_recurrent_hidden.iter())
            .cloned()
            .collect::<Vec<_>>();

        let input_lstm = input_lstm_sizes
            .windows(2)
            .map(|window| {
                nn::LstmConfig::new(window[0], window[1], self.input_recurrent_bias).init(device)
            })
            .collect::<Vec<_>>();

        let linear_sizes = [self.sound_block_size + self.input_recurrent_hidden.last().unwrap()]
            .iter()
            .chain(self.linear_hidden.iter())
            .chain([self.sound_block_size].iter())
            .cloned()
            .collect::<Vec<_>>();

        let linear = linear_sizes
            .windows(2)
            .map(|window| LinearConfig::new(window[0], window[1]).init(device))
            .collect::<Vec<_>>();

        Model {
            block_size: self.sound_block_size,
            input_lstm,
            linear,
        }
    }
}

// model

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    block_size: usize,
    input_lstm: Vec<nn::Lstm<B>>,
    linear: Vec<Linear<B>>,
}

impl<B: Backend> Model<B> {
    // change dimensions to match the input and output
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // reshape input to [batch_size, sound_block_len, sound_block_size]
        let dim2d = input.shape().dims;
        let input = input.split(self.block_size, 1);
        let input: Tensor<B, 3> = Tensor::<B, 2>::stack(input, 1);

        let mut lstm_output = input.clone();

        for lstm in &self.input_lstm {
            // apply lstm to each sound block
            let (new_output, _) = lstm.forward(lstm_output.clone(), None);
            // update output with lstm output
            lstm_output = new_output;
        }

        // cat the input and lstm output
        let linear_input = Tensor::<B, 3>::cat(vec![input, lstm_output], 2);

        let mut linear_output = linear_input;
        for linear_layer in &self.linear {
            // apply linear layer to each sound block
            linear_output = linear_layer.forward(linear_output);
        }

        // reshape output to [batch_size, sound_block_len, sound_block_size]
        linear_output.reshape([dim2d[0], dim2d[1]])
    }
}

// change dimensions to match the input and output
impl<B: Backend> Model<B> {
    pub fn forward_regression(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(input);
        let loss = MseLoss::new().forward(output.clone(), targets.clone(), loss::Reduction::Auto);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(batch.input, batch.target);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DataBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, batch: DataBatch<B>) -> RegressionOutput<B> {
        self.forward_regression(batch.input, batch.target)
    }
}
