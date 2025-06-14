use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig,
        loss::{self, MseLoss},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::DataBatch;

/*
Input:
Tensor<B, 3>
[batch_size, sound_block_len, sound_block_size]

Output:
Tensor<B, 3>
[batch_size, sound_block_len, sound_block_size]

*/

// config

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub sound_block_size: usize,
    pub input_recurrent_hidden: usize,
    pub input_recurrent_bias: bool,
    pub output_recurrent_hidden: usize,
    pub output_recurrent_bias: bool,
    pub linear_bias: bool,
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let input_recurrent = nn::LstmConfig::new(
            self.sound_block_size,
            self.input_recurrent_hidden,
            self.input_recurrent_bias,
        )
        .init(device);

        let output_recurrent = nn::LstmConfig::new(
            self.sound_block_size,
            self.output_recurrent_hidden,
            self.output_recurrent_bias,
        )
        .init(device);

        let linear = LinearConfig::new(
            self.sound_block_size + self.input_recurrent_hidden + self.output_recurrent_hidden,
            self.sound_block_size,
        )
        .init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        Model {
            block_size: self.sound_block_size,
            input_lstm: input_recurrent,
            input_lstm_hidden_size: self.input_recurrent_hidden,
            output_lstm: output_recurrent,
            output_lstm_hidden_size: self.output_recurrent_hidden,
            linear,
            dropout,
        }
    }
}

// model

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    block_size: usize,
    input_lstm: nn::Lstm<B>,
    input_lstm_hidden_size: usize,
    output_lstm: nn::Lstm<B>,
    output_lstm_hidden_size: usize,
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Model<B> {
    // change dimensions to match the input and output
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = input.device();

        // reshape input to [batch_size, sound_block_len, sound_block_size]
        let batch_size = input.shape().dims[0];
        let blocks = input.split(self.block_size, 1);
        let sound_block_len = blocks.len();

        // input LSTM data
        let mut input_lstm_state = Some(nn::LstmState::new(
            Tensor::<B, 2>::zeros([batch_size, self.input_lstm_hidden_size], &device),
            Tensor::<B, 2>::zeros([batch_size, self.input_lstm_hidden_size], &device),
        ));

        // output LSTM data
        let mut output_lstm_state = Some(nn::LstmState::new(
            Tensor::<B, 2>::zeros([batch_size, self.output_lstm_hidden_size], &device),
            Tensor::<B, 2>::zeros([batch_size, self.output_lstm_hidden_size], &device),
        ));

        // output tensor [sound_block_len][batch_size, 1, sound_block_size]
        // cat by the second dimension later
        let mut output = Vec::with_capacity(sound_block_len);

        for input_block in blocks {
            // make output tensor
            let linear_input = Tensor::<B, 2>::cat(
                vec![
                    input_block.clone(),
                    input_lstm_state.as_ref().unwrap().hidden.clone(),
                    output_lstm_state.as_ref().unwrap().hidden.clone(),
                ],
                1,
            );
            let linear_output = self.linear.forward(linear_input);
            output.push(linear_output.clone());

            // input LSTM
            let input_block_3d = input_block.reshape([batch_size, 1, self.block_size]);
            let (_, new_input_lstm_state) = self
                .input_lstm
                .forward(input_block_3d, input_lstm_state.take());
            input_lstm_state = Some(new_input_lstm_state);

            // output LSTM
            let linear_output_3d = linear_output.reshape([batch_size, 1, self.block_size]);
            let (_, new_output_lstm_state) = self
                .output_lstm
                .forward(linear_output_3d, output_lstm_state.take());
            output_lstm_state = Some(new_output_lstm_state);
        }

        // cat output tensors by the second dimension
        Tensor::<B, 2>::cat(output, 1)
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
