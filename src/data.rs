use std::path::PathBuf;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    train,
};
use rand::{Rng, SeedableRng, seq::SliceRandom};

pub struct SoundDataset {
    seed: u64,
    noise_amplitude_range: [f32; 2],

    block_len: usize,
    songs: Vec<Vec<f32>>,
    noise: Vec<Vec<f32>>,
}

impl SoundDataset {
    pub fn new(
        noise_amplitude_range: [f32; 2],
        training_block_len: usize,
        rnn_block_size: usize,
        data_dir: &str,
        seed: u64,
    ) -> Result<Self, String> {
        // file compression:
        // - data_dir
        //   - songs
        //     - *.wav
        //     - *.wav
        //     - ...
        //   - noise
        //     - *.wav
        //     - *.wav
        //     - ...

        // randomly select noise and clean songs and make a pair of original and noisy sound

        let songs_dir = PathBuf::from(data_dir).join("songs");
        let noise_dir = PathBuf::from(data_dir).join("noise");

        println!("Loading songs from: {}", songs_dir.display());

        let mut songs = std::fs::read_dir(songs_dir)
            .expect("Failed to read songs directory")
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let song = hound::WavReader::open(&path)
                    .ok()?
                    .into_samples::<f32>()
                    .filter_map(Result::ok)
                    .collect::<Vec<_>>();

                if song.len() < rnn_block_size {
                    eprintln!(
                        "Warning: {} is too short: {} samples",
                        path.display(),
                        song.len()
                    );
                    return None;
                }

                // split song into blocks of size `training_block_size`
                let blocks = song
                    .chunks(training_block_len * rnn_block_size)
                    .map(|block| block.to_vec())
                    .filter(|block| block.len() == training_block_len * rnn_block_size)
                    .collect::<Vec<_>>();

                Some(blocks)
            })
            .flatten()
            .collect::<Vec<_>>();

        println!("Loading noise from: {}", noise_dir.display());

        let mut noise = std::fs::read_dir(noise_dir)
            .expect("Failed to read noise directory")
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let noise = hound::WavReader::open(&path)
                    .ok()?
                    .into_samples::<f32>()
                    .filter_map(Result::ok)
                    .collect::<Vec<_>>();

                // split noise into blocks of size `rnn_block_size`
                let blocks = noise
                    .chunks(training_block_len * rnn_block_size)
                    .map(|block| block.to_vec())
                    .filter(|block| block.len() == training_block_len * rnn_block_size)
                    .collect::<Vec<_>>();

                Some(blocks)
            })
            .flatten()
            .collect::<Vec<_>>();

        println!(
            "Found {} songs and {} noise files",
            songs.len(),
            noise.len()
        );

        // sort songs and noise randomly
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        songs.shuffle(&mut rng);
        noise.shuffle(&mut rng);

        // ensure that we have at least one song and one noise
        if songs.is_empty() || noise.is_empty() {
            return Err("No songs or noise found in the specified directories".to_string());
        }

        Ok(Self {
            seed,
            noise_amplitude_range,
            block_len: training_block_len,
            songs,
            noise,
        })
    }

    pub fn split_by_ratio(self, ratio: f32) -> (Self, Self) {
        let split_index = (self.songs.len() as f32 * ratio) as usize;
        let (train_songs, test_songs) = self.songs.split_at(split_index);

        let split_index = (self.noise.len() as f32 * ratio) as usize;
        let (train_noise, test_noise) = self.noise.split_at(split_index);

        (
            SoundDataset {
                seed: self.seed,
                noise_amplitude_range: self.noise_amplitude_range,
                block_len: self.block_len,
                songs: train_songs.to_vec(),
                noise: train_noise.to_vec(),
            },
            SoundDataset {
                seed: self.seed,
                noise_amplitude_range: self.noise_amplitude_range,
                block_len: self.block_len,
                songs: test_songs.to_vec(),
                noise: test_noise.to_vec(),
            },
        )
    }
}

impl Dataset<NoisePair> for SoundDataset {
    fn get(&self, index: usize) -> Option<NoisePair> {
        let song_index = index / self.noise.len();
        let noise_index = index % self.noise.len();

        if song_index >= self.songs.len() || noise_index >= self.noise.len() {
            return None;
        }

        let song = &self.songs[song_index];
        let noise = &self.noise[noise_index];

        // crate a noisy version of the song
        let noisy_song: Vec<f32> = song
            .iter()
            .zip(noise.iter())
            .map(|(sample, &noise_sample)| {
                sample
                    + noise_sample
                        * rand::rngs::StdRng::seed_from_u64(index as u64 + self.seed).random_range(
                            self.noise_amplitude_range[0]..=self.noise_amplitude_range[1],
                        )
            })
            .collect();

        Some(NoisePair {
            clean: song.clone(),
            noisy: noisy_song,
        })
    }

    fn len(&self) -> usize {
        self.songs.len() * self.noise.len()
    }
}

#[derive(Clone, Debug)]
pub struct NoisePair {
    pub clean: Vec<f32>,
    pub noisy: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct DataBatcher {}

#[derive(Clone, Debug)]
pub struct DataBatch<B: Backend> {
    pub input: Tensor<B, 2>,
    pub target: Tensor<B, 2>,
}

impl<B: Backend> Batcher<B, NoisePair, DataBatch<B>> for DataBatcher {
    fn batch(&self, items: Vec<NoisePair>, device: &<B as Backend>::Device) -> DataBatch<B> {
        let items = items
            .iter()
            .map(|item| {
                (
                    Tensor::<B, 1>::from_data(&*item.clean, device),
                    Tensor::<B, 1>::from_data(&*item.noisy, device),
                )
            })
            .collect::<Vec<_>>();

        let (inputs, targets): (Vec<_>, Vec<_>) = items.into_iter().unzip();

        DataBatch {
            input: Tensor::stack(inputs, 0),
            target: Tensor::stack(targets, 0),
        }
    }
}
