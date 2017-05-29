extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate rusty_machine;
extern crate blas;
extern crate sample;
extern crate rayon;

use voting_experts::{cast_votes, split_string};
use rusty_machine::prelude::*;
use rusty_machine::learning::k_means::{KPlusPlus, KMeansClassifier};
use rusty_machine::data::transforms::{Standardizer, Transformer};

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::borrow::Cow;
use std::str::from_utf8;
use std::cmp::PartialOrd;
use std::i32;

use blas::c::*;

pub const NCOEFFS: usize = 14;
pub const NCLUSTERS: usize = 30;
pub const HOP: usize = 256;
pub const BIN: usize = 4096;
pub const PREEMPHASIS: f64 = 75f64;

mod sound;
pub use sound::{Sound, SoundDictionary, SoundSequence, Timestamp, audacity_labels_to_timestamps};

/// Partitions a sound file (from a path) into individual phonemes. It is possible to set the
/// depth of the trie and the threshold of votes needed to draw a boundary line. 
///
/// Uses the voting experts algorithm on a vector of discretized MFCC vectors. Currently tuned for
/// English language phonemes, but alternative settings could adapt for other languages or sources.
pub struct Partitioner<'a> {
    pub sound: Cow<'a, Sound>,
    pub depth: usize,
    pub threshold: usize,
    pub model: KMeansClassifier<KPlusPlus>,
}

impl<'a> Partitioner<'a> {
    pub fn new(sound: Cow<'a, Sound>) -> Self {
        let mut model = KMeansClassifier::new(NCLUSTERS);
        model.set_iters(50);
        Partitioner {
            sound: sound,
            depth: 5,
            threshold: 4,
            model: model,
        }
    }

    pub fn from_path(path: &'a Path) -> Result<Self, Box<Error>> {
        let sound = try!(Sound::from_path(path));
        Ok(Partitioner::new(Cow::Owned(sound)))
    }

    /// Builder method to set the depth of the trie.
    pub fn depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Builder method to set the minimum number of votes required to execute a split.
    pub fn threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn train(&mut self) -> Result<(), Box<Error>> {
        let cols = NCOEFFS;
        let rows = self.sound.mfccs().len() / NCOEFFS;
        let data: Matrix<f64> = Matrix::new(rows, cols, self.sound.mfccs().to_owned());
        // println!("##DATA \n{}", &data);
        let mut transformer = Standardizer::default();
        let data = transformer.transform(data.clone()).unwrap();

        // Retry forever.
        while let Err(e) = self.model.train(&data) {
            println!("Error: {}", e);
            // Use the following if training a GaussianMixtureModel
            // model.cov_option.reg_covar += 0.01;
            // println!("New reg_covar is {}", model.cov_option.reg_covar);
        }

        Ok(())
    }

    pub fn predict(&self, data: &Matrix<f64>) -> Result<Matrix<f64>, Box<Error>> {
        let cols = self.model.predict(&data).unwrap();
        let mut out = Matrix::zeros(data.rows(), NCLUSTERS);
        for (row, col) in out.iter_rows_mut().zip(cols.iter()) {
            row[*col] = 1.;
        }
        Ok(out)
    }

    /// Executes the partition. On success, returns a tuple containing the path of the file
    /// partitioned and a Vec of sample indices where each index corresponds to the beginning of
    /// the phoneme.
    pub fn partition(&self, predictions: Matrix<f64>) -> Result<Vec<usize>, Box<Error>> {
        // println!("##PREDICTIONS \n{}", &predictions);
        // Symbol to start the gibberish from
        let start_symbol = 'A' as u8;
        // Initialize memory for a u8 vector with one element per mfcc frame
        let mut byte_string = vec![0u8; predictions.rows()];
        // Look up the frame of each element in each cluster, and assign to it that cluster's label.
        // row: &[f64]
        for (idx, row) in predictions.iter_rows().enumerate() {
            let max_idx: u8 = idamax(row.len() as i32, row, 1) as u8;
            byte_string[idx] = max_idx + start_symbol;
        }

        let text_string = try!(from_utf8(&byte_string[..]));
        println!("{}", &text_string);
        let votes = cast_votes(&text_string, self.depth);
        let splits = split_string(&text_string, &votes, self.depth, self.threshold);
        let segment_lengths = splits.into_iter().map(|s| s.len() * HOP).collect();
        Ok(segment_lengths)
    }
}

/// Takes a `Sound`, and a series of sample lengths, and splits the file accordingly into a bunch
/// of short files, writing these splits to the disk.
pub fn write_splits(sound: &Sound, splits: &[usize], out_path: &Path) -> Result<(), Box<Error>> {
    let sample_rate = sound.sample_rate() as u32;
    let mut samples = sound.samples().iter();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Int,
    };

    // Works through the samples in order, writing files to disk
    for (idx, split) in splits.iter().enumerate() {
        let amplitude = i32::MAX as f64;
        let mut writer = try!(hound::WavWriter::create(
            format!("{}/{:05}_{}.wav", out_path.to_str().unwrap(), idx, split), spec));
        for sample in samples.by_ref().take(*split) {
            try!(writer.write_sample((sample * amplitude) as i32));
        }
        try!(writer.finalize());
    }

    Ok(())
}

/// Convenience Error type.
#[derive(Debug)]
pub struct CosError<'a>(pub &'a str);

impl<'a> Error for CosError<'a> {
    fn description(&self) -> &str {
        self.0
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl<'a> fmt::Display for CosError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Super-hack type containing f64 to be used in Ord
#[derive(PartialEq, PartialOrd)]
pub struct OrdF64(pub f64);

impl std::cmp::Eq for OrdF64 {}

impl std::cmp::Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
    extern crate hound;

    use super::*;
    use std::path::Path;
    use std::cmp::Ordering;

    #[test]
    fn test_sound_from_samples() {
        let mut file = hound::WavReader::open(Path::new("data/sample.wav")).unwrap();
        let mut samples: Vec<f64> = file.samples::<i32>().map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(8) as f64).collect();
        let sample_rate = file.spec().sample_rate;
        let sound = Sound::from_samples(samples.clone(), sample_rate as f64, None, None);
        samples.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(Ordering::Equal));
        println!("max i32: {}", i16::max_value());
        // println!("max val: {}", max_val);
        println!("max_power: {}", sound.max_power());
        println!("max sample: {}", samples[0]);
        assert!((samples[0] - 0.5961925502).abs() < 1e-9);
        assert!((sound.max_power() - 0.18730848034829456).abs() < 1e-12);
    }
}
