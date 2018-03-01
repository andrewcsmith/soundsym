extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate rusty_machine;
extern crate arrayfire;
extern crate sample;

use voting_experts::{cast_votes, split_string};

use rusty_machine::prelude::*;
use rusty_machine::learning::gmm::{CovOption, GaussianMixtureModel};
use rusty_machine::learning::UnSupModel;
use rusty_machine::data::transforms::{Transformer, Standardizer};

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::borrow::Cow;
use std::str::from_utf8;
use std::cmp::PartialOrd;
use std::i32;

pub const NCOEFFS: usize = 12;
pub const NCLUSTERS: usize = 20;
pub const HOP: usize = 512;
pub const BIN: usize = 2048;
pub const PREEMPHASIS: f64 = 150f64;

mod sound;
use sound::max_index;
pub use sound::{Sound, SoundDictionary, SoundSequence, Timestamp, audacity_labels_to_timestamps};

pub fn discretize(data: &Matrix<f64>) -> Matrix<f64> {
    let mut gmm = GaussianMixtureModel::new(NCLUSTERS);
    gmm.cov_option = CovOption::Regularized(0.1);
    let mut transformer = Standardizer::default();
    let transformed = transformer.transform(data.clone()).unwrap();
    gmm.set_max_iters(1000);
    while let Err(err) = gmm.train(&transformed) { 
        println!("Encountered an error in training, retrying: {}", &err.description());
    }
    gmm.predict(&transformed).unwrap()
}

/// Partitions a sound file (from a path) into individual phonemes. It is possible to set the
/// depth of the trie and the threshold of votes needed to draw a boundary line. 
///
/// Uses the voting experts algorithm on a vector of discretized MFCC vectors. Currently tuned for
/// English language phonemes, but alternative settings could adapt for other languages or sources.
pub struct Partitioner<'a> {
    pub sound: Cow<'a, Sound>,
    pub depth: usize,
    pub threshold: usize
}

impl<'a> Partitioner<'a> {
    pub fn new(sound: Cow<'a, Sound>) -> Self {
        Partitioner {
            sound: sound,
            depth: 5,
            threshold: 4
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

    /// Executes the partition. On success, returns a tuple containing the path of the file
    /// partitioned and a Vec of sample indices where each index corresponds to the beginning of
    /// the phoneme.
    pub fn partition(&self) -> Result<Vec<usize>, Box<Error>> {
        let cols = NCOEFFS;
        let rows = self.sound.mfccs().len() / NCOEFFS;
        let data: Matrix<f64> = Matrix::new(rows, cols, self.sound.mfccs().to_owned());
        // println!("##DATA \n{}", &data);
        let predictions = discretize(&data);

        // println!("##PREDICTIONS \n{}", &predictions);
        // Symbol to start the gibberish from
        let start_symbol = 'A' as u8;
        // Initialize memory for a u8 vector with one element per mfcc frame
        let mut byte_string = vec![0u8; rows];
        // Look up the frame of each element in each cluster, and assign to it that cluster's label.
        // row: &[f64]
        for (idx, row) in predictions.iter_rows().enumerate() {
            let max_idx: u8 = max_index(&row[..]) as u8;
            byte_string[idx] = max_idx + start_symbol;
        }

        let text_string = try!(from_utf8(&byte_string[..]));
        // println!("{}", &text_string);
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
        bits_per_sample: 32
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

    use rusty_machine::prelude::*;
    use super::*;
    use std::path::Path;
    use std::cmp::Ordering;

    #[test]
    fn test_discretize() {
        let path = Path::new("tests/sample.wav");
        let sound = Sound::from_path(path).unwrap();
        let cols = NCOEFFS;
        let rows = sound.mfccs().len() / NCOEFFS;
        let data: Matrix<f64> = Matrix::new(rows, cols, sound.mfccs().to_owned());
        // println!("data: \n{}", &data);
        let predictions = discretize(&data);
        println!("predictions: \n{:?}", predictions);
        assert_eq!(predictions.rows(), sound.mfcc_arrays().len());
        assert_eq!(predictions.cols(), NCLUSTERS);
        assert!(!predictions[[0, 0]].is_nan())
    }

    #[test]
    fn test_partitioner() {
        let path = Path::new("tests/sample.wav");
        let partitioner = Partitioner::from_path(path).unwrap();
        let splits = partitioner.partition().unwrap();
        println!("splits: {:?}", &splits);
        assert!(splits.len() > 0);
        let out_path = Path::new("tmp");
        write_splits(&partitioner.sound, &splits, &out_path);
    }

    #[test]
    fn test_sound_from_samples() {
        let mut file = hound::WavReader::open(Path::new("tests/sample.wav")).unwrap();
        let mut samples: Vec<f64> = file.samples::<i32>().map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(8) as f64).collect();
        let sample_rate = file.spec().sample_rate;
        let sound = Sound::from_samples(samples.clone(), sample_rate as f64, None, None);
        samples.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(Ordering::Equal));
        // println!("max i32: {}", i16::max_value());
        // println!("max val: {}", max_val);
        // println!("max_power: {}", sound.max_power());
        // println!("max sample: {}", samples[0]);
        // println!("max power: {}", sound.max_power());
        assert!((samples[0] - 0.6503654301602161).abs() < 1e-9);
        assert!((sound.max_power() - 0.25781895526454907).abs() < 1e-12);
    }
}
