extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate cogset;
extern crate blas;

use vox_box::spectrum::MFCC;
use vox_box::waves::{WindowType, Windower, Filter};
use cogset::{Euclid, Kmeans, KmeansBuilder};
use voting_experts::{cast_votes, split_string};
use blas::c::*;

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::borrow::Cow;
use std::str::from_utf8;
use std::cmp::{Ordering, PartialOrd};
use std::i32;

pub const NCOEFFS: usize = 16;
pub const HOP: usize = 2048;
pub const PREEMPHASIS: f64 = 150f64;

mod sound;

pub use sound::{Sound, SoundDictionary};

/// Clumps the various Euclid points using Kmeans.
pub fn discretize(data: &[Euclid<[f64; NCOEFFS]>]) -> Kmeans<[f64; NCOEFFS]> {
    let k = 50;
    let tol = 1e-12;
    KmeansBuilder::new().tolerance(tol).kmeans(data, k)
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
        let mfccs = self.sound.euclid_mfccs();
        let clusters = discretize(&mfccs[..]).clusters();

        // Symbol to start the gibberish from
        let start_symbol = 'A' as u8;
        // Initialize memory for a u8 vector with one element per mfcc frame
        let mut byte_string = vec![0u8; mfccs.len()];
        // Look up the frame of each element in each cluster, and assign to it that cluster's label.
        for (idx, cluster) in clusters.iter().enumerate() {
            let value = idx as u8 + start_symbol;
            for &index in cluster.1.iter() {
                byte_string[index] = value as u8;
            }
        }

        let text_string = try!(from_utf8(&byte_string[..]));
        let votes = cast_votes(&text_string, self.depth);
        let splits = split_string(&text_string, &votes, self.depth, self.threshold);
        let segment_lengths = splits.into_iter().map(|s| s.len() * HOP).collect();
        Ok(segment_lengths)
    }
}

/// Takes the path of a source file, and a series of sample lengths, and splits the file
/// accordingly into a bunch of short files
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
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[test]
fn test_discretize() {
    let path = Path::new("data/sample.wav");
    let sound = Sound::from_path(path).unwrap();
    let kmeans = discretize(&sound.euclid_mfccs()[..]);
    let clusters = kmeans.clusters();
}

#[test]
fn test_sound_from_samples() {
    let mut file = hound::WavReader::open(Path::new("data/sample.wav")).unwrap();
    let mut samples: Vec<f64> = file.samples::<i32>().map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(8) as f64).collect();
    let sample_rate = file.spec().sample_rate;
    let sound = Sound::from_samples(samples.clone(), sample_rate as f64, None);
    samples.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap_or(Ordering::Equal));
    println!("max i32: {}", i16::max_value());
    // println!("max val: {}", max_val);
    println!("max_power: {}", sound.max_power());
    println!("max sample: {}", samples[0]);
    assert!((samples[0] - 0.5961925502).abs() < 1e-9);
    assert!((sound.max_power() - 0.24058003456940572).abs() < 1e-12);
}
