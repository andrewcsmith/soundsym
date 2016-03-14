extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate cogset;

use vox_box::spectrum::MFCC;
use vox_box::waves::{WindowType, Windower, Filter};
use cogset::{Euclid, Kmeans, KmeansBuilder};

use std::path::Path;

pub const NCOEFFS: usize = 12;
pub const HOP: usize = 2048;

pub fn calc_mfccs(path: &Path) -> Vec<Euclid<[f64; NCOEFFS]>> {
    let mut file = hound::WavReader::open(path).unwrap();
    let sample_rate = file.spec().sample_rate;
    let mut samples: Vec<f64> = file.samples::<i32>().map(|s| s.unwrap() as f64).collect();
    samples.preemphasis(50f64 / sample_rate as f64);

    let sample_windows = Windower::new(WindowType::Hanning, &samples[..], HOP, HOP);
    let mfcc_calc = |frame: Vec<f64>| -> Euclid<[f64; NCOEFFS]> { 
        let mut mfccs = [0f64; NCOEFFS];
        let m = frame.mfcc(NCOEFFS, (100., 8000.), sample_rate as f64);
        for (i, c) in m.iter().enumerate() {
            mfccs[i] = *c;
        }
        Euclid(mfccs)
    };
    sample_windows.map(&mfcc_calc).collect()
}

pub fn discretize(data: &[Euclid<[f64; NCOEFFS]>]) -> Kmeans<[f64; NCOEFFS]> {
    let k = 50;
    let tol = 1e-10;
    KmeansBuilder::new().tolerance(tol).kmeans(data, k)
}

#[test]
fn test_discretize() {
    let path = Path::new("data/sample.wav");
    let mfccs = calc_mfccs(path);
    let kmeans = discretize(&mfccs[..]);
    let clusters = kmeans.clusters();

    let mut file = hound::WavReader::open(path).unwrap();
    let sample_rate = file.spec().sample_rate;
    let samples: Vec<f32> = file.samples::<i32>().map(|s| s.unwrap() as f32).collect();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 16
    };

    for (idx, cluster) in clusters.iter().enumerate() {
        let mut writer = hound::WavWriter::create(format!("data/output_{}.wav", idx), spec).unwrap();
        let mut last_sample = 0f32;
        for index in cluster.1.iter() {
            let offset = index * HOP;
            for sample in &samples[offset..(offset+HOP)] {
                writer.write_sample(((*sample + last_sample) / 2f32) as i16).unwrap();
                last_sample = *sample as f32;
            }
        }
        writer.finalize().unwrap();
    }
}

