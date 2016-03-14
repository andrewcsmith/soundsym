extern crate soundsym;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;

use std::path::Path;
use std::str::from_utf8;

use soundsym::*;
use voting_experts::{cast_votes, split_string};

fn main() {
    let path = Path::new("data/sample.wav");
    let mfccs = calc_mfccs(path);
    let clusters = discretize(&mfccs[..]).clusters();
    
    let mut file = hound::WavReader::open(path).unwrap();
    let sample_rate = file.spec().sample_rate;
    let mut samples = file.samples::<i32>().map(|s| s.unwrap());

    let start_symbol = "A".as_bytes()[0];
    println!("starting at {}", start_symbol);

    let mut byte_string = vec![0u8; mfccs.len()];

    for (idx, cluster) in clusters.iter().enumerate() {
        let value = idx as u8 + start_symbol;
        for &index in cluster.1.iter() {
            byte_string[index] = value as u8;
        }
    }

    let text_string = from_utf8(&byte_string[..]).unwrap();

    println!("sound file aka {}", &text_string);

    let votes = cast_votes(&text_string, 4);
    let splits = split_string(&text_string, &votes, 4, 3);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate,
        bits_per_sample: 24
    };

    for (idx, split) in splits.iter().enumerate() {
        println!("{}", split);
        let mut writer = hound::WavWriter::create(format!("data/{:02}_{}.wav", idx, split), spec).unwrap();
        for _ in 0..(split.len() * soundsym::HOP) {
            writer.write_sample(samples.next().unwrap()).unwrap();
        }
        writer.finalize().unwrap();
    }
}

