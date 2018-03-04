extern crate soundsym;
extern crate rusty_machine;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate getopts;

use std::path::Path;
use std::env;
use std::cmp::Ordering;

use soundsym::*;
use getopts::Options;

use rusty_machine::prelude::*;


/// Arranges the phonemes in a sound file in order of increasing loudness.
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.optopt("d", "depth", "depth of analysis trie", "DEPTH");
    opts.optopt("t", "threshold", "threshold for segmentation", "THRESHOLD");
    opts.reqopt("s", "sound", "path to input sound file", "SOUND");
    opts.reqopt("o", "output", "path to output sound file", "OUTPUT");

    let matches = opts.parse(&args[1..]).unwrap_or_else(|f| panic!(f.to_string()));

    let input_file = matches.opt_str("s").unwrap();
    let output_path = matches.opt_str("o").unwrap();

    let path = Path::new(&input_file);

    let mut partitioner = Partitioner::from_path(&path).unwrap()
        .threshold(matches.opt_str("t")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(3))
        .depth(matches.opt_str("d")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(4));

    partitioner.train().unwrap();
    let splits = partitioner.partition().unwrap();

    let mut input = hound::WavReader::open(&path).expect("Could not open input file");
    let spec = input.spec();
    let sample_rate = input.spec().sample_rate;
    let bits_down: u32 = 32 - spec.bits_per_sample as u32;
    let mut samples = input.samples::<i32>()
        .map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(bits_down) as f64);

    let mut sounds: Vec<Sound> = splits.iter().map(|split| {
        let sound_samples: Vec<f64> = samples.by_ref().take(*split).collect();
        Sound::from_samples(sound_samples, sample_rate as f64, None, None)
    }).collect();

    sounds.sort_by(|a, b| a.max_power().abs().partial_cmp(&b.max_power().abs()).unwrap_or(Ordering::Less));

    let mut output = hound::WavWriter::create(&Path::new(&output_path), spec).unwrap();
    for sound in sounds {
        println!("sound: {}", sound);
        for sample in sound.samples() {
            output.write_sample((sample * i32::max_value().wrapping_shr(bits_down) as f64) as i32)
                .expect("Could not write sample");
        }
    }
    output.finalize().expect("Could not finalize file");
}

