extern crate soundsym;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate getopts;

use std::path::Path;
use std::env;
use std::cmp::Ordering;

use soundsym::*;
use getopts::Options;

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("d", "depth", "depth of analysis trie", "DEPTH");
    opts.optopt("t", "threshold", "threshold for segmentation", "THRESHOLD");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };

    let path = Path::new("data/we_remember_mono.wav");
    let splits = Partitioner::from_path(&path).unwrap()
        .threshold(matches.opt_str("t")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(3))
        .depth(matches.opt_str("d")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(4))
        .partition().unwrap();

    let mut input = hound::WavReader::open(&path).unwrap();
    let spec = input.spec();
    let sample_rate = input.spec().sample_rate;
    let bits_down: u32 = 32 - spec.bits_per_sample as u32;
    let mut samples = input.samples::<i32>()
        .map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(bits_down) as f64);

    let mut sounds: Vec<Sound> = splits.iter().map(|split| {
        let sound_samples: Vec<f64> = samples.by_ref().take(*split).collect();
        Sound::from_samples(sound_samples, sample_rate as f64, None, None)
    }).collect();

    sounds.sort_by(|a, b| a.max_power().partial_cmp(&b.max_power()).unwrap_or(Ordering::Equal));

    let mut output = hound::WavWriter::create(&Path::new("output.wav"), spec).unwrap();
    for sound in sounds {
        println!("sound: {}", sound);
        for sample in sound.samples() {
            output.write_sample((sample * i32::max_value().wrapping_shr(bits_down) as f64) as i32);
        }
    }
    output.finalize();
}

