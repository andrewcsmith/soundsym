extern crate soundsym;
extern crate hound;

use std::path::Path;
use std::error::Error;
use std::iter;

use soundsym::*;
use hound::{WavWriter, WavSpec};

fn main() {
    match run() {
        Ok(_) => { },
        Err(e) => { println!("Error: {}", e); }
    }
}

fn run() -> Result<(), Box<Error>> {
    let percussion_path = Path::new("data/percussion");
    let mut dictionary = try!(SoundDictionary::from_path(&percussion_path));

    let spec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = try!(WavWriter::create("data/concat.wav", spec));

    let phoneme_path = Path::new("data/phonemes");
    for res in try!(phoneme_path.read_dir()) {
        let entry = try!(res);
        match entry.path().extension() {
            Some(ext) => { if ext != "wav" { continue } },
            None => { continue }
        }

        let phoneme = try!(Sound::from_path(&entry.path()));
        if phoneme.max_power() < 0.03 {
            for _ in 0..phoneme.samples().len() {
                try!(writer.write_sample(0));
            }
            // println!("{}, r, , ", &phoneme);
        } else {
            let sound = try!(dictionary.match_sound(&phoneme).ok_or(CosError("Problem matching sound.")));
            let samples: &Vec<f64> = sound.samples();
            for sample in samples.iter().chain(iter::repeat(&0f64)).take(phoneme.samples().len()) {
                try!(writer.write_sample((sample * i16::max_value() as f64 * 4f64.powf(phoneme.max_power())) as i16));
            }
            // println!("{}, {}", &phoneme, &sound);
        }
    }

    try!(writer.finalize());
    Ok(())
}

