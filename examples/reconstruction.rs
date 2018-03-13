/// This is an "offline" version of reconstruction, meant to demonstrate what
/// could conceivably be the final result.

extern crate soundsym;
extern crate getopts;

use std::error::Error;
use std::sync::Arc;
use std::path::Path;
use std::borrow::Cow;
use std::env;

use getopts::Options;

use soundsym::*;

fn main() {
    match go() {
        Ok(_) => { }
        Err(e) => { println!("Error: {}\nBacktrace: {:?}", e.description(), e.cause()); std::process::exit(1); }
    }
}

fn go() -> Result<(), Box<Error>> {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.reqopt("s", "", "set source file", "SOURCE");
    opts.optopt("d", "", "set dictionary path", "DICTIONARY");
    opts.reqopt("t", "", "set target file", "TARGET");
    opts.reqopt("o", "", "set output path", "OUT");
    opts.optopt("f", "", "set output folder", "FOLDER");

    let matches = opts.parse(&args[1..])?;

    let input_file = matches.opt_str("s").unwrap();
    let target_file = matches.opt_str("t").unwrap();
    let output_file = matches.opt_str("o").unwrap();
    let dictionary_path = matches.opt_str("d");
    let output_folder = matches.opt_str("f");

    let input_path = Path::new(&input_file);
    println!("Loading partitioner");
    let mut partitioner = Partitioner::from_path(&input_path)?
        .threshold(4)
        .depth(3);

    println!("Loaded partitioner");
    partitioner.train()?;
    println!("Trained partitioner");
    let splits = partitioner.partition()?;

    let dictionary = match dictionary_path {
        Some(ref path) => {
            SoundDictionary::from_path(&Path::new(path))?
        }
        None => {
            SoundDictionary::from_segments(&partitioner.sound, &splits[..])
        }
    };

    match output_folder {
        Some(folder) => {
            let path = Path::new(&folder);
            if path.exists() {
                write_splits(&partitioner.sound, &splits[..], &path)?;
            }
        }
        None => { }
    }

    let target_sound = Sound::from_path(&Path::new(&target_file))?;
    partitioner.sound = Cow::Owned(target_sound);
    let mut samples = partitioner.sound.samples().iter();
    let mut mfccs = partitioner.sound.mfccs().iter();
    let sample_rate = partitioner.sound.sample_rate();

    let target_segments = partitioner.partition()?.iter().map(|split| {
        let samp: Vec<f64> = samples.by_ref().take(*split).map(|s| *s).collect();
        let mfccs: Vec<f64> = mfccs.by_ref().take(*split / HOP * NCOEFFS).map(|s| *s).collect();
        Arc::new(Sound::from_samples(samp, sample_rate, Some(mfccs), None))
    }).collect();

    let sequence = SoundSequence::new(target_segments);

    let out_sound = sequence.clone_from_dictionary(&dictionary)?;
    out_sound.to_sound().write_file(Path::new(&output_file))?;

    Ok(())
}
