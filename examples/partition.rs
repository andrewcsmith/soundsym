extern crate time;
extern crate soundsym;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate getopts;
extern crate rusty_machine;

use std::path::Path;
use std::borrow::Cow;
use std::env;

use soundsym::*;
use getopts::Options;
use rusty_machine::prelude::*;

/*
 * Release mode benchmarks:
 *
 * with writing files:
 * real    0m13.687s
 * user    0m12.951s
 * sys     0m0.604s
 *
 * without writing files:
 * real    0m12.968s
 * user    0m12.616s
 * sys     0m0.306s
 *
 * only loading the sound:
 * real    0m12.265s
 * user    0m11.870s
 * sys     0m0.310s
 *
 * after fixing the dumb mean_mfcc thing
 * real    0m2.208s
 * user    0m2.117s
 * sys     0m0.084s
 */

fn main() {
    let mut current_time = time::get_time();
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.optopt("o", "", "set output directory", "OUT");
    opts.optopt("d", "depth", "depth of analysis trie", "DEPTH");
    opts.optopt("t", "threshold", "threshold for segmentation", "THRESHOLD");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };

    let mut new_time = time::get_time();
    println!("time to initialize: {}", new_time - current_time);
    current_time = new_time;

    let path = Path::new("data/inventing.wav");
    let sound = Sound::from_path(path).unwrap();

    new_time = time::get_time();
    println!("time to initialize sound: {}", new_time - current_time);
    current_time = new_time;

    let mut partitioner = Partitioner::new(Cow::Owned(sound))
        .threshold(matches.opt_str("t")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(3))
        .depth(matches.opt_str("d")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(4));

    partitioner.train().unwrap();
    let cols = NCOEFFS;
    let rows = partitioner.sound.mfccs().len() / NCOEFFS;

    let data: Matrix<f64> = Matrix::new(rows, cols, partitioner.sound.mfccs().to_owned());
    let predictions = partitioner.train().unwrap();
    let splits = partitioner.partition().unwrap();

    new_time = time::get_time();
    println!("time to partition: {}", new_time - current_time);
    println!("splits: {:?}", &splits);
    println!("found {} partitions", splits.len());

    match matches.opt_str("o") {
        Some(out_str) => {
            println!("{}", &out_str);
            let out_path = Path::new(&out_str[..]);
            write_splits(&partitioner.sound, &splits[..], &out_path).unwrap();
        }
        None => { }
    }
}

