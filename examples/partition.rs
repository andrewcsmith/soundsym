extern crate soundsym;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;
extern crate getopts;

use std::path::Path;
use std::str::from_utf8;
use std::env;

use soundsym::*;
use getopts::Options;

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.reqopt("o", "", "set output directory", "OUT");
    opts.optopt("d", "depth", "depth of analysis trie", "DEPTH");
    opts.optopt("t", "threshold", "threshold for segmentation", "THRESHOLD");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!(f.to_string()) }
    };

    let out_str = matches.opt_str("o").unwrap_or_else(|| panic!("No output file specified!"));

    let out_path = Path::new(&out_str[..]);
    let path = Path::new("data/we_remember.wav");
    let (path, splits) = Partitioner::new(&path)
        .threshold(matches.opt_str("t")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(3))
        .depth(matches.opt_str("d")
               .and_then(|s| s.parse::<usize>().ok()).unwrap_or(4))
        .partition().unwrap();
    write_splits(path, &splits, &out_path).unwrap();
}

