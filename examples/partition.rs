extern crate soundsym;
extern crate voting_experts;
extern crate vox_box;
extern crate hound;

use std::path::Path;
use std::str::from_utf8;

use soundsym::*;

fn main() {
    let path = Path::new("data/long_sample.wav");
    let (path, splits) = partition(path).unwrap();
    write_splits(path, splits).unwrap();
}

