extern crate soundsym;

use std::path::Path;
use std::error::Error;

use soundsym::*;

fn main() {
    match run() {
        Ok(_) => { },
        Err(e) => { println!("Error: {}", e); }
    }
}

fn run() -> Result<(), Box<Error>> {
    let phoneme_path = Path::new("data/phoneme.wav");
    let percussion_path = Path::new("data/percussion");
    let mut dictionary = try!(SoundDictionary::from_path(&percussion_path));
    let phoneme = try!(Sound::from_path(&phoneme_path));
    let sound = try!(dictionary.match_sound(&phoneme));
    println!("Found sound: {}", &sound);
    Ok(())
}

