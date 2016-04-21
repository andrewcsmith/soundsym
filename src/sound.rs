extern crate hound;

use vox_box::spectrum::MFCC;
use vox_box::waves::{WindowType, Windower, Filter};
use cogset::{Euclid, Kmeans, KmeansBuilder};
use voting_experts::{cast_votes, split_string};
use blas::c::*;

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::cmp::{Ordering, PartialOrd};

use super::*;

/// Calculates cosine similarity of the MFCC vectors of two Sounds. Truncates the longer Sound to
/// match the shorter Sound.
#[inline]
pub fn cosine_sim(me: &Sound, you: &Sound) -> f64 {
    let len = if me.mfccs().len() < you.mfccs().len() {
        me.mfccs().len() as i32
    } else {
        you.mfccs().len() as i32
    };
    let dot = ddot(len, &me.mfccs()[..], 1, &you.mfccs()[..], 1);
    let nrm = dnrm2(len, &me.mfccs()[..], 1) * dnrm2(len, &you.mfccs()[..], 1);
    dot / nrm
}

/// Similar to the above, but does not calculate the norm of `you`. Suitable for comparing/ranking
/// elements that have the same "you" value.
#[inline]
pub fn cosine_sim_const_you(me: &Sound, you: &Sound) -> Result<f64, CosError<'static>> {
    if me.mfccs().len() != you.mfccs().len() {
        return Err(CosError("Vectors for cosine_sim must be same length"))
    }
    let dot = ddot(me.mfccs().len() as i32, &me.mfccs()[..], 1, &you.mfccs()[..], 1);
    let nrm = dnrm2(me.mfccs().len() as i32, &me.mfccs()[..], 1);
    if nrm == 0f64 {
        Err(CosError("Norm equals zero"))
    } else {
        Ok(dot / nrm)
    }
}

/// Cache of Sounds that can be referenced using an outside sound, to find the "most similar"
/// sound.
pub struct SoundDictionary(pub Vec<Sound>);

impl SoundDictionary {
    /// Generates a SoundDictionary by walking a given path. Calculates MFCCs of all the Sounds in
    /// advance.
    pub fn from_path(path: &Path) -> Result<SoundDictionary, Box<Error>> {
        let dir = try!(path.read_dir());
        let mut out = SoundDictionary(Vec::<Sound>::with_capacity(dir.size_hint().0)); 

        for res in dir {
            let entry = try!(res);
            match entry.path().extension() {
                Some(ext) => { if ext != "wav" { continue } },
                None => { continue }
            }

            out.0.push(try!(Sound::from_path(&entry.path())));
        }
        Ok(out)
    }

    /// Finds the most similar Sound in the SoundDictionary to `other`. Requires that the two
    /// Sounds have mfccs of the same length
    pub fn match_sound<'a>(&'a mut self, other: &Sound) -> Result<&'a Sound, Box<Error>> {
        self.0.iter().min_by_key(|a| OrdF64(cosine_sim(a, &other)));
        Ok(&self.0[0])
    }
}

/// Struct holding sample information about a Sound and some pre-calculated data.
pub struct Sound {
    /// Name for reference, often the filename. Can be changed at will.
    pub name: Option<String>,
    max_power: f64,
    samples: Vec<f64>,
    mfccs: Vec<f64>
}

impl fmt::Display for Sound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}, {}, {}", &self.name.clone().unwrap_or("n/a".to_string()), &self.max_power, &self.samples.len())
    }
}

impl Sound {
    /// Generate a Sound from a Vec of f64 samples. Calculates max_power and mfccs.
    pub fn from_samples(old_samples: Vec<f64>, sample_rate: f64, name: Option<String>) -> Sound {
        let mut samples = old_samples.clone();
        samples.preemphasis(PREEMPHASIS / sample_rate);

        let sample_windows = Windower::new(WindowType::Hanning, &samples[..], HOP, HOP);

        let mfcc_calc = |frame: Vec<f64>| -> Euclid<[f64; NCOEFFS]> { 
            let mut mfccs = [0f64; NCOEFFS];
            let m = frame.mfcc(NCOEFFS, (100., 8000.), sample_rate as f64);
            for (i, c) in m.iter().enumerate() {
                mfccs[i] = *c;
            }
            Euclid(mfccs)
        };

        let power_calc = |frame: Vec<f64>| -> f64 {
            let len = frame.len() as f64;
            (frame.iter().fold(0., |acc, s| acc + s.powi(2)) / len).sqrt()
        };

        // A single vector of mfccs
        let mfccs = sample_windows.map(&mfcc_calc)
            .fold(Vec::<f64>::with_capacity(samples.len() * NCOEFFS / HOP), |mut acc, v| {
                acc.extend_from_slice(&v.0[..]);
                acc
            });

        let max_power = {
            let power_windows = Windower::new(WindowType::Rectangle, &old_samples[..], 256, 512);
            let mut powers: Vec<f64> = power_windows.map(&power_calc).collect();
            powers.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            powers.first().unwrap_or(&0f64).clone()
        };

        Sound { 
            max_power: max_power,
            name: name,
            samples: old_samples,
            mfccs: mfccs
        }
    }

    /// Generate a Sound by reading in a WAV file. Returns an Error if the file does not exist or
    /// if there is a problem reading.
    pub fn from_path(path: &Path) -> Result<Sound, Box<Error>> {
        let mut file = try!(hound::WavReader::open(&path));
        let bits_down: u32 = 32 - file.spec().bits_per_sample as u32;
        let samples: Vec<f64> = file.samples::<i32>()
            .map(|s| s.unwrap() as f64 / i32::max_value().wrapping_shr(bits_down) as f64).collect();
        let sample_rate = file.spec().sample_rate;
        let tag = path.file_stem()
            .map(|s| s.to_os_string())
            .and_then(|s| s.into_string().ok());
        Ok(Sound::from_samples(samples.clone(), sample_rate as f64, tag))
    }
    
    pub fn max_power(&self) -> f64 {
        self.max_power
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    pub fn samples<'a>(&'a self) -> &Vec<f64> {
        &self.samples
    }

    /// Gets a straight vector of MFCCs. This will be a single stream of f64 values, where each
    /// chunk of `NCOEFFS` values corresponds to a single frame.
    pub fn mfccs<'a>(&'a self) -> &Vec<f64> {
        &self.mfccs
    }

    /// Get the vector of MFCCs, broken into individual NCOEFFS-dimensional Euclid points, suitable
    /// for feeding to the `cogset` methods.
    pub fn euclid_mfccs(&self) -> Vec<Euclid<[f64; NCOEFFS]>> {
        self.mfccs.chunks(NCOEFFS)
            .fold(Vec::<Euclid<[f64; NCOEFFS]>>::with_capacity(self.mfccs.len() / NCOEFFS), |mut acc, v| {
                let mut arr = [0f64; NCOEFFS];
                for (idx, i) in v.iter().enumerate() { arr[idx] = *i; }
                acc.push(Euclid(arr));
                acc
            })
    }
}

