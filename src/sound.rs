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
use std::{f64, i32};

use super::*;

/// Calculates cosine similarity of the MFCC vectors of two Sounds. Truncates the longer Sound to
/// match the shorter Sound.
#[inline]
pub fn cosine_sim(me: &[f64], you: &[f64]) -> f64 {
    let len = if me.len() < you.len() {
        me.len() as i32
    } else {
        you.len() as i32
    };
    let dot = ddot(len, me, 1, you, 1);
    let nrm = dnrm2(len, me, 1) * dnrm2(len, you, 1);
    dot / nrm
}

/// Similar to the above, but does not calculate the norm of `you`. Suitable for comparing/ranking
/// elements that have the same "you" value.
#[inline]
pub fn cosine_sim_const_you(me: &[f64], you: &[f64]) -> Result<f64, CosError<'static>> {
    if me.len() != you.len() {
        return Err(CosError("Vectors for cosine_sim must be same length"))
    }
    let dot = ddot(me.len() as i32, &me, 1, &you, 1);
    let nrm = dnrm2(me.len() as i32, &me, 1);
    if nrm == 0f64 {
        Err(CosError("Norm equals zero"))
    } else {
        Ok(dot / nrm)
    }
}

/// Angular cosine similarity. This bounds the result to `[0, 1]` while maintaining ordering. It is
/// also a proper metric.
pub fn cosine_sim_angular(me: &[f64], you: &[f64]) -> f64 {
    let cos_sim = cosine_sim(me, you);
    1. - (cos_sim.acos() / f64::consts::PI)
}

/// Struct holding sample information about a Sound and some pre-calculated data.
#[derive(Debug, Clone)]
pub struct Sound {
    /// Name for reference, often the filename. Can be changed at will.
    pub name: Option<String>,
    max_power: f64,
    samples: Vec<f64>,
    sample_rate: f64,
    mfccs: Vec<f64>,
    mean_mfccs: Euclid<[f64; NCOEFFS]>
}

impl fmt::Display for Sound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}, {}, {}", &self.name.clone().unwrap_or("n/a".to_string()), &self.max_power, &self.samples.len())
    }
}

impl Sound {
    /// Generate a Sound from a Vec of f64 samples. Calculates max_power and mfccs.
    pub fn from_samples(samples: Vec<f64>, sample_rate: f64, name: Option<String>) -> Sound {
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
        let mfccs = Windower::new(WindowType::Hanning, &samples[..], HOP, HOP)
            .map(&mfcc_calc)
            .fold(Vec::<f64>::with_capacity(samples.len() * NCOEFFS / HOP), |mut acc, v| {
                acc.extend_from_slice(&v.0[..]);
                acc
            });

        let max_power = {
            let power_windows = Windower::new(WindowType::Rectangle, &samples[..], 256, 512);
            let mut powers: Vec<f64> = power_windows.map(&power_calc).collect();
            powers.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            powers.first().unwrap_or(&0f64).clone()
        };

        let mean_mfccs = mfcc_calc(samples.clone());

        Sound { 
            max_power: max_power,
            name: name,
            samples: samples,
            sample_rate: sample_rate,
            mfccs: mfccs,
            mean_mfccs: mean_mfccs
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

    /// Writes the Sound to a 32-bit WAV file.
    pub fn write_file(&self, path: &Path) -> Result<(), Box<Error>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate as u32,
            bits_per_sample: 32
        };

        let mut file = try!(hound::WavWriter::create(&path, spec));
        for sample in self.samples.iter() {
            try!(file.write_sample((i32::max_value() as f64 * sample) as i32));
        }
        try!(file.finalize());
        Ok(())
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

    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
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

    pub fn mean_mfccs(&self) -> &Euclid<[f64; NCOEFFS]> {
        &self.mean_mfccs
    }
}

/// Cache of Sounds that can be referenced using an outside sound, to find the "most similar"
/// sound.
pub struct SoundDictionary {
    pub sounds: Vec<Sound>
}

impl SoundDictionary {
    /// Generates a SoundDictionary by walking a given path. Calculates MFCCs of all the Sounds in
    /// advance.
    pub fn from_path(path: &Path) -> Result<SoundDictionary, Box<Error>> {
        let dir = try!(path.read_dir());
        let mut out = SoundDictionary {sounds: Vec::<Sound>::with_capacity(dir.size_hint().0) }; 

        for res in dir {
            let entry = try!(res);
            match entry.path().extension() {
                Some(ext) => { if ext != "wav" { continue } },
                None => { continue }
            }

            out.sounds.push(try!(Sound::from_path(&entry.path())));
        }
        Ok(out)
    }

    /// Finds the most similar Sound in the SoundDictionary to `other`. Requires that the two
    /// Sounds have mfccs of the same length
    pub fn match_sound(&self, other: &Sound) -> Option<&Sound> {
        self.at_distance(0., other)
    }

    /// Finds the Sound closest to a particular distance away from a given Sound.
    pub fn at_distance(&self, distance: f64, other: &Sound) -> Option<&Sound> {
        self.sounds.iter().min_by_key(|a| OrdF64((cosine_sim_angular(&a.mean_mfccs().0, &other.mean_mfccs().0) - distance).abs()))
    }
}

/// Sequence of sounds and the distances between them
#[derive(Debug)]
pub struct SoundSequence<'a> {
    sounds: Vec<&'a Sound>,
    distances: Vec<f64>
}

impl<'a> IntoIterator for SoundSequence<'a> {
    type Item = &'a Sound;
    type IntoIter = ::std::vec::IntoIter<&'a Sound>;

    fn into_iter(self) -> Self::IntoIter {
        self.sounds.into_iter()
    }
}

impl<'a> SoundSequence<'a> {
    /// Create a new SoundSequence from a list of Sounds.
    pub fn new(sounds: Vec<&'a Sound>) -> SoundSequence {
        let distances = sounds.windows(2)
            .map(|pair| cosine_sim_angular(&pair[0].mean_mfccs().0, &pair[1].mean_mfccs().0))
            .collect();

        SoundSequence {
            sounds: sounds,
            distances: distances
        }
    }

    /// Given a starting sound, list of distances, and a dictionary, constructs a SoundSequence
    /// where each distance from one to the next is as close as possible to the target distances.
    pub fn from_distances(distances: &[f64], start: &'a Sound, dict: &'a SoundDictionary) -> Result<SoundSequence<'a>, String> {
        let mut sounds = Vec::<&Sound>::with_capacity(distances.len() + 1);
        sounds.push(start);

        for distance in distances {
            match dict.at_distance(*distance, sounds.last().unwrap()) {
                Some(s) => { sounds.push(s) },
                None => { return Err(format!("Cannot find sound at distance {}", distance)); }
            }
        }

        Ok(SoundSequence::new(sounds))
    }

    /// Gets a reference to the list of sounds
    pub fn sounds(&self) -> &Vec<&'a Sound> {
        &self.sounds
    }

    /// Concatenates all the Sounds into a single Sound
    pub fn to_sound(&self) -> Sound {
        let nsamples: usize = self.sounds.iter().fold(0, |acc, s| acc + s.samples.len());
        let mut samples: Vec<f64> = Vec::with_capacity(nsamples);
        for sound in self.sounds.iter() {
            samples.extend_from_slice(&sound.samples[..]);
        }
        Sound::from_samples(samples, self.sounds[0].sample_rate, None)
    }
}

#[test]
fn test_from_distances() {
    let dict = SoundDictionary::from_path(Path::new("data/phonemes")).unwrap();
    let start: &Sound = &dict.sounds[4];
    let distances = vec![0.5, 0.4, 0.3, 0.6, 0.2];
    let seq = SoundSequence::from_distances(&distances[..], &start, &dict).unwrap();
    for sound in seq.sounds() {
        println!("{}", sound.name.clone().unwrap_or("(no name)".to_string()));
    }
    let concat = seq.to_sound();
    concat.write_file(Path::new("tests/test_from_distances.wav")).unwrap();
}

#[test]
fn test_discretize_phonemes() {
    let dict = SoundDictionary::from_path(Path::new("data/phonemes")).unwrap();
    let mfcc_vecs: Vec<Euclid<[f64; NCOEFFS]>> = dict.sounds.iter()
        .filter_map(|s| { 
            if s.max_power() > 0.05 {
                Some(s.mean_mfccs().clone())
            } else {
                None
            }
        }).collect();
    let clusters = discretize(&mfcc_vecs[..]).clusters();

    for (idx, cluster) in clusters.iter().enumerate() {
        let mut sounds = Vec::<&Sound>::new();
        for sound_idx in cluster.1.iter() {
            sounds.push(&dict.sounds[*sound_idx])
        }
        SoundSequence::new(sounds).to_sound()
            .write_file(Path::new(&format!("tests/sound_cluster_{}.wav", idx).as_str())).unwrap();
    }
}

#[test]
fn test_sequence_to_sound() {
    let dict = SoundDictionary::from_path(Path::new("tests/dict")).unwrap();
    let sounds = vec![
        &dict.sounds[4],
        &dict.sounds[8],
        &dict.sounds[12]
    ];
    let seq = SoundSequence::new(sounds);
    let concat = seq.to_sound();
    concat.write_file(Path::new("tests/test_sequence_to_sound.wav")).unwrap();
}

