use sample::{window, ToSampleSlice, FromSampleSlice, ToFrameSlice};

use vox_box::spectrum::MFCC;
use vox_box::waves::MaxAmplitude;
use vox_box::periodic::{Pitched, Hanning};

use rulinalg::utils;

use std::path::Path;
use std::error::Error;
use std::fmt;
use std::{f32, f64, i32};
use std::sync::Arc;
use std::fs::File;
use std::io;
use std::io::BufRead;

use super::*;

/// Calculates cosine similarity of the MFCC vectors of two Sounds. Truncates the longer Sound to
/// match the shorter Sound.
#[inline]
pub fn cosine_sim(me: &[f64], you: &[f64]) -> f64 {
    let len = if me.len() < you.len() {
        me.len() as usize
    } else {
        you.len() as usize
    };

    let nrm = norm(me) * norm(you);
    let dot = utils::dot(&me[..len], &you[..len]);
    dot / nrm
}

#[inline]
fn norm(me: &[f64]) -> f64 {
    me.iter().fold(0., |memo, item| item * item + memo)
}

/// Similar to the above, but does not calculate the norm of `you`. Suitable for comparing/ranking
/// elements that have the same "you" value.
#[inline]
#[allow(unused)]
pub fn cosine_sim_const_you(me: &[f64], you: &[f64]) -> Result<f64, CosError<'static>> {
    let len = if me.len() != you.len() {
        return Err(CosError("Vectors for cosine_sim must be same length"))
    } else {
        me.len() as usize
    };

    let nrm = norm(me);
    if nrm == 0f64 {
        Err(CosError("Norm equals zero"))
    } else {
        Ok(utils::dot(&me[..len], &you[..len]) / nrm)
    }
}

/// Angular cosine similarity. This bounds the result to `[0, 1]` while maintaining ordering. It is
/// also a proper metric.
#[inline]
pub fn cosine_sim_angular(me: &[f64; NCOEFFS], you: &[f64; NCOEFFS]) -> f64 {
    let sim: f64 = match cosine_sim(me, you) {
        x if x > 1.0 => 1.0,
        x if x < -1.0 => 1.0,
        x => x
    };
    sim.acos() * f64::consts::FRAC_1_PI
}

/// Struct holding sample information about a Sound and some pre-calculated data.
#[derive(Debug, Clone)]
pub struct Sound {
    /// Name for reference, often the filename. Can be changed at will.
    pub name: Option<String>,
    max_power: f64,
    pitch_confidence: Option<f64>,
    samples: Vec<f64>,
    sample_rate: f64,
    mfccs: Vec<f64>,
    mean_mfccs: [f64; NCOEFFS]
}

impl fmt::Display for Sound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}, {}, {}", &self.name.clone().unwrap_or("n/a".to_string()), &self.max_power, &self.samples.len())
    }
}

impl Sound {
    /// Generate a Sound from a Vec of f64 samples. Calculates max_power and mfccs.
    pub fn from_samples(samples: Vec<f64>, sample_rate: f64, mfccs: Option<Vec<f64>>, name: Option<String>) -> Sound {
        // println!("mfccs analyzing");
        let mfccs: Vec<f64> = mfccs.unwrap_or(analyze_mfccs(sample_rate, &samples[..]));
        // println!("mfccs analyzed");
        let max_power = analyze_max_power(&samples[..]);
        // println!("max power analyzed");
        let mean_mfccs = analyze_mean_mfccs(&mfccs[..]);
        // println!("mean mfccs analyzed");

        let sound = Sound { 
            max_power: max_power,
            pitch_confidence: None,
            name: name,
            samples: samples,
            sample_rate: sample_rate,
            mfccs: mfccs,
            mean_mfccs: mean_mfccs
        };

        sound
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
        Ok(Sound::from_samples(samples.clone(), sample_rate as f64, None, tag))
    }

    /// Writes the Sound to a 32-bit WAV file.
    pub fn write_file(&self, path: &Path) -> Result<(), Box<Error>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Int
        };

        let mut file = try!(hound::WavWriter::create(&path, spec));
        for sample in self.samples.iter() {
            try!(file.write_sample((i32::max_value() as f64 * sample) as i32));
        }
        try!(file.finalize());
        Ok(())
    }

    pub fn push_samples(&mut self, new_samples: &[f64]) {
        let initial_frames = self.num_frames();
        self.samples.extend_from_slice(new_samples);
        let samples_to_analyze = &self.samples[initial_frames * HOP..];

        {
            let new_mfccs = analyze_mfccs(self.sample_rate, samples_to_analyze);
            self.mfccs.extend_from_slice(&new_mfccs[..]);
            let new_frames = self.num_frames() - initial_frames;
            let new_mean_mfccs = analyze_mean_mfccs(&new_mfccs[..]);
            for (coeff, new) in self.mean_mfccs.iter_mut().zip(new_mean_mfccs.iter()) {
                *coeff = (*coeff * initial_frames as f64 + new * new_frames as f64) * 0.5;
            }
        }

        {
            let new_max_power = self.max_power.max(analyze_max_power(samples_to_analyze));
            self.max_power = new_max_power;
        }
    }

    pub fn max_power(&self) -> f64 {
        self.max_power
    }

    pub fn pitch_confidence(&self) -> f64 {
        match self.pitch_confidence {
            Some(x) => x,
            None => analyze_pitch_confidence(&self.samples[..])
        }
    }

    pub fn preload_pitch_confidence(&mut self) {
        self.pitch_confidence = Some(analyze_pitch_confidence(&self.samples[..]));
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

    /// Get the vector of MFCCs, broken into individual NCOEFFS-dimensional stack arrays
    pub fn mfcc_arrays(&self) -> Vec<[f64; NCOEFFS]> {
        self.mfccs.chunks(NCOEFFS)
            .fold(Vec::<[f64; NCOEFFS]>::with_capacity(self.mfccs.len() / NCOEFFS), |mut acc, v| {
                let mut arr = [0f64; NCOEFFS];
                for (idx, i) in v.iter().enumerate() { arr[idx] = *i; }
                acc.push(arr);
                acc
            })
    }

    pub fn mean_mfccs(&self) -> &[f64; NCOEFFS] {
        &self.mean_mfccs
    }

    pub fn num_frames(&self) -> usize {
        self.mfccs.len() / NCOEFFS
    }
}

fn analyze_mfccs(sample_rate: f64, samples: &[f64]) -> Vec<f64> {
    let mfcc_calc = |frame: &[f64]| -> [f64; NCOEFFS] { 
        let mut mfccs = [0f64; NCOEFFS];
        let m = frame.mfcc(NCOEFFS, (100., 8000.), sample_rate as f64);
        for (i, c) in m.iter().enumerate() {
            mfccs[i] = *c;
        }
        mfccs
    };

    let mut frame_buffer: Vec<f64> = Vec::with_capacity(BIN);

    // A single vector of mfccs
    window::Windower::hanning(
        <&[[f64; 1]]>::from_sample_slice(samples).unwrap(), BIN, HOP)
        .map(|frame| {
            for s in frame.take(BIN) {
                frame_buffer.extend_from_slice(<&[f64]>::to_sample_slice(&s[..]));
            }
            let mfccs = mfcc_calc(&frame_buffer[..]);
            frame_buffer.clear();
            mfccs
        })
        .fold(Vec::<f64>::with_capacity(samples.len() * NCOEFFS / BIN), |mut acc, v| {
            acc.extend_from_slice(&v[..]);
            acc
        })
}

fn analyze_max_power(samples: &[f64]) -> f64 {
    window::Windower::rectangle(
        <&[[f64; 1]]>::from_sample_slice(&samples[..]).unwrap(), BIN, HOP)
        .map(|frame| {
            let mut count: usize = 0;
            (frame.take(BIN).fold(0., |acc, s| {
                let sample_slice: &[f64] = <&[f64]>::to_sample_slice(&s[..]);
                count += 1;
                acc + sample_slice[0].powi(2)
            }) / count as f64).sqrt()
        })
        .fold(0., |acc, el| acc.max(el))
}

fn analyze_pitch_confidence(samples: &[f64]) -> f64 {
    let maxima: f64 = samples.to_sample_slice().max_amplitude();
    // Window the sound and find the maximum pitch confidence anywhere in the sound
    let frame_slice: &[[f64; 1]] = &samples[..].to_frame_slice().unwrap();
    window::Windower::hanning(frame_slice, 2048, 1024).map(|chunk| {
        let chunk_data: Vec<[f64; 1]> = chunk.collect();
        chunk_data.to_sample_slice().pitch::<Hanning>(44100., 0.2, maxima, maxima, 100., 500.)
    }).fold(0f64, |acc, x| (x[0].strength as f64).max(acc))
}

fn analyze_mean_mfccs(mfccs: &[f64]) -> [f64; NCOEFFS] {
    let nframes = mfccs.len() / NCOEFFS;
    let mfcc_arrays: &[[f64; NCOEFFS]] = <&[[f64; NCOEFFS]]>::from_sample_slice(&mfccs[..]).unwrap();
    let sums = mfcc_arrays.iter().fold([0f64; NCOEFFS], |mut acc, el| {
        for (idx, s) in el.iter().enumerate() {
            acc[idx] += *s;
        }
        acc
    });
    let mean_iter = sums.iter().map(|el| el / nframes as f64);
    let mut out = [0f64; NCOEFFS];
    for (idx, m) in (0..NCOEFFS).zip(mean_iter) {
        out[idx] = m;
    }
    out
}

/// Cache of Sounds that can be referenced using an outside sound, to find the "most similar"
/// sound.
pub struct SoundDictionary {
    pub sounds: Vec<Arc<Sound>>
}

impl SoundDictionary {
    /// An empty SoundDictionary
    pub fn new() -> SoundDictionary {
        SoundDictionary {
            sounds: Vec::new()
        }
    }

    /// Generates a SoundDictionary by walking a given path. Calculates MFCCs of all the Sounds in
    /// advance.
    pub fn from_path(path: &Path) -> Result<SoundDictionary, Box<Error>> {
        let dir = try!(path.read_dir());
        let mut out = SoundDictionary {
            sounds: Vec::<Arc<Sound>>::with_capacity(dir.size_hint().0) 
        }; 

        for res in dir {
            let entry = try!(res);
            match entry.path().extension() {
                Some(ext) => { if ext != "wav" { continue } },
                None => { continue }
            }

            out.sounds.push(Arc::new(try!(Sound::from_path(&entry.path()))));
        }
        Ok(out)
    }

    /// Builds a SoundDictionary from a source sound and a list of segment lengths
    pub fn from_segments(sound: &Sound, segments: &[usize]) -> SoundDictionary {
        let mut dict = SoundDictionary::new();
        dict.add_segments(&sound, segments);
        dict
    }

    /// Adds segments from a given Sound into the dictionary
    pub fn add_segments(&mut self, sound: &Sound, segments: &[usize]) {
        let mut samples = sound.samples().iter();
        let mut mfccs = sound.mfccs().iter();
        let new_sounds: Vec<Arc<Sound>> = segments.iter().map(|seg| {
            let samp: Vec<f64> = samples.by_ref().take(*seg).map(|s| *s).collect();
            let mfccs: Vec<f64> = mfccs.by_ref().take(*seg / HOP * NCOEFFS).map(|s| *s).collect();
            (samp, mfccs)
        }).map(|(samp, mfccs)| {
            let mut sound = Sound::from_samples(samp, sound.sample_rate(), Some(mfccs), None);
            // sound.preload_pitch_confidence();
            Arc::new(sound)
        }).collect();
        self.sounds.extend(new_sounds);
    }

    /// Finds the most similar Sound in the SoundDictionary to `other`
    pub fn match_sound(&self, other: &Sound) -> Option<Arc<Sound>> {
        self.at_distance(1., other)
    }

    /// Finds the Sound closest to a particular distance away from a given Sound.
    pub fn at_distance(&self, distance: f64, other: &Sound) -> Option<Arc<Sound>> {
        let (min_idx, min_distance) = self.sounds.iter()
            .map(|s| {
                let sim = cosine_sim(s.mfccs(), other.mfccs());
                // println!("\n\nMean 1: {:?}\nMean 2: {:?}", s.mean_mfccs(), other.mean_mfccs());
                // println!("Sim: {}", sim);
                sim
            })
            .map(|v| (v - distance).abs())
            .enumerate()
            .fold((0usize, 2f64), |(mut min_idx, mut min_distance), (idx, distance)| {
                if distance < min_distance { 
                    min_idx = idx;
                    min_distance = distance;
                }
                (min_idx, min_distance)
            });
        // println!("idx: {:04}\tdist: {:0.3}", min_idx, min_distance);
        Some(self.sounds[min_idx].clone())
    }
}

/// Sequence of sounds and the distances between them
#[derive(Debug)]
pub struct SoundSequence {
    sounds: Vec<Arc<Sound>>,
    distances: Vec<f64>
}

impl fmt::Display for SoundSequence {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(write!(f, "Sounds:\n"));
        for sound in self.sounds.iter() {
            try!(write!(f, "{}\n", sound));
        }
        Ok(())
    }
}

impl SoundSequence {
    /// Create a new SoundSequence from a list of Sounds.
    pub fn new(sounds: Vec<Arc<Sound>>) -> SoundSequence {
        let distances = sounds.windows(2)
            .map(|pair| cosine_sim_angular(&pair[0].mean_mfccs(), &pair[1].mean_mfccs()))
            .collect();

        SoundSequence {
            sounds: sounds,
            distances: distances
        }
    }

    /// Given a starting sound, list of distances, and a dictionary, constructs a SoundSequence
    /// where each distance from one to the next is as close as possible to the target distances.
    pub fn from_distances(distances: &[f64], start: Arc<Sound>, dict: &SoundDictionary) -> Result<SoundSequence, String> {
        let mut sounds = Vec::<Arc<Sound>>::with_capacity(distances.len() + 1);
        sounds.push(start);

        for distance in distances {
            match dict.at_distance(*distance, sounds.last().unwrap()) {
                Some(s) => { sounds.push(s) },
                None => { return Err(format!("Cannot find sound at distance {}", distance)); }
            }
        }

        Ok(SoundSequence::new(sounds))
    }

    pub fn from_timestamps(sound: Arc<Sound>, timestamps: &[Timestamp]) -> Result<SoundSequence, String> {
        Ok(SoundSequence::new(timestamps.iter().fold(Vec::with_capacity(timestamps.len()), |mut acc, timestamp| {
            let &Timestamp(start, end, ref label) = timestamp.clone();
            let start_sample = (start * sound.sample_rate() as f64).round() as usize;
            let end_sample = (end * sound.sample_rate() as f64).round() as usize;
            let samples: Vec<f64> = (sound.samples()[start_sample..(end_sample + 1)]).iter().cloned().collect();
            let new_sound = Sound::from_samples(samples, sound.sample_rate(), None, label.clone());
            acc.push(Arc::new(new_sound));
            acc
        })))
    }

    /// Gets a reference to the list of sounds
    pub fn sounds(&self) -> &Vec<Arc<Sound>> {
        &self.sounds
    }
    
    pub fn distances(&self) -> &Vec<f64> {
        &self.distances
    }

    pub fn morph_to(&self, distances: &[f64], dict: &SoundDictionary) -> Result<SoundSequence, String> {
        let mut sounds = Vec::<Arc<Sound>>::with_capacity(self.sounds.len());
        for (sound, distance) in self.sounds().iter().zip(distances.iter()) {
            match dict.at_distance(*distance, sound) {
                Some(s) => { sounds.push(s.clone()) },
                None => { return Err(format!("Cannot find sound at distance {}", distance)); }
            }
        }
        Ok(SoundSequence::new(sounds))
    }

    pub fn clone_from_dictionary(&self, dict: &SoundDictionary) -> Result<SoundSequence, String> {
        let mut sounds = Vec::<Arc<Sound>>::with_capacity(self.sounds.len());
        for sound in self.sounds.iter() {
            match dict.match_sound(sound) {
                Some(s) => { 
                    let diff = sound.samples().len() as i64 - s.samples().len() as i64;
                    let new_sound = if diff > 0 {
                        let samps: Vec<f64> = s.samples().iter().chain([0.].iter().cycle().take(diff as usize)).cloned().collect();
                        Arc::new(Sound::from_samples(samps, sound.sample_rate(), None, None))
                    } else if diff < 0 {
                        let samps: Vec<f64> = s.samples().iter().take(sound.samples().len()).cloned().collect();
                        Arc::new(Sound::from_samples(samps, sound.sample_rate(), None, None))
                    } else {
                        s
                    };
                    sounds.push(new_sound);
                },
                None => { }
            }
        }
        Ok(SoundSequence::new(sounds))
    }

    /// Concatenates all the Sounds into a single Sound
    pub fn to_sound(&self) -> Sound {
        let nsamples: usize = self.sounds.iter().fold(0, |acc, s| acc + s.samples.len());
        let mut samples: Vec<f64> = Vec::with_capacity(nsamples);
        for sound in self.sounds.iter() {
            samples.extend_from_slice(&sound.samples[..]);
        }
        let sample_rate = self.sounds.get(0).map(|s| s.sample_rate).unwrap_or(44100.);
        Sound::from_samples(samples, sample_rate, None, None)
    }
}

pub fn max_index(vals: &[f64]) -> usize {
    vals.iter().enumerate()
        .fold((0usize, 0f64), |(max_idx, max_dist), (idx, dist)| {
            if *dist > max_dist {
                (idx, *dist)
            } else {
                (max_idx, max_dist)
            }
        }).0
}

pub fn min_index(vals: &[f64]) -> usize {
    vals.iter().enumerate()
        .fold((0usize, 1f64), |(min_idx, min_dist), (idx, dist)| {
            if *dist < min_dist {
                (idx, *dist)
            } else {
                (min_idx, min_dist)
            }
        }).0
}

pub struct Timestamp(f64, f64, Option<String>); 

#[allow(unused)]
pub fn audacity_labels_to_timestamps(path: &Path) -> Result<Vec<Timestamp>, io::Error> {
    let mut file = try!(File::open(&path));
    let mut reader = io::BufReader::new(file);

    let mut line = String::new();
    let mut timestamps = Vec::new();

    while reader.read_line(&mut line).unwrap() > 0 {
        {
            let mut stamp = Timestamp(0., 0., None);
            let mut split = line.trim().split('\t');
            stamp.0 = split.next().map(|s| s.parse::<f64>()).unwrap_or(Ok(0.)).unwrap_or(0.);
            stamp.1 = split.next().map(|s| s.parse::<f64>()).unwrap_or(Ok(0.)).unwrap_or(0.);
            stamp.2 = split.next().map(|s| s.to_string());
            timestamps.push(stamp);
        }

        line.clear();
    }

    Ok(timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn test_audacity_labels_to_timestamps() {
        let labels = Path::new("tests/vowel.txt");
        let timestamps = audacity_labels_to_timestamps(&labels).unwrap();
        assert!((timestamps[0].0 - 0.7065779155923718).abs() < 1e-10);
        assert!((timestamps[26].1 - 5.59353222977394).abs() < 1e-10);
        assert_eq!(timestamps[44].2, Some("ning".to_string()));
        assert_eq!(timestamps.len(), 55);
    }

    #[test]
    fn test_sound_sequence_from_timestamps() {
        let sound = Sound::from_path(&Path::new("tests/Section_7_1.wav")).unwrap();
        let timestamps = vec![
            Timestamp(0.7065779155923718, 0.7619218551399829, Some("o".to_string())),
            Timestamp(0.7619218551399829, 1.0201935730288352, Some("s".to_string()))
        ];
        let ss = SoundSequence::from_timestamps(Arc::new(sound), &timestamps[..]).unwrap();
        assert_eq!(ss.sounds()[0].name, Some("o".to_string()));
    }

    // #[test]
    fn test_from_distances() {
        let dict = SoundDictionary::from_path(Path::new("data/phonemes")).unwrap();
        let start: Arc<Sound> = dict.sounds[4].clone();
        let distances = vec![0.5, 0.4, 0.3, 0.6, 0.2];
        let seq = SoundSequence::from_distances(&distances[..], start, &dict).unwrap();
        for sound in seq.sounds() {
            println!("{}", sound.name.clone().unwrap_or("(no name)".to_string()));
        }
        let concat = seq.to_sound();
        concat.write_file(Path::new("tests/test_from_distances.wav")).unwrap();
    }

    #[test]
    fn test_sequence_to_sound() {
        let dict = SoundDictionary::from_path(Path::new("tests/dict")).unwrap();
        let sounds = vec![
            dict.sounds[4].clone(),
            dict.sounds[8].clone(),
            dict.sounds[12].clone()
        ];
        let seq = SoundSequence::new(sounds);
        let concat = seq.to_sound();
        concat.write_file(Path::new("tests/test_sequence_to_sound.wav")).unwrap();
    }

    #[test]
    fn test_sound_should_match_itself() {
        let dict = SoundDictionary::from_path(Path::new("tests/dict")).unwrap();
        let sound = &dict.sounds[4];
        println!("{:?}", sound.mean_mfccs());
        println!("{:?}", cosine_sim(sound.mean_mfccs(), sound.mean_mfccs()));
        println!("{:?}", cosine_sim_angular(sound.mean_mfccs(), sound.mean_mfccs()));
        assert!(cosine_sim_angular(sound.mean_mfccs(), sound.mean_mfccs()) < 1.0e-6);
        assert_eq!(sound.samples(), dict.match_sound(sound).unwrap().samples());
    }

    #[test]
    fn test_angular_distance() {
        let mfccs = [0.1, 0.4, 0.2, 0.8, 0., 0., 0., 0., 0., 0., 0., 0.];
        assert_eq!(cosine_sim_angular(&mfccs, &mfccs), 0.0);
    }

    #[test]
    fn test_push_samples() {
        let samples = vec![0f64; 2048];
        let mut sound = Sound::from_samples(samples, 44100., None, None);
        sound.push_samples(&vec![0f64; 2048 + 1024][..]);
        assert_eq!(sound.samples().len(), 5120);
        assert_eq!(sound.num_frames(), 5);
    }

    #[test]
    fn test_empty_sound() {
        let mut sound = Sound::from_samples(Vec::new(), 44100., None, None);
        sound.push_samples(&vec![0f64; 4096 + 1024]);
        assert_eq!(sound.num_frames(), 5);
    }
}

