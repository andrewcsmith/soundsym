extern crate soundsym;
extern crate vox_box;
extern crate portaudio;
extern crate hound;
extern crate crossbeam;
extern crate bounded_spsc_queue;

use soundsym::*;
use portaudio::*;

use std::borrow::Cow;
use std::thread;
use std::path::Path;
use std::sync::Arc;
use std::mem::transmute;
use bounded_spsc_queue::{Producer, Consumer};

const BLOCK_SIZE: usize = 64;

fn main() {
    run().unwrap();
}

#[allow(dead_code)]
enum DictionaryHandlerEvent {
    Refresh,
    Quit
}

fn run() -> Result<(), Box<Error>> {
    let pa = try!(PortAudio::new());
    let settings: DuplexStreamSettings<f32, f32> = try!(pa.default_duplex_stream_settings(1, 1, 44100., BLOCK_SIZE as u32));

    let (recorded_sound, recorded_sound_recv) = bounded_spsc_queue::make::<[f32; BLOCK_SIZE]>(65536);
    let mut frames_elapsed: usize = 0;

    let target = Arc::new(Sound::from_path(Path::new("./Section_7_1.wav")).unwrap());
    let timestamps: Vec<Timestamp> = audacity_labels_to_timestamps(Path::new("./vowel.txt")).unwrap();
    let target_sequence = Arc::new(SoundSequence::from_timestamps(target.clone(), &timestamps[..]).unwrap());
    let (sound_to_play, sound_to_play_recv) = bounded_spsc_queue::make::<f64>(target.samples().len() * 2);
    let (should_calculate_dictionary, should_calculate_dictionary_recv) = bounded_spsc_queue::make::<DictionaryHandlerEvent>(256);

    println!("nsegments: {}", target_sequence.sounds().len());
    dictionary_handler(recorded_sound_recv, target_sequence.clone(), sound_to_play, should_calculate_dictionary_recv);

    let callback = move |DuplexStreamCallbackArgs { in_buffer, out_buffer, frames, .. }| {

        unsafe {
            assert_eq!(BLOCK_SIZE, in_buffer.len());
            let in_buffer: &[f32; BLOCK_SIZE] = transmute(in_buffer.as_ptr());
            match recorded_sound.try_push(*in_buffer) {
                Some(_) => { println!("warning: sound buffer is full"); }
                None => { }
            }
        }

        if frames_elapsed % (441000 / 64) == 0 && frames_elapsed > 1500 {
            match should_calculate_dictionary.try_push(DictionaryHandlerEvent::Refresh) {
                Some(_) => { println!("warning: dictionary event buffer is full"); },
                None => { }
            }
        }

        let mut idx = 0;
        while let Some(s) = sound_to_play_recv.try_pop() {
            out_buffer[idx] = s as f32;
            idx += 1;
            if idx == frames { break; }
        }

        frames_elapsed += 1;
        Continue
    };

    let mut stream = try!(pa.open_non_blocking_stream(settings, callback));
    println!("starting stream");
    try!(stream.start());
    while stream.is_active().unwrap_or(false) { }
    Ok(())
}

fn dictionary_handler(recorded_sound: Consumer<[f32; BLOCK_SIZE]>, target: Arc<SoundSequence>, sound_to_play: Producer<f64>, should_calculate_dictionary: Consumer<DictionaryHandlerEvent>) {
    thread::spawn(move || {
        let mut buf = Vec::<f64>::with_capacity(65536);
        loop {
            while let Some(incoming_sound) = recorded_sound.try_pop() {
                for s in incoming_sound.iter() {
                    buf.push(*s as f64);
                }
            };

            if let Some(DictionaryHandlerEvent::Refresh) = should_calculate_dictionary.try_pop() {
                // TODO: Sound should be able to append samples and should calculate the MFCCs
                // and mean MFCCs accordingly. That way, this operation will be constant time.
                //
                // The operation to append samples should accept a boxed iterator.
                let sound = Sound::from_samples(buf.clone(), 44100., None);
                let partitioner = Partitioner::new(Cow::Borrowed(&sound));
                let splits = partitioner.threshold(7).depth(7).partition().unwrap();
                let dict = SoundDictionary::from_segments(&sound, &splits[..]);
                println!("nsegs: {}", dict.sounds.len());
                let new_sound = target.clone_from_dictionary(&dict).unwrap().to_sound();
                println!("samps: {}", new_sound.samples().len());

                for s in new_sound.samples() {
                    // Blocks until there's space for new sound
                    sound_to_play.push(*s);
                }
            };
        };
    });
}

