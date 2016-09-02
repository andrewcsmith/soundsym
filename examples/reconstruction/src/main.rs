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
    let mut settings: DuplexStreamSettings<f32, f32> = try!(pa.default_duplex_stream_settings(1, 1, 44100., BLOCK_SIZE as u32));
    settings.out_params = StreamParameters::new(DeviceIndex(3), 1, true, 0.);

    let (recorded_sound, recorded_sound_recv) = bounded_spsc_queue::make::<[f32; BLOCK_SIZE]>(65536);
    let mut frames_elapsed: usize = 0;

    let target = Sound::from_path(Path::new("./we_remember_mono.wav")).unwrap();
    let nsamples = target.samples().len();
    let mut target_samples_iter = target.samples().iter();
    let splits = Partitioner::new(Cow::Borrowed(&target)).partition().unwrap();

    let sound_vec: Vec<Arc<Sound>> = splits.iter().map(|nsamples| {
        Arc::new(Sound::from_samples(target_samples_iter.by_ref().take(*nsamples).cloned().collect(), 44100., None))
    }).collect();

    let target_sequence = Arc::new(SoundSequence::new(sound_vec));
    let (sound_to_play, sound_to_play_recv) = bounded_spsc_queue::make::<f64>(nsamples * 2);
    let (should_calculate_dictionary, should_calculate_dictionary_recv) = bounded_spsc_queue::make::<DictionaryHandlerEvent>(256);

    println!("nsegments: {}", target_sequence.sounds().len());
    dictionary_handler(recorded_sound_recv, target_sequence.clone(), sound_to_play, should_calculate_dictionary_recv);

    let callback = move |DuplexStreamCallbackArgs { in_buffer, out_buffer, frames, .. }| {
        let in_buffer: &[f32; BLOCK_SIZE] = unsafe { transmute(in_buffer.as_ptr()) };
        match recorded_sound.try_push((*in_buffer)) {
            Some(_) => { println!("warning: sound buffer is full"); }
            None => { }
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
                let sound = Sound::from_samples(buf.clone(), 44100., None);
                let partitioner = Partitioner::new(Cow::Borrowed(&sound));
                let splits = partitioner.threshold(3).depth(5).partition().unwrap();
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

