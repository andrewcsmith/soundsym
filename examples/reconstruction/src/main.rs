extern crate soundsym;
extern crate vox_box;
extern crate portaudio;
extern crate hound;
extern crate crossbeam;
extern crate bounded_spsc_queue;
extern crate piston_window;

#[macro_use] extern crate conrod;

use soundsym::*;
use portaudio::*;

use std::borrow::Cow;
use std::thread;
use std::path::Path;
use std::sync::Arc;
use std::mem::transmute;
use bounded_spsc_queue::{Producer, Consumer};

const BLOCK_SIZE: usize = 64;

widget_ids! {
    struct Ids { canvas, plot }
}

fn main() {
    run().unwrap();
}

#[allow(dead_code)]
enum DictionaryHandlerEvent {
    Refresh,
    Quit
}

use DictionaryHandlerEvent::*;

fn run() -> Result<(), Box<Error>> {
    crossbeam::scope(|scope| {
        // Read in the target file and create sequence using timestamps
        let target = Arc::new(Sound::from_path(Path::new("./Section_7_1.wav")).unwrap());
        let timestamps: Vec<Timestamp> = audacity_labels_to_timestamps(Path::new("./vowel.txt")).unwrap();
        let target_sequence = Arc::new(SoundSequence::from_timestamps(target.clone(), &timestamps[..]).unwrap());

        // Initialize the command queues
        let (input_buffer_producer, input_buffer_receiver) = bounded_spsc_queue::make::<[f32; BLOCK_SIZE]>(65536);
        let (audio_playback_producer, audio_playback_receiver) = bounded_spsc_queue::make::<f64>(target.samples().len() * 2);
        let (dictionary_commands_producer, dictionary_commands_receiver) = bounded_spsc_queue::make::<DictionaryHandlerEvent>(256);

        scope.spawn(move || dictionary_handler(input_buffer_receiver, target_sequence.clone(), audio_playback_producer, dictionary_commands_receiver));
        scope.spawn(move || audio_handler(input_buffer_producer, audio_playback_receiver, dictionary_commands_producer));
    });

    Ok(())
}

fn audio_handler(input_buffer_producer: Producer<[f32; BLOCK_SIZE]>, audio_playback_receiver: Consumer<f64>, dictionary_commands_producer: Producer<DictionaryHandlerEvent>) -> Result<(), Box<Error>> {
    let pa = try!(PortAudio::new());
    let mut frames_elapsed: usize = 0;
    let settings: DuplexStreamSettings<f32, f32> = try!(pa.default_duplex_stream_settings(1, 1, 44100., BLOCK_SIZE as u32));
    let callback = move |DuplexStreamCallbackArgs { in_buffer, out_buffer, frames, .. }| {

        unsafe {
            assert_eq!(BLOCK_SIZE, in_buffer.len());
            let in_buffer: &[f32; BLOCK_SIZE] = transmute(in_buffer.as_ptr());
            match input_buffer_producer.try_push(*in_buffer) {
                Some(_) => { println!("warning: sound buffer is full"); }
                None => { }
            }
        }

        if frames_elapsed % (441000 / 32) == 0 && frames_elapsed > 1500 {
            match dictionary_commands_producer.try_push(Refresh) {
                Some(_) => { println!("warning: dictionary event buffer is full"); },
                None => { }
            }
        }

        for s in out_buffer.iter_mut() {
            match audio_playback_receiver.try_pop() {
                Some(input) => { *s = input as f32 }
                None => { *s = 0. }
            }
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

fn dictionary_handler(input_buffer_receiver: Consumer<[f32; BLOCK_SIZE]>, target_sequence: Arc<SoundSequence>, audio_buffer_producer: Producer<f64>, dictionary_commands_receiver: Consumer<DictionaryHandlerEvent>) {
    let mut sound = Sound::from_samples(Vec::<f64>::with_capacity(65536), 44100., None, None);
    let mut buf = Vec::<f64>::with_capacity(65536);

    loop {
        while let Some(incoming_sound) = input_buffer_receiver.try_pop() {
            for s in incoming_sound.iter() {
                buf.push(*s as f64);
            }
            sound.push_samples(&buf[..]);
            buf.clear();
        };

        match dictionary_commands_receiver.try_pop() {
            Some(Refresh) => {
                let partitioner = Partitioner::new(Cow::Borrowed(&sound));
                let splits = partitioner.threshold(7).depth(7).partition().unwrap();
                let dict = SoundDictionary::from_segments(&sound, &splits[..]);
                println!("nsegs: {}", dict.sounds.len());
                let new_sound = target_sequence.clone_from_dictionary(&dict).unwrap().to_sound();
                println!("samps: {}", new_sound.samples().len());

                for s in new_sound.samples() {
                    // Blocks until there's space for new sound
                    audio_buffer_producer.push(*s);
                }
            },
            _ => { }
        }
    };
}

