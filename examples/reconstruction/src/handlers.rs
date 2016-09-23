extern crate soundsym;
extern crate portaudio;
extern crate piston_window;
extern crate bounded_spsc_queue;
extern crate conrod;
extern crate find_folder;

use soundsym::*;
use portaudio::{Continue, DuplexStreamCallbackArgs, DuplexStreamSettings, PortAudio};
use bounded_spsc_queue::{Producer, Consumer};
use crossbeam::sync::SegQueue;

use std::borrow::Cow;
use std::sync::Arc;
use std::mem::transmute;

use super::*;

const BLOCK_SIZE: usize = 64;

pub fn gui_handler(audio_commands_producer: Producer<AudioHandlerEvent>, dictionary_commands_producer: Producer<DictionaryHandlerEvent>) -> Result<(), Error> {
    use piston_window::{EventLoop, PistonWindow, UpdateEvent, WindowSettings};
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    let mut window: PistonWindow = 
        try!(WindowSettings::new("Control window", [WIDTH, HEIGHT])
            .opengl(piston_window::OpenGL::V3_2)
            .samples(4)
            .exit_on_esc(true)
            .build());
    window.set_ups(60);
    let mut ui = conrod::UiBuilder::new().build();
    let ids = Ids::new(ui.widget_id_generator());
    let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
    let font_path = assets.join("LH-Line1-Sans-Thin.ttf");
    try!(ui.fonts.insert_from_file(font_path));
    let mut text_texture_cache = conrod::backend::piston_window::GlyphCache::new(&mut window, WIDTH, HEIGHT);
    let image_map = conrod::image::Map::new();

    let mut threshold_text = String::new();
    let mut depth_text = String::new();

    while let Some(event) = window.next() {
        use input::{Event, Input, Button};
        use input::keyboard::Key;

        if let Some(e) = conrod::backend::piston_window::convert_event(event.clone(), &window) {
            ui.handle_event(e);
        }

        // Handle the raw events, primarily keyboard events
        match event {
            Event::Input(Input::Press(Button::Keyboard(key))) => { 
                match key {
                    Key::Space => {
                        dictionary_commands_producer.push(DictionaryHandlerEvent::Refresh);
                    }
                    _ => { }
                }
            }
            _ => { }
        }

        event.update(|_| {
            use conrod::{color, widget, Colorable, Positionable, Sizeable, Widget, Labelable};
            let ui = &mut ui.set_widgets();
            widget::Canvas::new().color(color::DARK_CHARCOAL).set(ids.canvas, ui);

            let button = widget::Button::new()
                .w_h(200., 50.)
                .middle()
                .label("Reconstruct")
                .set(ids.button, ui);
            
            for edit in widget::TextBox::new(&threshold_text)
                .align_text_middle()
                .set(ids.threshold_box, ui) 
            {
                match edit {
                    widget::text_box::Event::Update(new_text) => {
                        threshold_text = new_text;
                    }
                    widget::text_box::Event::Enter => {
                        if let Ok(new_threshold) = threshold_text.parse::<usize>() {
                            dictionary_commands_producer.push(DictionaryHandlerEvent::SetThreshold(new_threshold));
                        }
                    }
                }
            }

            for edit in widget::TextBox::new(&depth_text) 
                .align_text_middle()
                .set(ids.depth_box, ui)
            {
                match edit {
                    widget::text_box::Event::Update(new_text) => {
                        depth_text = new_text;
                    }
                    widget::text_box::Event::Enter => {
                        if let Ok(new_depth) = depth_text.parse::<usize>() {
                            dictionary_commands_producer.push(DictionaryHandlerEvent::SetDepth(new_depth));
                        }
                    }
                }
            }

            if button.was_clicked() {
                dictionary_commands_producer.push(DictionaryHandlerEvent::Refresh);
            }

        });

        window.draw_2d(&event, |c, g| {
            if let Some(primitives) = ui.draw_if_changed() {
                fn texture_from_image<T>(img: &T) -> &T { img };
                conrod::backend::piston_window::draw(c, g, primitives,
                                                     &mut text_texture_cache,
                                                     &image_map,
                                                     texture_from_image);
            }
        });
    }

    audio_commands_producer.push(AudioHandlerEvent::Quit);
    dictionary_commands_producer.push(DictionaryHandlerEvent::Quit);
    Ok(())
}

pub fn audio_handler(input_buffer_producer: Producer<[f32; BLOCK_SIZE]>, audio_playback_queue: Arc<SegQueue<f64>>, audio_commands_receiver: Consumer<AudioHandlerEvent>) -> Result<(), Error> {
    use AudioHandlerEvent::*;

    let pa = try!(PortAudio::new());
    let mut frames_elapsed: usize = 0;
    let settings: DuplexStreamSettings<f32, f32> = try!(pa.default_duplex_stream_settings(1, 1, 44100., BLOCK_SIZE as u32).map_err(Error::PortAudio));
    let callback = move |DuplexStreamCallbackArgs { in_buffer, out_buffer, .. }| {

        unsafe {
            assert_eq!(BLOCK_SIZE, in_buffer.len());
            let in_buffer: &[f32; BLOCK_SIZE] = transmute(in_buffer.as_ptr());
            match input_buffer_producer.try_push(*in_buffer) {
                Some(_) => { println!("warning: sound buffer is full"); }
                None => { }
            }
        }

        for s in out_buffer.iter_mut() {
            match audio_playback_queue.try_pop() {
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

    while stream.is_active().unwrap_or(false) { 
        match audio_commands_receiver.try_pop() {
            Some(Quit) => { 
                try!(stream.stop()); 
            }
            None => { }
        }
    }

    Ok(())
}

pub fn dictionary_handler(input_buffer_receiver: Consumer<[f32; BLOCK_SIZE]>, target_sequence: Arc<SoundSequence>, audio_playback_queue: Arc<SegQueue<f64>>, dictionary_commands_receiver: Consumer<DictionaryHandlerEvent>) {
    use DictionaryHandlerEvent::*;

    let mut sound = Sound::from_samples(Vec::<f64>::with_capacity(65536), 44100., None, None);
    let mut buf = Vec::<f64>::with_capacity(65536);
    let mut threshold = 7;
    let mut depth = 7;

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
                let splits = partitioner.threshold(threshold).depth(depth).partition().unwrap();
                let dict = SoundDictionary::from_segments(&sound, &splits[..]);
                println!("nsegs: {}", dict.sounds.len());
                let new_sound = target_sequence.clone_from_dictionary(&dict).unwrap().to_sound();
                println!("samps: {}", new_sound.samples().len());

                for s in new_sound.samples() {
                    audio_playback_queue.push(*s);
                }
            }
            Some(SetThreshold(x)) => { threshold = x; }
            Some(SetDepth(x)) => { depth = x; }
            Some(Quit) => { return; }
            _ => { }
        }
    };
}

