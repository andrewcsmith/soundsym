#[allow(dead_code)]
pub enum DictionaryHandlerEvent {
    Refresh,
    SetThreshold(usize),
    SetDepth(usize),
    Quit
}

pub enum AudioHandlerEvent {
    Quit
}

