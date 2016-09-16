use std;
use conrod;
use portaudio;

#[derive(Debug)]
pub enum Error {
    PortAudio(portaudio::Error),
    Font(conrod::text::font::Error),
    String(String)
}

impl From<String> for Error {
    fn from(string: String) -> Error {
        Error::String(string)
    }
}

impl From<portaudio::Error> for Error {
    fn from(pa_error: portaudio::Error) -> Error {
        Error::PortAudio(pa_error)
    }
}

impl From<conrod::text::font::Error> for Error {
    fn from(err: conrod::text::font::Error) -> Error {
        Error::Font(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Error::PortAudio(ref err) => write!(f, "PortAudio error: {}", err),
            Error::Font(ref err) => write!(f, "Font error: {}", err),
            Error::String(ref err) => write!(f, "String error: {}", err)
        }
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::PortAudio(ref err) => err.description(),
            Error::Font(ref err) => err.description(),
            Error::String(ref err) => err
        }
    }

    fn cause(&self) -> Option<&std::error::Error> {
        match *self {
            Error::PortAudio(ref err) => Some(err),
            Error::Font(ref err) => Some(err),
            Error::String(_) => None
        }
    }
}



