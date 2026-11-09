//! The output device, opened on a thread of its own and held there.
//!
//! Opening one is a round trip to the OS's audio server — tens of
//! milliseconds on CoreAudio, more when a Bluetooth headset is waking — and
//! none of it is the editor's to wait for, so startup never does. Until it
//! opens, the commands the main thread writes simply wait in the exchange. A
//! machine with no device at all, a CI runner among them, logs once and runs
//! silent.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::thread::JoinHandle;

use super::mixer::Mixer;

const OPENING: u8 = 0;
const OPEN: u8 = 1;
const FAILED: u8 = 2;

pub struct Device {
    status: Arc<AtomicU8>,
    quit: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Device {
    pub fn open(mixer: Mixer) -> Self {
        let status = Arc::new(AtomicU8::new(OPENING));
        let quit = Arc::new(AtomicBool::new(false));
        let thread = {
            let (status, quit) = (status.clone(), quit.clone());
            std::thread::Builder::new()
                .name("orrin-audio".to_owned())
                .spawn(move || hold(mixer, &status, &quit))
        };
        let thread = thread
            .map_err(|error| {
                tracing::warn!("audio: the output thread did not start ({error}); running silent");
                status.store(FAILED, Ordering::Release);
            })
            .ok();
        Self {
            status,
            quit,
            thread,
        }
    }

    /// No device opened and none will.
    pub fn failed(&self) -> bool {
        self.status.load(Ordering::Acquire) == FAILED
    }
}

/// The device buffer asked for, in frames: 21 ms at 48 kHz, the latency
/// between a script's call and the ear. rodio's own default aims for 50 ms
/// without asking the device, and CoreAudio on an M5 Pro refuses anything over
/// 1024, so the request is clamped to the range the device reports.
const BUFFER_FRAMES: u32 = 1024;

fn open() -> Result<rodio::stream::MixerDeviceSink, rodio::DeviceSinkError> {
    use rodio::cpal::traits::{DeviceTrait, HostTrait};
    use rodio::cpal::{BufferSize, SupportedBufferSize};

    let device = rodio::cpal::default_host()
        .default_output_device()
        .ok_or(rodio::DeviceSinkError::NoDevice)?;
    let supported = device
        .default_output_config()
        .map_err(rodio::DeviceSinkError::DefaultSinkConfigError)?;
    let buffer = match *supported.buffer_size() {
        SupportedBufferSize::Range { min, max } => {
            BufferSize::Fixed(BUFFER_FRAMES.clamp(min, max.max(min)))
        }
        SupportedBufferSize::Unknown => BufferSize::Default,
    };
    rodio::DeviceSinkBuilder::from_device(device)?
        .with_buffer_size(buffer)
        .open_sink_or_fallback()
}

/// Open the default device, give it the mix, and keep it open until told.
fn hold(mixer: Mixer, status: &AtomicU8, quit: &AtomicBool) {
    let mut sink = match open() {
        Ok(sink) => sink,
        Err(error) => {
            let mut reason = error.to_string();
            let mut cause = std::error::Error::source(&error);
            while let Some(inner) = cause {
                reason = format!("{reason}: {inner}");
                cause = inner.source();
            }
            tracing::warn!("audio: no output device ({reason}); running silent");
            status.store(FAILED, Ordering::Release);
            return;
        }
    };
    sink.log_on_drop(false);
    let config = sink.config();
    tracing::info!(
        "audio: {} channels at {} Hz",
        config.channel_count(),
        config.sample_rate()
    );
    sink.mixer().add(mixer);
    status.store(OPEN, Ordering::Release);
    while !quit.load(Ordering::Acquire) {
        std::thread::park();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}
