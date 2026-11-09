//! An audio file as samples: decoded whole for a sound, opened as a stream
//! for music.
//!
//! A sound is decoded up front because it is short and may play many times at
//! once, and a voice reading decoded samples costs a multiply-add a frame. A
//! three-minute track decoded the same way is 70 MB of `f32`, so music is
//! decoded as it plays instead, on the device's thread, one voice at a time.

use std::io::Cursor;
use std::num::NonZero;

use rodio::Source;
use rodio::source::UniformSourceIterator;

use super::mixer::{RATE, Stream};

/// Decoded audio, interleaved, at its own rate and channel count.
pub struct Clip {
    channels: u16,
    rate: u32,
    samples: Box<[f32]>,
}

impl Clip {
    pub fn new(channels: u16, rate: u32, samples: Box<[f32]>) -> Self {
        assert!(channels > 0 && rate > 0, "a clip has a channel and a rate");
        Self {
            channels,
            rate,
            samples,
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Frame `index` as a stereo pair: a mono clip in both ears, the first two
    /// channels of anything wider, or every channel's mean in both when
    /// `downmix`.
    #[inline]
    pub fn frame(&self, index: usize, downmix: bool) -> [f32; 2] {
        let channels = self.channels as usize;
        let frame = &self.samples[index * channels..(index + 1) * channels];
        match (channels, downmix) {
            (1, _) => [frame[0]; 2],
            (_, false) => [frame[0], frame[1]],
            (_, true) => [frame.iter().sum::<f32>() / channels as f32; 2],
        }
    }

    /// Decode a WAV, Ogg Vorbis or MP3 file whole. `extension` is a hint for
    /// the container, not a requirement.
    pub fn decode(bytes: Vec<u8>, extension: &str) -> Result<Self, String> {
        let mut decoder = builder(bytes, extension)
            .build()
            .map_err(|error| error.to_string())?;
        let channels = decoder.channels();
        let rate = decoder.sample_rate();
        let mut samples: Vec<f32> = decoder.by_ref().collect();
        if decoder.channels() != channels || decoder.sample_rate() != rate {
            return Err(format!(
                "it changes from {channels} channels at {rate} Hz to {} at {} part way \
                 through, which a clip cannot; re-export it in one format",
                decoder.channels(),
                decoder.sample_rate()
            ));
        }
        let channels = channels.get();
        samples.truncate(samples.len() / channels as usize * channels as usize);
        if samples.is_empty() {
            return Err("it decodes to no samples".to_owned());
        }
        Ok(Self::new(channels, rate.get(), samples.into_boxed_slice()))
    }
}

/// Open a file to be decoded as it plays, already converted to the mixer's
/// stereo at [`RATE`], looping back to its start when `looping`.
pub fn stream(bytes: Vec<u8>, extension: &str, looping: bool) -> Result<Stream, String> {
    let channels = NonZero::new(2).unwrap();
    let rate = NonZero::new(RATE).unwrap();
    let builder = builder(bytes, extension).with_gapless(true);
    let error = |error: rodio::decoder::DecoderError| error.to_string();
    Ok(if looping {
        let decoder = builder.build_looped().map_err(error)?;
        Box::new(UniformSourceIterator::new(decoder, channels, rate))
    } else {
        let decoder = builder.build().map_err(error)?;
        Box::new(UniformSourceIterator::new(decoder, channels, rate))
    })
}

fn builder(bytes: Vec<u8>, extension: &str) -> rodio::decoder::DecoderBuilder<Cursor<Vec<u8>>> {
    let length = bytes.len() as u64;
    rodio::Decoder::builder()
        .with_data(Cursor::new(bytes))
        .with_byte_len(length)
        .with_seekable(true)
        .with_hint(extension)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A 16-bit PCM WAV, written by hand so the test owns every byte of it.
    pub fn wav(channels: u16, rate: u32, samples: &[i16]) -> Vec<u8> {
        let data = (samples.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data.to_le_bytes());
        for sample in samples {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        out
    }

    #[test]
    fn a_wav_decodes_to_its_samples_rate_and_channels() {
        let samples = [0, 16384, -16384, 32767, -32768, 0];
        let clip = Clip::decode(wav(2, 22_050, &samples), "wav").unwrap();
        assert_eq!(clip.channels, 2);
        assert_eq!(clip.rate(), 22_050);
        assert_eq!(clip.frames(), 3);
        assert_eq!(clip.frame(0, false), [0.0, 0.5]);
        assert_eq!(clip.frame(1, false), [-0.5, 32767.0 / 32768.0]);
        assert_eq!(clip.frame(2, true), [-0.5; 2]);
    }

    #[test]
    fn bytes_that_are_not_audio_are_an_error_rather_than_silence() {
        let error = Clip::decode(b"not a sound".to_vec(), "wav").err().unwrap();
        assert!(!error.is_empty());
        assert!(stream(b"not a sound".to_vec(), "ogg", false).is_err());
    }

    /// A stream is the mixer's stereo at its rate whatever the file was:
    /// a 24 kHz mono file comes out as twice as many stereo frames.
    #[test]
    fn a_stream_is_converted_to_the_mixers_format() {
        let samples = vec![8192i16; 2400];
        let out: Vec<f32> = stream(wav(1, RATE / 2, &samples), "wav", false)
            .unwrap()
            .collect();
        let frames = out.len() / 2;
        assert!((4790..=4810).contains(&frames), "{frames} frames");
        assert!((out[1000] - 0.25).abs() < 1e-3, "{}", out[1000]);
        assert_eq!(out[1000], out[1001]);
    }

    #[test]
    fn a_looping_stream_starts_again() {
        let out: Vec<f32> = stream(wav(2, RATE, &[1000; 20]), "wav", true)
            .unwrap()
            .take(400)
            .collect();
        assert_eq!(out.len(), 400);
        assert!(out.iter().all(|sample| *sample > 0.0));
    }
}
