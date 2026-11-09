//! The half of audio that runs on the device's thread: every voice summed
//! into one stereo stream at [`RATE`], a block at a time.
//!
//! The device's callback is a real-time deadline, so the rule here is the one
//! a frame has, only stricter: nothing allocates, nothing blocks, and nothing
//! the main thread does can make it wait. The main thread never touches a
//! voice. It writes [`Command`]s into the shared [`Exchange`], and the mixer
//! takes them with a `try_lock` at the top of a block — contended, it plays
//! the block with what it already had and asks again next block, 5 ms later.
//! The exchange's vectors swap rather than copy, so their capacity survives,
//! and a finished voice's body goes back to the main thread to be dropped
//! there, because freeing a decoder is not a deadline's business.
//!
//! Every change of gain is a linear ramp across one block, a stop included:
//! a step in amplitude is a click, and the main thread only sends a new gain
//! once a frame. Resampling is linear interpolation per voice, which is also
//! what a clip authored at 44.1 kHz pays to play at 48.

use std::num::NonZero;
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;

use super::clip::Clip;

/// The mixer's own sample rate. Fixed rather than the device's, so a clip's
/// resampling step is known before any device has opened; rodio converts the
/// mix to whatever the device asks for, once.
pub const RATE: u32 = 48_000;

/// Frames per block: the granularity of a gain ramp and of the command
/// exchange. 5.3 ms at [`RATE`].
pub const BLOCK: usize = 256;

/// Interleaved stereo at [`RATE`], for a voice decoded as it plays.
pub type Stream = Box<dyn Iterator<Item = f32> + Send>;

/// What a voice plays.
pub enum Body {
    Clip { clip: Arc<Clip>, looping: bool },
    Stream(Stream),
}

pub enum Command {
    /// `downmix` folds every channel into one before the gains, which is what
    /// a spatial source wants: it is a point, and a stereo clip's own width
    /// would pan it twice.
    Play {
        id: u64,
        body: Body,
        gains: [f32; 2],
        downmix: bool,
    },
    Gains {
        id: u64,
        gains: [f32; 2],
    },
    Stop {
        id: u64,
    },
    Pause(bool),
}

/// What the two threads hand each other.
#[derive(Default)]
pub struct Exchange {
    /// Main to mixer.
    pub commands: Vec<Command>,
    /// Mixer to main: voices that ended, by id.
    pub finished: Vec<u64>,
    /// Mixer to main: their bodies, to be dropped off the audio thread.
    pub retired: Vec<Body>,
}

pub type Shared = Arc<Mutex<Exchange>>;

/// Enough that a busy scene never grows a vector on the audio thread.
pub const VOICES: usize = 256;

pub fn shared() -> Shared {
    Arc::new(Mutex::new(Exchange {
        commands: Vec::with_capacity(VOICES),
        finished: Vec::with_capacity(VOICES),
        retired: Vec::with_capacity(VOICES),
    }))
}

struct Voice {
    id: u64,
    body: Body,
    /// In the clip's frames, fractional between them.
    cursor: f64,
    /// The clip's frames per output frame.
    step: f64,
    gains: [f32; 2],
    target: [f32; 2],
    downmix: bool,
    stopping: bool,
}

impl Voice {
    /// Add one block of this voice into `out`. True once it has ended.
    fn render(&mut self, out: &mut [f32]) -> bool {
        let frames = out.len() / 2;
        let [from_left, from_right] = self.gains;
        let [to_left, to_right] = self.target;
        let per_frame = 1.0 / frames as f32;
        let mut ended = false;
        for (f, pair) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let Some([left, right]) = self.next_frame() else {
                ended = true;
                break;
            };
            let t = (f + 1) as f32 * per_frame;
            pair[0] += left * (from_left + (to_left - from_left) * t);
            pair[1] += right * (from_right + (to_right - from_right) * t);
        }
        self.gains = self.target;
        ended || self.stopping
    }

    #[inline]
    fn next_frame(&mut self) -> Option<[f32; 2]> {
        match &mut self.body {
            Body::Stream(stream) => {
                let left = stream.next()?;
                Some([left, stream.next().unwrap_or(0.0)])
            }
            Body::Clip { clip, looping } => {
                let frames = clip.frames();
                if self.cursor >= frames as f64 {
                    if !*looping || frames == 0 {
                        return None;
                    }
                    self.cursor %= frames as f64;
                }
                let index = self.cursor as usize;
                let t = (self.cursor - index as f64) as f32;
                let a = clip.frame(index, self.downmix);
                let b = match index + 1 {
                    next if next < frames => clip.frame(next, self.downmix),
                    _ if *looping => clip.frame(0, self.downmix),
                    _ => a,
                };
                self.cursor += self.step;
                Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            }
        }
    }
}

pub struct Mixer {
    shared: Shared,
    voices: Vec<Voice>,
    inbox: Vec<Command>,
    finished: Vec<u64>,
    retired: Vec<Body>,
    block: Box<[f32]>,
    at: usize,
    paused: bool,
}

impl Mixer {
    pub fn new(shared: Shared) -> Self {
        Self {
            shared,
            voices: Vec::with_capacity(VOICES),
            inbox: Vec::with_capacity(VOICES),
            finished: Vec::with_capacity(VOICES),
            retired: Vec::with_capacity(VOICES),
            block: vec![0.0; BLOCK * 2].into_boxed_slice(),
            at: BLOCK * 2,
            paused: false,
        }
    }

    fn render(&mut self) {
        self.exchange();
        self.at = 0;
        self.block.fill(0.0);
        if self.paused {
            return;
        }
        let mut index = 0;
        while index < self.voices.len() {
            if self.voices[index].render(&mut self.block) {
                let voice = self.voices.swap_remove(index);
                self.finished.push(voice.id);
                self.retired.push(voice.body);
            } else {
                index += 1;
            }
        }
    }

    /// Trade with the main thread, if it is not mid-write: its commands for
    /// what has ended since the last trade.
    fn exchange(&mut self) {
        let mut exchange = match self.shared.try_lock() {
            Ok(exchange) => exchange,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        std::mem::swap(&mut exchange.commands, &mut self.inbox);
        exchange.finished.append(&mut self.finished);
        exchange.retired.append(&mut self.retired);
        drop(exchange);

        let mut inbox = std::mem::take(&mut self.inbox);
        for command in inbox.drain(..) {
            self.apply(command);
        }
        self.inbox = inbox;
    }

    fn apply(&mut self, command: Command) {
        match command {
            Command::Play {
                id,
                body,
                gains,
                downmix,
            } => {
                let step = match &body {
                    Body::Clip { clip, .. } => clip.rate() as f64 / RATE as f64,
                    Body::Stream(_) => 1.0,
                };
                self.voices.push(Voice {
                    id,
                    body,
                    cursor: 0.0,
                    step,
                    gains,
                    target: gains,
                    downmix,
                    stopping: false,
                });
            }
            Command::Gains { id, gains } => {
                if let Some(voice) = self.voice(id)
                    && !voice.stopping
                {
                    voice.target = gains;
                }
            }
            Command::Stop { id } => {
                if let Some(voice) = self.voice(id) {
                    voice.target = [0.0; 2];
                    voice.stopping = true;
                }
            }
            Command::Pause(paused) => self.paused = paused,
        }
    }

    fn voice(&mut self, id: u64) -> Option<&mut Voice> {
        self.voices.iter_mut().find(|voice| voice.id == id)
    }
}

impl Iterator for Mixer {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.at == self.block.len() {
            self.render();
        }
        let sample = self.block[self.at];
        self.at += 1;
        Some(sample)
    }
}

impl rodio::Source for Mixer {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> rodio::ChannelCount {
        NonZero::new(2).unwrap()
    }

    fn sample_rate(&self) -> rodio::SampleRate {
        NonZero::new(RATE).unwrap()
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send(shared: &Shared, command: Command) {
        shared.lock().unwrap().commands.push(command);
    }

    fn play(shared: &Shared, id: u64, clip: Clip, looping: bool, gains: [f32; 2]) {
        send(
            shared,
            Command::Play {
                id,
                body: Body::Clip {
                    clip: Arc::new(clip),
                    looping,
                },
                gains,
                downmix: false,
            },
        );
    }

    fn pull(mixer: &mut Mixer, frames: usize) -> Vec<[f32; 2]> {
        (0..frames)
            .map(|_| [mixer.next().unwrap(), mixer.next().unwrap()])
            .collect()
    }

    fn mono(rate: u32, samples: &[f32]) -> Clip {
        Clip::new(1, rate, samples.into())
    }

    #[test]
    fn a_flat_mono_clip_comes_out_of_both_channels_at_its_gains() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        let samples: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        play(&shared, 1, mono(RATE, &samples), false, [0.5, 1.0]);

        let out = pull(&mut mixer, 64);
        for (frame, sample) in out.iter().zip(&samples) {
            assert_eq!(*frame, [sample * 0.5, *sample]);
        }
    }

    /// The end of a voice reaches the main thread at the next exchange, with
    /// its body to drop there.
    #[test]
    fn a_finished_voice_is_reported_and_its_body_retired() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        play(&shared, 7, mono(RATE, &[1.0; 10]), false, [1.0; 2]);

        let out = pull(&mut mixer, BLOCK * 2);
        assert!(out[..10].iter().all(|frame| *frame == [1.0; 2]));
        assert!(out[10..].iter().all(|frame| *frame == [0.0; 2]));
        let exchange = shared.lock().unwrap();
        assert_eq!(exchange.finished, [7]);
        assert_eq!(exchange.retired.len(), 1);
    }

    #[test]
    fn a_clip_at_half_the_rate_lasts_twice_as_long() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        play(&shared, 1, mono(RATE / 2, &[1.0; 100]), false, [1.0; 2]);

        let sounding = pull(&mut mixer, BLOCK)
            .iter()
            .filter(|frame| frame[0] > 0.0)
            .count();
        assert_eq!(sounding, 200);
    }

    #[test]
    fn a_looping_clip_wraps_and_never_finishes() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        play(&shared, 1, mono(RATE, &[0.1, 0.2, 0.3]), true, [1.0; 2]);

        let out = pull(&mut mixer, BLOCK * 4);
        for (i, frame) in out.iter().enumerate() {
            assert_eq!(frame[0], [0.1, 0.2, 0.3][i % 3], "frame {i}");
        }
        assert!(shared.lock().unwrap().finished.is_empty());
    }

    /// A new gain is reached across one block, never in a step.
    #[test]
    fn a_change_of_gains_ramps_across_one_block() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        play(&shared, 1, mono(RATE, &[1.0]), true, [0.0; 2]);
        send(
            &shared,
            Command::Gains {
                id: 1,
                gains: [1.0, 0.5],
            },
        );

        let out = pull(&mut mixer, BLOCK * 2);
        for (f, frame) in out[..BLOCK].iter().enumerate() {
            let t = (f + 1) as f32 / BLOCK as f32;
            assert!((frame[0] - t).abs() < 1e-6, "frame {f}: {frame:?}");
            assert!((frame[1] - t * 0.5).abs() < 1e-6, "frame {f}: {frame:?}");
        }
        assert!(out[BLOCK..].iter().all(|frame| *frame == [1.0, 0.5]));
    }

    #[test]
    fn a_stop_fades_out_over_one_block_rather_than_clicking() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        play(&shared, 3, mono(RATE, &[1.0]), true, [1.0; 2]);
        pull(&mut mixer, BLOCK);
        send(&shared, Command::Stop { id: 3 });

        let out = pull(&mut mixer, BLOCK * 2);
        for (f, frame) in out[..BLOCK].iter().enumerate() {
            let expected = 1.0 - (f + 1) as f32 / BLOCK as f32;
            assert!((frame[0] - expected).abs() < 1e-6, "frame {f}: {frame:?}");
        }
        assert_eq!(out[BLOCK - 1], [0.0; 2]);
        assert!(out[BLOCK..].iter().all(|frame| *frame == [0.0; 2]));
        assert_eq!(shared.lock().unwrap().finished, [3]);
    }

    #[test]
    fn a_paused_mixer_is_silent_and_resumes_where_it_stopped() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        let ramp: Vec<f32> = (0..BLOCK * 4).map(|i| i as f32).collect();
        play(&shared, 1, mono(RATE, &ramp), false, [1.0; 2]);
        pull(&mut mixer, BLOCK);

        send(&shared, Command::Pause(true));
        assert!(
            pull(&mut mixer, BLOCK)
                .iter()
                .all(|frame| *frame == [0.0; 2])
        );
        send(&shared, Command::Pause(false));
        assert_eq!(pull(&mut mixer, 1)[0], [BLOCK as f32; 2]);
    }

    #[test]
    fn a_spatial_voice_hears_a_stereo_clip_as_one_point() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        send(
            &shared,
            Command::Play {
                id: 1,
                body: Body::Clip {
                    clip: Arc::new(Clip::new(2, RATE, [1.0, 0.0].repeat(8).into())),
                    looping: false,
                },
                gains: [1.0, 0.25],
                downmix: true,
            },
        );
        assert_eq!(pull(&mut mixer, 1)[0], [0.5, 0.125]);
    }

    #[test]
    fn a_stream_plays_until_it_runs_out() {
        let shared = shared();
        let mut mixer = Mixer::new(shared.clone());
        let samples: Vec<f32> = (0..20).map(|i| i as f32).collect();
        send(
            &shared,
            Command::Play {
                id: 9,
                body: Body::Stream(Box::new(samples.clone().into_iter())),
                gains: [1.0; 2],
                downmix: false,
            },
        );

        let out = pull(&mut mixer, BLOCK * 2);
        for (f, frame) in out[..10].iter().enumerate() {
            assert_eq!(*frame, [samples[2 * f], samples[2 * f + 1]]);
        }
        assert!(out[10..].iter().all(|frame| *frame == [0.0; 2]));
        assert_eq!(shared.lock().unwrap().finished, [9]);
    }
}
