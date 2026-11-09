//! Reading and decoding sound files off the main thread.
//!
//! A WAV is a copy, but an Ogg or an MP3 of a few seconds is milliseconds of
//! decode, and a music file is megabytes of read before its first sample, so
//! neither happens in a frame. One thread does both, started the first time
//! something asks, and stays out of the frame's pool for the reason the import
//! threads do (see `threads`): a long decode queued on a rayon worker would put
//! every parallel step of the frame behind it.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use orrin_ecs::FxHashMap;
use orrin_registry::AssetId;

use super::clip::{self, Clip};
use super::mixer::Stream;
use crate::asset_db::AssetDatabase;

#[derive(Clone)]
pub enum ClipState {
    Loading,
    Ready(Arc<Clip>),
    Failed,
}

/// A file opened for a voice, which `voice` is waiting on.
pub struct Opened {
    pub voice: u64,
    pub stream: Option<Stream>,
}

enum Job {
    Decode {
        id: AssetId,
        name: String,
        file: PathBuf,
    },
    Open {
        voice: u64,
        name: String,
        file: PathBuf,
        looping: bool,
    },
}

enum Done {
    Decoded(AssetId, Option<Clip>),
    Opened(Opened),
}

#[derive(Default)]
pub struct Clips {
    states: FxHashMap<AssetId, ClipState>,
    jobs: Option<Sender<Job>>,
    done: Option<Receiver<Done>>,
}

impl Clips {
    pub fn state(&self, id: AssetId) -> Option<&ClipState> {
        self.states.get(&id)
    }

    /// Start decoding sound `id`, unless it is decoded or decoding already.
    pub fn request(&mut self, id: AssetId, db: Option<&AssetDatabase>) {
        if !self.states.contains_key(&id) {
            self.decode(id, db);
        }
    }

    /// Decode `id` again, if anything has asked for it: its file changed.
    /// What is playing keeps the samples it started with.
    pub fn changed(&mut self, id: AssetId, db: &AssetDatabase) {
        if self.states.contains_key(&id) {
            self.decode(id, Some(db));
        }
    }

    fn decode(&mut self, id: AssetId, db: Option<&AssetDatabase>) {
        let job = located(id, db).map(|(name, file)| Job::Decode { id, name, file });
        let state = match job.and_then(|job| self.send(job)) {
            Ok(()) => ClipState::Loading,
            Err(error) => {
                tracing::error!("{error}");
                ClipState::Failed
            }
        };
        self.states.insert(id, state);
    }

    /// Open sound `id` to be decoded as `voice` plays it. False, having said
    /// why, if it cannot even be asked for.
    pub fn open(
        &mut self,
        voice: u64,
        id: AssetId,
        looping: bool,
        db: Option<&AssetDatabase>,
    ) -> bool {
        let job = located(id, db).map(|(name, file)| Job::Open {
            voice,
            name,
            file,
            looping,
        });
        match job.and_then(|job| self.send(job)) {
            Ok(()) => true,
            Err(error) => {
                tracing::error!("{error}");
                false
            }
        }
    }

    /// Everything decoded since the last call goes into the table; the files
    /// opened for voices are handed back.
    pub fn receive(&mut self, opened: &mut Vec<Opened>) {
        let Some(done) = &self.done else {
            return;
        };
        while let Ok(result) = done.try_recv() {
            match result {
                Done::Decoded(id, clip) => {
                    let state =
                        clip.map_or(ClipState::Failed, |clip| ClipState::Ready(Arc::new(clip)));
                    self.states.insert(id, state);
                }
                Done::Opened(file) => opened.push(file),
            }
        }
    }

    pub fn insert(&mut self, id: AssetId, clip: Clip) {
        self.states.insert(id, ClipState::Ready(Arc::new(clip)));
    }

    fn send(&mut self, job: Job) -> Result<(), String> {
        if self.jobs.is_none() {
            let (jobs, inbox) = channel();
            let (outbox, done) = channel();
            std::thread::Builder::new()
                .name("orrin-audio-load".to_owned())
                .spawn(move || run(inbox, outbox))
                .map_err(|error| format!("the sound loading thread did not start: {error}"))?;
            self.jobs = Some(jobs);
            self.done = Some(done);
        }
        let jobs = self.jobs.as_ref().expect("started above");
        jobs.send(job)
            .map_err(|_| "the sound loading thread has stopped".to_owned())
    }
}

/// Sound `id`'s name, for errors, and its file.
fn located(id: AssetId, db: Option<&AssetDatabase>) -> Result<(String, PathBuf), String> {
    let db = db.ok_or_else(|| format!("sound {id} cannot play: there is no asset database"))?;
    let file = db.audio_file(id)?;
    let name = db
        .get(id)
        .map_or_else(|| id.to_string(), |entry| entry.path.clone());
    Ok((name, file))
}

fn run(jobs: Receiver<Job>, done: Sender<Done>) {
    let read = |name: &str, file: &PathBuf| {
        let extension = file
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        std::fs::read(file)
            .map(|bytes| (bytes, extension))
            .map_err(|error| format!("sound `{name}` could not be read: {error}"))
    };
    for job in jobs {
        let result = match job {
            Job::Decode { id, name, file } => {
                let clip = read(&name, &file).and_then(|(bytes, extension)| {
                    Clip::decode(bytes, &extension)
                        .map_err(|error| format!("sound `{name}` did not decode: {error}"))
                });
                Done::Decoded(id, clip.map_err(|error| tracing::error!("{error}")).ok())
            }
            Job::Open {
                voice,
                name,
                file,
                looping,
            } => {
                let stream = read(&name, &file).and_then(|(bytes, extension)| {
                    clip::stream(bytes, &extension, looping)
                        .map_err(|error| format!("sound `{name}` did not open: {error}"))
                });
                Done::Opened(Opened {
                    voice,
                    stream: stream.map_err(|error| tracing::error!("{error}")).ok(),
                })
            }
        };
        if done.send(result).is_err() {
            return;
        }
    }
}
