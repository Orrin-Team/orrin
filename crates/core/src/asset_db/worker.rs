//! The thread imports run on.
//!
//! **One thread of its own, not the pool.** The pool is the frame's: it is
//! sized to the machine, and a 40 ms decode parked on one of its workers is a
//! 40 ms hole in whichever parallel sweep lands on it next (see
//! [`threads`](crate::threads)). An import is the opposite kind of work — long,
//! latency-tolerant and mostly waiting on a disk — so it gets a thread that the
//! frame never waits for. Compressing a texture is the exception to "mostly
//! waiting", and is spread across a few threads more for the length of that
//! one job (`texture::encode_chain`), still none of them the pool's.
//!
//! **Two queues.** Everything in the project is imported at startup so the
//! browser can say what failed, but an asset the open scene is drawing should
//! not wait behind a few hundred it is not. A job that brings an artifact back
//! for the GPU goes in `urgent` and is always taken first.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use orrin_registry::AssetId;

use super::cache::{Artifact, Cache, CacheKey, Summary};
use super::import;
use super::sidecar::{AssetKind, Settings};
use super::texture::Target;
use crate::gfx::thumbnail::Thumbnail;
use crate::scene::{MeshPreview, render_mesh};

/// One import to run.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: AssetId,
    pub kind: AssetKind,
    pub source: PathBuf,
    /// Relative to the assets directory, for messages.
    pub rel: String,
    pub settings: Settings,
    /// What the device samples, which a texture's artifact is made for.
    pub target: Target,
    /// Whether to bring the artifact back, for an asset something is drawing.
    pub payload: bool,
    /// The artifact the asset's slot already holds, which is not brought back
    /// again.
    pub known: Option<CacheKey>,
    /// Import even when the cache holds the artifact: a Reimport.
    pub fresh: bool,
}

/// An artifact on its way to the GPU, with anything derived from it that is
/// cheaper made here than on the frame's thread.
pub enum Payload {
    Texture(super::cache::TextureArtifact),
    Mesh(crate::scene::CpuMesh, MeshPreview),
}

/// A job that went through.
pub struct Imported {
    pub key: CacheKey,
    /// Whether the artifact was already in the cache.
    pub cached: bool,
    /// The files read besides the source, relative to the source's directory.
    pub inputs: Vec<String>,
    pub payload: Option<Payload>,
    /// The asset browser's picture of it, stored beside the artifact.
    pub thumbnail: Option<Thumbnail>,
    /// What the artifact is, for the Import page.
    pub summary: Option<Summary>,
}

/// A job's outcome, sent back to the database.
pub struct Done {
    pub job: Job,
    pub outcome: Result<Imported, String>,
}

/// Run `job` against `cache`: read the inputs, hash them, and import only when
/// the cache has no artifact under that key.
///
/// An artifact nothing is waiting for is only checked to exist, so a startup
/// with everything cached reads sources and no artifacts. One that is in the
/// cache but does not read is rebuilt when it is brought back, rather than
/// trusted or reported, since a damaged cache file is never the user's problem.
/// One without its thumbnail — a cache written before there were any — is
/// imported again to make one.
pub fn run(job: &Job, cache: &Cache) -> Result<Imported, String> {
    let Some(importer) = import::importer(job.kind) else {
        return Err(format!(
            "{} is a {}, which is not imported",
            job.rel,
            job.kind.label()
        ));
    };
    let inputs = import::read_inputs(job.kind, &job.source, &job.rel)?;
    // The source is hashed under a fixed name, so renaming or moving a file
    // finds the artifact it already had; the files it reads keep their own
    // names, which are what tell one buffer from another.
    let named: Vec<(&str, &[u8])> = inputs
        .iter()
        .enumerate()
        .map(|(index, (name, bytes))| {
            (
                if index == 0 { "" } else { name.as_str() },
                bytes.as_slice(),
            )
        })
        .collect();
    let settings = import::key_settings(job.kind, &job.settings, job.target);
    let key = super::cache::key(importer, &settings, &named);
    let read = || {
        inputs
            .iter()
            .skip(1)
            .map(|(name, _)| name.clone())
            .collect()
    };

    let wanted = job.payload && job.known != Some(key);
    let pictured = if job.fresh {
        None
    } else {
        cache.load_thumbnail(key)
    };
    if pictured.is_some() && !wanted && cache.path(key).is_file() {
        return Ok(Imported {
            key,
            cached: true,
            inputs: read(),
            payload: None,
            thumbnail: pictured,
            summary: cache.summary(key),
        });
    }

    let found = pictured.is_some().then(|| cache.load(key)).flatten();
    let (artifact, cached, thumbnail) = match found {
        Some(Ok(artifact)) => (artifact, true, pictured),
        Some(Err(_)) | None => {
            let (artifact, thumbnail) = import::import_with_thumbnail(
                job.kind,
                &job.rel,
                &job.settings,
                job.target,
                &inputs,
            )?;
            let stored = cache.store(key, &artifact).and_then(|()| match &thumbnail {
                Some(thumbnail) => cache.store_thumbnail(key, thumbnail),
                None => Ok(()),
            });
            if let Err(error) = stored {
                // Imported fine and only the write failed: the asset still
                // loads, and the next run imports it again.
                tracing::warn!("{}: {error}", job.rel);
            }
            (artifact, false, thumbnail)
        }
    };
    let summary = cache.summary(key);

    let payload = wanted.then(|| match artifact {
        Artifact::Texture(texture) => Payload::Texture(texture),
        Artifact::Mesh(mesh) => {
            let preview = render_mesh(&mesh);
            Payload::Mesh(mesh, preview)
        }
    });
    Ok(Imported {
        key,
        cached,
        inputs: read(),
        payload,
        thumbnail,
        summary,
    })
}

#[derive(Default)]
struct Queue {
    urgent: VecDeque<Job>,
    background: VecDeque<Job>,
    /// Queued plus running: what [`Worker::wait_idle`] waits on.
    busy: usize,
    closed: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    changed: Condvar,
}

/// The import thread and the two ends of its conversation.
pub struct Worker {
    shared: Arc<Shared>,
    results: Receiver<Done>,
}

impl Worker {
    pub fn spawn(cache: Cache) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue::default()),
            changed: Condvar::new(),
        });
        let (sender, results) = mpsc::channel();
        let thread_shared = shared.clone();
        std::thread::Builder::new()
            .name("orrin-import".to_owned())
            .spawn(move || serve(&thread_shared, &cache, &sender))
            .map_err(|error| format!("could not start the import thread: {error}"))?;
        Ok(Self { shared, results })
    }

    pub fn push(&self, job: Job) {
        let mut queue = self.shared.queue.lock().unwrap();
        queue.busy += 1;
        if job.payload {
            queue.urgent.push_back(job);
        } else {
            queue.background.push_back(job);
        }
        self.shared.changed.notify_all();
    }

    /// Every job finished since the last call.
    pub fn finished(&self) -> impl Iterator<Item = Done> + '_ {
        self.results.try_iter()
    }

    /// Whether anything is queued or running.
    pub fn is_busy(&self) -> bool {
        self.shared.queue.lock().unwrap().busy > 0
    }

    /// Block until the queue is empty, or `timeout` passes. `false` on timeout.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let queue = self.shared.queue.lock().unwrap();
        let (queue, result) = self
            .shared
            .changed
            .wait_timeout_while(queue, timeout, |queue| queue.busy > 0)
            .unwrap();
        drop(queue);
        !result.timed_out()
    }
}

impl Drop for Worker {
    /// Lets the thread finish the job it is on and exit, without waiting for
    /// it: shutting the editor should not wait out a decode.
    fn drop(&mut self) {
        self.shared.queue.lock().unwrap().closed = true;
        self.shared.changed.notify_all();
    }
}

fn serve(shared: &Shared, cache: &Cache, results: &Sender<Done>) {
    loop {
        let job = {
            let mut queue = shared.queue.lock().unwrap();
            loop {
                if queue.closed {
                    return;
                }
                if let Some(job) = queue
                    .urgent
                    .pop_front()
                    .or_else(|| queue.background.pop_front())
                {
                    break job;
                }
                queue = shared.changed.wait(queue).unwrap();
            }
        };
        let outcome = guarded(&job.rel, || run(&job, cache));
        // Sent before `busy` drops, so a caller that saw the queue go idle
        // finds every result already waiting.
        let gone = results.send(Done { job, outcome }).is_err();
        let mut queue = shared.queue.lock().unwrap();
        queue.busy -= 1;
        shared.changed.notify_all();
        if gone {
            return;
        }
    }
}

/// `run`, with a panic in it turned into a failed import naming the file.
///
/// The thread serves every import the session makes, so one file that trips a
/// bug in a decoder or an encoder must fail alone: if it took the thread with
/// it, every later import would wait forever and the browser would say
/// "importing" until the editor closed.
fn guarded(rel: &str, run: impl FnOnce() -> Result<Imported, String>) -> Result<Imported, String> {
    // Unwind-safe in fact: the job and the cache are only read, and the cache
    // is written through a temporary file and a rename, so a panic part-way
    // leaves nothing a later import would trust.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "with no message".to_owned());
        Err(format!(
            "{rel} could not be imported: the importer panicked ({message}). This is an \
             engine bug; the file is worth keeping for a report"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicking_import_fails_by_name_instead_of_ending_the_thread() {
        let Err(err) = guarded("rocks/rock.png", || panic!("seed table overflow")) else {
            panic!("a panic is a failed import");
        };
        assert_eq!(
            err,
            "rocks/rock.png could not be imported: the importer panicked (seed table \
             overflow). This is an engine bug; the file is worth keeping for a report"
        );
    }
}
