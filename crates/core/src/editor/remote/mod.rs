//! The editor's remote: a loopback listener that answers a client's requests
//! against the session the developer has open — the world, the registry, the
//! asset index, the console and `EditorState`. `orrin mcp` is the client an
//! agent reaches it through. The protocol is `orrin_remote`'s, and its header
//! is the wire's one description; this is the server.
//!
//! # What a connection may do
//!
//! **Read, today.** Every method answers from what the frame's thread already
//! holds and changes nothing, not even an entity's lack of an
//! [`EntityId`](orrin_registry::EntityId):
//! an entity that has none is named by its slot, `#12`, which holds for this
//! session only, rather than given one the way an edit or a save would. Which
//! world was read — the scene being edited or the fork a playing game runs
//! on — is in every answer that reads one ([`answer`] lists the methods).
//!
//! **Only once asked, and only with the token.** The remote is off unless
//! Settings ▸ Agents turns it on or `ORRIN_REMOTE=1` does, and it needs a
//! project: the port and token go to its `.orrin/remote`, owner-readable, and
//! the file goes when the remote stops. A peer that does not open with the
//! magic and that token is dropped having been told nothing. A connection's
//! thread that panicked would end that connection alone.
//!
//! # Threads
//!
//! The listener and each connection are plain OS threads, as the live link's
//! are ([`link`](crate::asset_db::link)), never the frame's pool. A
//! connection has two: one reads and decodes, and one writes, so the frame's
//! thread hands an answer over without ever waiting on a socket. A request
//! that does not decode is answered from the reading thread and never reaches
//! the frame.
//!
//! **Shutdown** is the link's: every socket is shut down, which returns its
//! reader's `read` and fails its writer's `write`, the listener is connected
//! to once to return its `accept`, and all of them are joined.
//!
//! # In the frame
//!
//! **One point.** [`Remote::serve`] runs once a frame from `app.rs`, after the
//! console drain — so `console_read` sees this frame's lines — and before the
//! editor's UI, whether or not the overlay is drawn. Off it costs the check of
//! an `Option`; on and idle, one atomic load, and neither allocates.
//!
//! **A budget.** At most [`BUDGET`] requests are answered a frame, oldest
//! first; the rest wait for the next. A flood costs latency, not frame time,
//! and past [`QUEUED`] waiting a request is refused as busy rather than held.
//!
//! **Nothing wakes the loop, because nothing has to.** The event loop runs on
//! `ControlFlow::Poll` and asks for a redraw at every `about_to_wait`, and
//! winit delivers that redraw to an unfocused or covered window too, so a
//! request is answered within a frame whether or not the editor has focus.
//! Should the loop ever learn to sleep, a request will have to wake it.
//!
//! An exported game has no remote: [`Remote::start`] refuses in one, and the
//! app only offers it to the editor.

mod answer;

use std::collections::VecDeque;
use std::io::Write;
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use orrin_ecs::{FxHashMap, World};
use orrin_registry::Registry;
use orrin_remote::wire::{self, Hello, Request, Response};
use orrin_remote::{Discovery, Token};

use super::state::EditorState;

/// How long a peer has to send its 44-byte preamble.
const HANDSHAKE: Duration = Duration::from_secs(5);

/// How long one frame may take to leave. A client that stops reading holds a
/// writer thread no longer than this, never the frame's thread.
const WRITE: Duration = Duration::from_secs(5);

/// Requests answered in one frame. A read of a large scene is a walk of the
/// world, so this bounds what a client can take from one frame.
pub const BUDGET: usize = 8;

/// Requests waiting, across every connection, before another is refused.
pub const QUEUED: usize = 256;

/// What the remote is doing, for Settings ▸ Agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteStatus {
    pub port: u16,
    /// Connections past the handshake.
    pub peers: usize,
}

type Peer = u64;

struct Waiting {
    peer: Peer,
    request: Request,
}

struct Connection {
    /// A handle on the socket, for shutting it down from outside.
    socket: TcpStream,
    reader: Option<JoinHandle<()>>,
    /// Frames for the writer thread; `None` until the handshake is through.
    send: Option<Sender<Vec<u8>>>,
}

#[derive(Default)]
struct Connections {
    next: Peer,
    open: FxHashMap<Peer, Connection>,
    /// Set under this lock, so the listener cannot add a connection that the
    /// shutdown has already passed over.
    stopped: bool,
}

struct Shared {
    stop: AtomicBool,
    /// Whether the queue holds anything: the one load an idle frame pays.
    /// Written only under the queue's lock, so it cannot miss a request.
    news: AtomicBool,
    queue: Mutex<VecDeque<Waiting>>,
    connections: Mutex<Connections>,
    peers: AtomicUsize,
    token: Token,
    hello: Vec<u8>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Every section under these locks is a few moves that cannot panic; a
    // poisoned one would only mean a thread died elsewhere holding it.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn send(&self, peer: Peer, frame: Vec<u8>) {
        let connections = lock(&self.connections);
        if let Some(send) = connections.open.get(&peer).and_then(|c| c.send.as_ref()) {
            let _ = send.send(frame);
        }
    }

    /// Queue a request for the frame, or say why it was not.
    fn post(&self, peer: Peer, request: Request) -> Result<(), Request> {
        let mut queue = lock(&self.queue);
        if queue.len() >= QUEUED {
            return Err(request);
        }
        queue.push_back(Waiting { peer, request });
        self.news.store(true, Ordering::Release);
        Ok(())
    }
}

/// The remote while it listens.
pub struct Remote {
    shared: Arc<Shared>,
    addr: SocketAddr,
    listener: Option<JoinHandle<()>>,
    discovery: Discovery,
    editor_dir: PathBuf,
    /// This frame's requests, kept for its allocation.
    batch: Vec<Waiting>,
}

impl Remote {
    /// Listen on a port of `127.0.0.1` the OS picks, and say where in
    /// `editor_dir`'s remote file. `project` is the name a client is greeted
    /// with.
    pub fn start(editor_dir: &Path, project: &str) -> Result<Self, String> {
        if cfg!(feature = "export") {
            return Err("an exported game has no remote".to_owned());
        }
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| format!("the remote could not listen on 127.0.0.1: {error}"))?;
        let addr = listener
            .local_addr()
            .map_err(|error| format!("the remote's port could not be read: {error}"))?;
        let pid = std::process::id();
        let hello = wire::encode(
            wire::HELLO,
            &Hello {
                version: wire::VERSION,
                project: project.to_owned(),
                pid,
            },
        )?;
        let token = Token::generate();
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            news: AtomicBool::new(false),
            queue: Mutex::new(VecDeque::new()),
            connections: Mutex::new(Connections::default()),
            peers: AtomicUsize::new(0),
            token: token.clone(),
            hello,
        });
        let listening = shared.clone();
        let thread = std::thread::Builder::new()
            .name("orrin-remote".to_owned())
            .spawn(move || listen(&listening, &listener))
            .map_err(|error| format!("could not start the remote's thread: {error}"))?;
        let mut remote = Self {
            shared,
            addr,
            listener: Some(thread),
            discovery: Discovery {
                port: addr.port(),
                token,
                pid,
            },
            editor_dir: editor_dir.to_owned(),
            batch: Vec::new(),
        };
        // Written once something is listening, so a client that reads it can
        // connect; a remote nobody can find is no remote.
        if let Err(error) = remote.discovery.write(editor_dir) {
            remote.stop();
            return Err(error);
        }
        Ok(remote)
    }

    pub fn status(&self) -> RemoteStatus {
        RemoteStatus {
            port: self.addr.port(),
            peers: self.shared.peers.load(Ordering::Acquire),
        }
    }

    /// Answer up to [`BUDGET`] of the requests waiting, oldest first.
    pub fn serve(&mut self, world: &mut World, registry: &Registry, state: &mut EditorState) {
        if !self.shared.news.load(Ordering::Acquire) {
            return;
        }
        crate::profile_scope!("remote");
        let Self { shared, batch, .. } = self;
        {
            let mut queue = lock(&shared.queue);
            let take = queue.len().min(BUDGET);
            batch.extend(queue.drain(..take));
            shared.news.store(!queue.is_empty(), Ordering::Release);
        }
        for Waiting { peer, request } in batch.drain(..) {
            let answer = answer::answer(&request, world, registry, state);
            let response = Response::answer(Some(request.id), answer);
            let frame = wire::encode(wire::RESPONSE, &response).unwrap_or_else(|why| {
                let refused = Response::answer(
                    Some(request.id),
                    Err(format!(
                        "the answer to `{}` is too large to send ({why}); ask for less, with \
                         `limit`",
                        request.method
                    )),
                );
                wire::encode(wire::RESPONSE, &refused).unwrap_or_default()
            });
            shared.send(peer, frame);
        }
    }

    /// Stop every thread, wait for them, and take the remote file back.
    fn stop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        let readers: Vec<JoinHandle<()>> = {
            let mut connections = lock(&self.shared.connections);
            connections.stopped = true;
            connections
                .open
                .values_mut()
                .filter_map(|connection| {
                    let _ = connection.socket.shutdown(Shutdown::Both);
                    connection.reader.take()
                })
                .collect()
        };
        if let Some(listener) = self.listener.take() {
            // Returns the listener's `accept`. Should the backlog be full, it
            // returns anyway, with one of the connections already waiting.
            let _ = TcpStream::connect_timeout(&self.addr, Duration::from_secs(1));
            let _ = listener.join();
        }
        for reader in readers {
            let _ = reader.join();
        }
        self.discovery.remove(&self.editor_dir);
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.stop();
    }
}

fn listen(shared: &Arc<Shared>, listener: &TcpListener) {
    for stream in listener.incoming() {
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        let Ok(stream) = stream else {
            // Out of descriptors, most likely; waiting is what frees one.
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        let _ = stream.set_nodelay(true);
        let Ok(socket) = stream.try_clone() else {
            continue;
        };
        let mut connections = lock(&shared.connections);
        if connections.stopped {
            let _ = socket.shutdown(Shutdown::Both);
            return;
        }
        let peer = connections.next;
        connections.next += 1;
        let reading = shared.clone();
        let reader = std::thread::Builder::new()
            .name("orrin-remote-in".to_owned())
            .spawn(move || connect(&reading, peer, stream));
        match reader {
            Ok(reader) => {
                connections.open.insert(
                    peer,
                    Connection {
                        socket,
                        reader: Some(reader),
                        send: None,
                    },
                );
            }
            Err(_) => {
                let _ = socket.shutdown(Shutdown::Both);
            }
        }
    }
}

/// One connection's reading thread, from accept to close, after which its
/// requests still waiting are dropped: there is nobody to answer.
fn connect(shared: &Shared, peer: Peer, stream: TcpStream) {
    let addr = stream.peer_addr().ok();
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(shared, peer, stream)));
    let greeted = lock(&shared.connections)
        .open
        .remove(&peer)
        .is_some_and(|connection| connection.send.is_some());
    let from = addr.map_or_else(String::new, |addr| format!(" from {addr}"));
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(why)) => tracing::warn!("{why}{from}"),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "with no message".to_owned());
            tracing::error!("a remote connection{from} ended on an engine bug ({message})");
        }
    }
    if greeted {
        lock(&shared.queue).retain(|waiting| waiting.peer != peer);
        shared.peers.fetch_sub(1, Ordering::AcqRel);
        tracing::info!("an agent disconnected from the remote{from}");
    }
}

/// The handshake, then every frame until the connection closes. `Err` for a
/// connection ended on a protocol error, to be warned about.
fn serve(shared: &Shared, peer: Peer, mut stream: TcpStream) -> Result<(), String> {
    let _ = stream.set_read_timeout(Some(HANDSHAKE));
    let _ = stream.set_write_timeout(Some(WRITE));
    match wire::opening(&mut stream) {
        wire::Opening::Ours(token) if token.matches(&shared.token) => {}
        wire::Opening::Ours(_) => {
            tracing::warn!(
                "a connection to the remote offered a token that is not this editor's and was \
                 dropped; a client reading another session's remote file would do that"
            );
            return Ok(());
        }
        wire::Opening::Stranger => return Ok(()),
        wire::Opening::Version(theirs) => {
            let message = format!(
                "this client speaks remote protocol {theirs} and the editor speaks {}; use the \
                 `orrin` that came with this editor",
                wire::VERSION
            );
            if let Ok(frame) = wire::frame(wire::ERROR, message.as_bytes()) {
                let _ = stream.write_all(&frame);
            }
            return Ok(());
        }
    }
    let _ = stream.set_read_timeout(None);

    let (send, outgoing) = mpsc::channel::<Vec<u8>>();
    let mut out = stream
        .try_clone()
        .map_err(|error| format!("the remote could not answer a connection: {error}"))?;
    std::thread::Builder::new()
        .name("orrin-remote-out".to_owned())
        .spawn(move || {
            for frame in outgoing {
                if out.write_all(&frame).is_err() {
                    let _ = out.shutdown(Shutdown::Both);
                    return;
                }
            }
        })
        .map_err(|error| format!("could not start a remote thread: {error}"))?;
    let _ = send.send(shared.hello.clone());
    {
        let mut connections = lock(&shared.connections);
        match connections.open.get_mut(&peer) {
            Some(connection) => connection.send = Some(send),
            // The remote is stopping, and has already shut this socket down.
            None => return Ok(()),
        }
    }
    shared.peers.fetch_add(1, Ordering::AcqRel);
    match stream.peer_addr() {
        Ok(addr) => tracing::info!("an agent connected to the remote from {addr}"),
        Err(_) => tracing::info!("an agent connected to the remote"),
    }

    let mut payload = Vec::new();
    loop {
        let kind = match wire::read_frame(&mut stream, &mut payload) {
            Ok(kind) => kind,
            Err(wire::ReadError::Closed) => return Ok(()),
            Err(wire::ReadError::TooLarge(len)) => {
                let why = format!(
                    "a frame of {len} bytes is over the remote's limit of {}, so the connection \
                     was closed",
                    wire::MAX_FRAME
                );
                if let Ok(frame) = wire::frame(wire::ERROR, why.as_bytes()) {
                    shared.send(peer, frame);
                }
                return Err(why);
            }
        };
        if kind != wire::REQUEST {
            // A kind added since this engine, skipped by its length.
            continue;
        }
        let refused = match wire::decode_request(&payload) {
            Ok(request) => match shared.post(peer, request) {
                Ok(()) => continue,
                Err(request) => Response::answer(
                    Some(request.id),
                    Err(format!(
                        "the editor is busy: {QUEUED} requests are already waiting; try again"
                    )),
                ),
            },
            Err(unreadable) => Response::answer(unreadable.id, Err(unreadable.reason)),
        };
        if let Ok(frame) = wire::encode(wire::RESPONSE, &refused) {
            shared.send(peer, frame);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{ErrorKind, Read};

    use super::*;
    use crate::asset_db::tests::Dir;

    /// Whether the editor has closed its end, having sent nothing at all.
    fn closed_unanswered(mut stream: TcpStream) -> bool {
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut rest = Vec::new();
        let read = stream.read_to_end(&mut rest);
        rest.is_empty()
            && match read {
                Ok(_) => true,
                Err(error) => error.kind() == ErrorKind::ConnectionReset,
            }
    }

    fn raw(port: u16, opening: &[u8]) -> TcpStream {
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        stream.write_all(opening).unwrap();
        stream
    }

    /// A browser, a wrong token and a preamble cut short are each dropped
    /// having been told nothing, and none of them counts as connected; the
    /// token from the remote file, which its owner alone can read, is greeted.
    #[test]
    fn only_the_token_in_the_remote_file_is_answered() {
        let dir = Dir::new("remote-token");
        let remote = Remote::start(dir.path(), "game").unwrap();
        let found = Discovery::read(dir.path()).unwrap().expect("a remote file");
        assert_eq!(found.port, remote.status().port);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let file = orrin_remote::discovery::path(dir.path());
            let mode = std::fs::metadata(file).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        let http = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert!(closed_unanswered(raw(found.port, http)), "a browser");
        let wrong = wire::preamble(&Token::generate());
        assert!(closed_unanswered(raw(found.port, &wrong)), "a wrong token");
        let right = wire::preamble(&found.token);
        let cut = raw(found.port, &right[..right.len() - 1]);
        cut.shutdown(Shutdown::Write).unwrap();
        assert!(closed_unanswered(cut), "a token cut short");
        assert_eq!(remote.status().peers, 0);

        let mut client = raw(found.port, &right);
        let mut payload = Vec::new();
        assert_eq!(wire::read_frame(&mut client, &mut payload), Ok(wire::HELLO));
        assert_eq!(
            wire::decode_hello(&payload).unwrap().project,
            "game",
            "greeted"
        );

        drop(remote);
        assert!(
            Discovery::read(dir.path()).unwrap().is_none(),
            "the file goes with the remote"
        );
    }
}
