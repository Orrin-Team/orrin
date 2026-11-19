//! The live link: a Blender add-on streams the mesh it is editing into the
//! slot its asset already has, between frames and without a save — the shape
//! of Unity's MeshSync and Unreal's Direct Link. A vertex dragged in Blender
//! shows in the viewport a frame or two later. The file stays the truth: a
//! save reimports through the watcher and the `.blend` importer as any save
//! does, and when the link lets go the slot goes back to what is on disk.
//!
//! # What a connection may do
//!
//! **Loopback only.** The listener binds `127.0.0.1`, never a wildcard
//! address, so nothing off this machine can reach it. A port that cannot be
//! bound is an error returned to whoever asked, and nothing else in the
//! database changes.
//!
//! **It can only replace what an existing mesh slot draws.** A connection
//! never writes a file, creates an asset or touches a sidecar. It names an
//! asset by its path relative to the assets directory, and that string is only
//! ever compared against the index ([`find`](super::AssetDatabase::find)),
//! never joined onto the file system — which is why a hostile path such as
//! `../../.ssh/id_ed25519` is merely an asset the project does not have.
//!
//! **Its bytes are untrusted.** A length over [`MAX_FRAME`] is refused from the
//! header, before anything is allocated for it, and ends the connection, since
//! the stream is then out of step. A mesh whose counts do not account for its
//! payload exactly, whose index count is not a multiple of three, which indexes
//! past its vertices, which has no triangles, or which carries a number that is
//! not finite is refused with an error frame naming the path and the reason,
//! and changes nothing. Decoding is total — no input panics it — and a
//! connection's thread that panicked anyway would end that connection alone,
//! reported, with the editor and every other connection unaffected.
//!
//! # The wire protocol
//!
//! Version [`VERSION`], on TCP to `127.0.0.1`, port [`DEFAULT_PORT`] unless the
//! editor's Settings ▸ Blender says otherwise. Every integer is little-endian,
//! every float an IEEE `f32`, and a *string* is a `u32` byte length followed by
//! that many bytes of UTF-8, unterminated.
//!
//! **Handshake.** The client speaks first, with the 8 bytes `ORRNLINK`
//! ([`MAGIC`]) and then its protocol version as a `u32` — 12 bytes, not a frame.
//! The engine reads the magic alone first, and a peer that opens with anything
//! else (a browser, a port scanner) is dropped having been told nothing; so is
//! one that has not sent the 12 bytes within five seconds. A peer of another
//! version is sent one `error` frame saying to update the add-on from
//! Settings ▸ Blender, then dropped. Otherwise the engine answers with a
//! `hello` frame, and from then on either side may send frames.
//!
//! **Frames.** A `u32` payload length, a `u8` kind, then the payload. A length
//! over [`MAX_FRAME`] (256 MiB) is a protocol error that ends the connection.
//! A peer that receives a kind it does not know skips that many bytes and
//! carries on, which is how kinds are added without a new version.
//!
//! | kind | name | from | payload |
//! | --- | --- | --- | --- |
//! | 1 | `hello` | engine | `u32` version; string: the assets directory |
//! | 2 | `error` | engine | string: asset path, empty when none applies; string: message |
//! | 3 | `live` | engine | string: asset path |
//! | 4 | `mesh` | client | string: asset path; `u32` V; `u32` I; then `3V` positions, `3V` normals and `2V` UVs as `f32`; then `I` indices as `u32` |
//! | 5 | `release` | client | string: asset path |
//!
//! The assets directory in `hello` is absolute but not resolved through links,
//! so a client resolves both it and its own file's path before comparing them.
//! An asset path is relative to that directory with `/` separators, as the
//! index has it — `models/tree.blend`.
//!
//! A `mesh` frame's payload is exactly `4 + path + 8 + 32V + 4I` bytes. Its
//! three vertex arrays are each contiguous rather than interleaved, so a client
//! fills them straight from numpy. Its geometry is what [`blend::EXPORT`]
//! would export of the file, so the stream and the saved file agree: every
//! object visible in the viewport with its modifiers applied, triangulated,
//! merged into one mesh in world space, in glTF's conventions — `+Y` up,
//! metres, UVs with their origin where a `.glb` has it — and split per face
//! corner wherever a normal or a UV differs, since a vertex here carries one of
//! each. The sidecar's `scale` is the engine's to apply, as it is for the file.
//!
//! `live` says that this connection's stream of that asset is now what its slot
//! draws; `error` says why a mesh was not. Each is sent once per change for a
//! path rather than per mesh, so a client streaming thirty meshes a second into
//! a path the project lacks hears about it once. `release` hands an asset back
//! to its file. A client keeps reading what the engine sends: one that leaves a
//! frame unread for five seconds is disconnected.
//!
//! # Threads
//!
//! The listener and each connection are plain OS threads, as the watcher's and
//! the import thread are, never the frame's pool ([`threads`](crate::threads)).
//! A connection has two: one reads and decodes — a 100k-triangle mesh in
//! 1.8 ms on an M5 Pro, tangents included — so the frame's thread receives a
//! finished [`CpuMesh`](crate::scene::CpuMesh), and one writes, so the frame's
//! thread hands a reply over without ever waiting on a socket.
//!
//! **Latest wins.** Between the connections and the database there is one slot
//! per asset path holding the newest decoded mesh, not a queue: a frame's
//! thread that stalls comes back to one mesh per asset rather than a backlog of
//! megabytes, and three meshes arriving within a frame cost one replace.
//!
//! **Shutdown** cannot leave a thread parked. Turning the link off, or dropping
//! the database, shuts every connection's socket down, which returns its
//! reader's blocking `read` and fails its writer's `write`, and connects once to
//! the listener, which returns its blocking `accept` to find the flag set. All
//! of them are then joined, so the port is free once [`set_link`] returns.
//!
//! # In the database
//!
//! **Between frames.** [`update`](super::AssetDatabase::update) takes what has
//! arrived after it has installed finished imports and bound pending handles.
//! With the link off that costs a check of an `Option`, and on and idle one
//! atomic load; neither allocates.
//!
//! **Only a drawn asset goes live.** The path must name an indexed mesh asset,
//! or the mesh is answered with an error. One that nothing in the open scene
//! draws has no slot to put the mesh in, so the mesh is dropped, the client is
//! told, and the asset is not live; the next mesh after something draws it
//! makes it so. A live mesh is scaled by the sidecar's `scale`, so it does not
//! change size on save, and its bounds and collider triangles follow it into
//! [`MeshBounds`](crate::scene::MeshBounds). The asset browser's picture stays
//! the file's.
//!
//! **While an asset is live, the slot is the stream's.** A disk import of it
//! that finishes meanwhile still updates the entry — status, key, thumbnail,
//! summary — but does not touch the slot, because the import of a save lands
//! most of a second after the save, and an edit streamed in between would be
//! overwritten by the older saved state with nothing to bring it back until the
//! next edit. A failed import is reported and leaves the slot alone too: never
//! the pink placeholder over a mesh being edited. A live asset is
//! [`Ready`](super::LoadState::Ready).
//!
//! **Letting go returns to disk.** On a `release`, on the connection closing for
//! any reason, on the link being turned off, and when a live asset is deleted or
//! renamed, the asset stops being live and its slot is queued for the artifact
//! of what is on disk, through the same path any import takes. A live replace
//! forgets the slot's artifact key, which is what stops the install from taking
//! that artifact for the one already drawn. Unsaved geometry never outlives its
//! connection.
//!
//! **Two Blenders.** More than one client may connect, a thread pair each. An
//! asset is held by every connection whose mesh of it has been drawn, the
//! newest mesh is the one drawn, and the asset goes back to its file when the
//! last of them lets go. A release from one connection drops only that
//! connection's mesh still waiting for a frame. Two Blenders editing one file
//! is a conflict the file itself has; this resolves it simply, not well.
//!
//! An exported game has no link: [`set_link`] refuses a pack.
//!
//! [`set_link`]: super::AssetDatabase::set_link
//! [`blend::EXPORT`]: super::blend::EXPORT

mod wire;

use std::io::Write;
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use orrin_ecs::{FxHashMap, World};
use orrin_registry::AssetId;

use super::{AssetDatabase, AssetKind, LoadState, PENDING, Settings, handles};
use crate::geom::Aabb;
use crate::gfx::RenderBackend;
use crate::scene::{CpuMesh, MeshBounds, MeshHandle};

pub use wire::{MAGIC, MAX_FRAME, VERSION};

/// The port the editor listens on unless told otherwise. Unassigned by IANA,
/// and the issue that introduced the link.
pub const DEFAULT_PORT: u16 = 47103;

/// How long a peer has to send its 12-byte preamble.
const HANDSHAKE: Duration = Duration::from_secs(5);

/// How long one frame may take to leave. A client that stops reading holds a
/// writer thread no longer than this, never the frame's thread.
const WRITE: Duration = Duration::from_secs(5);

/// What the link is doing, for a settings page and the console.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkStatus {
    /// The port listened on: the one asked for, or the one the OS chose for 0.
    pub port: u16,
    /// Connections past the handshake.
    pub peers: usize,
    /// The assets drawing a stream, by path, sorted.
    pub live: Vec<String>,
}

type Peer = u64;

/// The newest mesh decoded for a path, and the connection it came from.
struct Arrival {
    peer: Peer,
    mesh: CpuMesh,
    bounds: Aabb,
}

/// What the connections have handed the database since it last looked.
#[derive(Default)]
struct Inbox {
    meshes: FxHashMap<String, Arrival>,
    released: Vec<(Peer, String)>,
    refused: Vec<(Peer, wire::Refusal)>,
    joined: Vec<SocketAddr>,
    gone: Vec<Peer>,
    /// Connections ended on a protocol error or an engine bug, each a warning.
    failed: Vec<String>,
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
    /// Whether the inbox holds anything: the one load an idle frame pays.
    news: AtomicBool,
    inbox: Mutex<Inbox>,
    connections: Mutex<Connections>,
    peers: AtomicUsize,
    /// The assets directory, as `hello` carries it.
    assets: String,
    #[cfg(test)]
    received: std::sync::atomic::AtomicU64,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Every section under these locks is a few moves that cannot panic; a
    // poisoned one would only mean a thread died elsewhere holding it.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    fn post(&self, post: impl FnOnce(&mut Inbox)) {
        let mut inbox = lock(&self.inbox);
        post(&mut inbox);
        self.news.store(true, Ordering::Release);
        #[cfg(test)]
        self.received.fetch_add(1, Ordering::Relaxed);
    }

    fn send(&self, peer: Peer, frame: Vec<u8>) {
        let connections = lock(&self.connections);
        if let Some(send) = connections.open.get(&peer).and_then(|c| c.send.as_ref()) {
            let _ = send.send(frame);
        }
    }
}

/// The link while it listens: what its threads share, and what the frame's
/// thread keeps of which assets are live.
pub(super) struct Link {
    shared: Arc<Shared>,
    addr: SocketAddr,
    listener: Option<JoinHandle<()>>,
    /// Each live asset, by id, with the path it was streamed as and the
    /// connections holding it.
    live: FxHashMap<AssetId, Live>,
    /// What each connection was last told about each path.
    told: FxHashMap<Peer, FxHashMap<String, Told>>,
}

struct Live {
    path: String,
    holders: Vec<Peer>,
}

#[derive(PartialEq)]
enum Told {
    Live,
    /// Refused, with the reason, which is also a console warning.
    Refused(String),
    /// Nothing draws it, which is no fault and not a warning.
    Undrawn,
}

impl Link {
    fn start(port: u16, assets: &std::path::Path) -> Result<Self, String> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port)).map_err(|error| {
            format!(
                "the live link could not listen on 127.0.0.1:{port}: {error}. Another program, \
                 or another editor, may have it; choose another port in Settings ▸ Blender"
            )
        })?;
        let addr = listener
            .local_addr()
            .map_err(|error| format!("the live link's port could not be read: {error}"))?;
        let assets = std::path::absolute(assets).unwrap_or_else(|_| assets.to_owned());
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            news: AtomicBool::new(false),
            inbox: Mutex::new(Inbox::default()),
            connections: Mutex::new(Connections::default()),
            peers: AtomicUsize::new(0),
            assets: assets.to_string_lossy().into_owned(),
            #[cfg(test)]
            received: Default::default(),
        });
        let listening = shared.clone();
        let listener = std::thread::Builder::new()
            .name("orrin-link".to_owned())
            .spawn(move || listen(&listening, &listener))
            .map_err(|error| format!("could not start the live link's thread: {error}"))?;
        Ok(Self {
            shared,
            addr,
            listener: Some(listener),
            live: FxHashMap::default(),
            told: FxHashMap::default(),
        })
    }

    fn is_live(&self, id: AssetId) -> bool {
        self.live.contains_key(&id)
    }

    /// Send `peer` what `told` says about `path`, unless it was the last thing
    /// it was told about it. Whether it was news.
    fn tell(&mut self, peer: Peer, path: &str, told: Told) -> bool {
        let paths = self.told.entry(peer).or_default();
        if paths.get(path) == Some(&told) {
            return false;
        }
        let frame = match &told {
            Told::Live => wire::live(path),
            Told::Refused(reason) => wire::error(path, reason),
            Told::Undrawn => wire::error(
                path,
                "nothing in the editor's open scene draws it, so there is nothing to update yet",
            ),
        };
        paths.insert(path.to_owned(), told);
        self.shared.send(peer, frame);
        true
    }

    /// Stop every thread and wait for them, which [`Drop`] does too.
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
    }
}

impl Drop for Link {
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
            .name("orrin-link-in".to_owned())
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

/// One connection's reading thread, from accept to close, after which it
/// takes back whatever the connection was still holding.
fn connect(shared: &Shared, peer: Peer, stream: TcpStream) {
    let addr = stream.peer_addr().ok();
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve(shared, peer, stream)));
    let greeted = lock(&shared.connections)
        .open
        .remove(&peer)
        .is_some_and(|connection| connection.send.is_some());
    let failed = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(why)) => Some(why),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "with no message".to_owned());
            Some(format!(
                "the live link's connection ended on an engine bug ({message})"
            ))
        }
    };
    if !greeted && failed.is_none() {
        return;
    }
    shared.post(|inbox| {
        if greeted {
            inbox.meshes.retain(|_, arrival| arrival.peer != peer);
            inbox.released.retain(|(from, _)| *from != peer);
            inbox.refused.retain(|(from, _)| *from != peer);
            inbox.gone.push(peer);
        }
        if let Some(failed) = failed {
            let from = addr.map_or_else(String::new, |addr| format!(" from {addr}"));
            inbox.failed.push(format!("{failed}{from}"));
        }
    });
    if greeted {
        shared.peers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The handshake, then every frame until the connection closes. `Err` for a
/// connection ended on a protocol error, to be warned about.
fn serve(shared: &Shared, peer: Peer, mut stream: TcpStream) -> Result<(), String> {
    let _ = stream.set_read_timeout(Some(HANDSHAKE));
    let _ = stream.set_write_timeout(Some(WRITE));
    match wire::opening(&mut stream) {
        wire::Opening::Ours => {}
        wire::Opening::Stranger => return Ok(()),
        wire::Opening::Version(theirs) => {
            let message = format!(
                "this add-on speaks live link protocol {theirs} and the editor speaks {VERSION}; \
                 update the add-on from the editor's Settings ▸ Blender"
            );
            let _ = stream.write_all(&wire::error("", &message));
            return Ok(());
        }
    }
    let _ = stream.set_read_timeout(None);

    let (send, outgoing) = mpsc::channel::<Vec<u8>>();
    let mut out = stream
        .try_clone()
        .map_err(|error| format!("the live link could not answer a connection: {error}"))?;
    std::thread::Builder::new()
        .name("orrin-link-out".to_owned())
        .spawn(move || {
            for frame in outgoing {
                if out.write_all(&frame).is_err() {
                    let _ = out.shutdown(Shutdown::Both);
                    return;
                }
            }
        })
        .map_err(|error| format!("could not start a live link thread: {error}"))?;
    let _ = send.send(wire::hello(&shared.assets));
    {
        let mut connections = lock(&shared.connections);
        match connections.open.get_mut(&peer) {
            Some(connection) => connection.send = Some(send),
            // The link is stopping, and has already shut this socket down.
            None => return Ok(()),
        }
    }
    if let Ok(addr) = stream.peer_addr() {
        shared.post(|inbox| inbox.joined.push(addr));
    }
    // Counted after it is posted, so whoever sees the count finds the post.
    shared.peers.fetch_add(1, Ordering::AcqRel);

    let mut payload = Vec::new();
    loop {
        let kind = match wire::read_frame(&mut stream, &mut payload) {
            Ok(kind) => kind,
            Err(wire::ReadError::Closed) => return Ok(()),
            Err(wire::ReadError::TooLarge(len)) => {
                let why = format!(
                    "a frame of {len} bytes is over the live link's limit of {MAX_FRAME}, so the \
                     connection was closed"
                );
                shared.send(peer, wire::error("", &why));
                return Err(why);
            }
        };
        match kind {
            wire::MESH => match wire::decode_mesh(&payload) {
                Ok(decoded) => {
                    let arrival = Arrival {
                        peer,
                        mesh: decoded.mesh,
                        bounds: decoded.bounds,
                    };
                    let mut replaced = None;
                    shared.post(|inbox| replaced = inbox.meshes.insert(decoded.path, arrival));
                    // The mesh it replaced is freed here, outside the lock.
                    drop(replaced);
                }
                Err(refusal) => shared.post(|inbox| refuse(inbox, peer, refusal)),
            },
            wire::RELEASE => match wire::decode_release(&payload) {
                Ok(path) => shared.post(|inbox| {
                    if inbox.meshes.get(&path).is_some_and(|a| a.peer == peer) {
                        inbox.meshes.remove(&path);
                    }
                    if !inbox.released.iter().any(|r| *r == (peer, path.clone())) {
                        inbox.released.push((peer, path));
                    }
                }),
                Err(refusal) => shared.post(|inbox| refuse(inbox, peer, refusal)),
            },
            // A kind added since this engine, skipped by its length.
            _ => {}
        }
    }
}

fn refuse(inbox: &mut Inbox, peer: Peer, refusal: wire::Refusal) {
    inbox
        .refused
        .retain(|(from, kept)| !(*from == peer && kept.path == refusal.path));
    inbox.refused.push((peer, refusal));
}

impl AssetDatabase {
    /// Listen for the live link on `port` of `127.0.0.1`, rebinding when the
    /// port changed, or stop it for `None`, giving back everything live. Port
    /// 0 asks the OS for one, which [`link_status`](Self::link_status) reports.
    ///
    /// A port that cannot be bound is returned, naming it, and leaves a link
    /// already running as it was. A pack has no link.
    pub fn set_link(&mut self, port: Option<u16>) -> Result<(), String> {
        let Some(port) = port else {
            if let Some(link) = self.link.take() {
                self.let_go_all(link);
                self.notes
                    .push("the live link stopped listening".to_owned());
            }
            return Ok(());
        };
        if self.pack.is_some() {
            return Err("an exported game has no live link".to_owned());
        }
        if port != 0
            && self
                .link
                .as_ref()
                .is_some_and(|link| link.addr.port() == port)
        {
            return Ok(());
        }
        let link = Link::start(port, &self.dir)?;
        let addr = link.addr;
        if let Some(old) = self.link.replace(link) {
            self.let_go_all(old);
        }
        self.notes
            .push(format!("the live link is listening on {addr} for Blender"));
        Ok(())
    }

    /// What the live link is doing; `None` while it is off.
    pub fn link_status(&self) -> Option<LinkStatus> {
        let link = self.link.as_ref()?;
        let mut live: Vec<String> = link.live.values().map(|live| live.path.clone()).collect();
        live.sort();
        Some(LinkStatus {
            port: link.addr.port(),
            peers: link.shared.peers.load(Ordering::Acquire),
            live,
        })
    }

    /// Whether the slot of mesh `id` is drawing a stream rather than its file.
    pub(super) fn is_live(&self, id: AssetId) -> bool {
        self.link.as_ref().is_some_and(|link| link.is_live(id))
    }

    /// Take what the connections handed over since the last frame: swap each
    /// newest mesh into its slot, and give back what was let go.
    pub(super) fn serve_link(&mut self, world: &World, backend: &mut impl RenderBackend) {
        let Some(link) = &mut self.link else {
            return;
        };
        if !link.shared.news.load(Ordering::Acquire) {
            return;
        }
        let inbox = {
            let mut inbox = lock(&link.shared.inbox);
            link.shared.news.store(false, Ordering::Release);
            std::mem::take(&mut *inbox)
        };
        for addr in inbox.joined {
            self.notes
                .push(format!("Blender connected to the live link from {addr}"));
        }
        self.warnings.extend(inbox.failed);

        let mut loosened = Vec::new();
        for peer in inbox.gone {
            link.told.remove(&peer);
            for (&id, live) in &mut link.live {
                if live.holders.contains(&peer) {
                    live.holders.retain(|&holder| holder != peer);
                    loosened.push(id);
                }
            }
            self.notes
                .push("a Blender connection to the live link closed".to_owned());
        }
        for (peer, path) in inbox.released {
            if let Some(paths) = link.told.get_mut(&peer) {
                paths.remove(&path);
            }
            for (&id, live) in &mut link.live {
                if live.path == path && live.holders.contains(&peer) {
                    live.holders.retain(|&holder| holder != peer);
                    loosened.push(id);
                }
            }
        }
        for (peer, refusal) in inbox.refused {
            if link.tell(peer, &refusal.path, Told::Refused(refusal.reason.clone())) {
                self.warnings.push(refused(&refusal.path, &refusal.reason));
            }
        }

        for (path, arrival) in inbox.meshes {
            let Some(link) = &mut self.link else {
                return;
            };
            let found = self.entries.iter().find(|e| e.on_disk && e.path == path);
            let (id, scale) = match found {
                Some(entry) if entry.kind == AssetKind::Mesh => match entry.settings {
                    Settings::Mesh(mesh) => (entry.id, mesh.scale),
                    _ => (entry.id, 1.0),
                },
                found => {
                    let reason = match found {
                        Some(entry) => {
                            format!("that asset is a {}, not a mesh", entry.kind.label())
                        }
                        None => "the project has no mesh asset at that path; save the file under \
                                 the project's assets folder"
                            .to_owned(),
                    };
                    if link.tell(arrival.peer, &path, Told::Refused(reason.clone())) {
                        self.warnings.push(refused(&path, &reason));
                    }
                    continue;
                }
            };
            let Some(slot) = self.meshes.handle(id).filter(|h| h & PENDING == 0) else {
                link.tell(arrival.peer, &path, Told::Undrawn);
                continue;
            };
            let Arrival {
                peer,
                mut mesh,
                mut bounds,
            } = arrival;
            if scale != 1.0 {
                for vertex in &mut mesh.vertices {
                    vertex.position = vertex.position.map(|v| v * scale);
                }
                bounds = Aabb {
                    min: bounds.min * scale,
                    max: bounds.max * scale,
                };
            }
            let handle = MeshHandle(slot);
            backend.replace_mesh(handle, &mesh);
            if let Some(mut table) = world.get_resource_mut::<MeshBounds>() {
                table.insert(handle, bounds);
                table.insert_geometry(handle, &mesh);
            }
            // No key: the artifact on disk is no longer what the slot holds,
            // so letting go installs it rather than taking it as drawn.
            let ready = handles::Resident {
                state: LoadState::Ready,
                key: None,
            };
            self.meshes.set_resident(id, ready);
            let live = link.live.entry(id).or_insert_with(|| Live {
                path: path.clone(),
                holders: Vec::new(),
            });
            let went_live = live.holders.is_empty();
            if !live.holders.contains(&peer) {
                live.holders.push(peer);
            }
            link.tell(peer, &path, Told::Live);
            if went_live {
                self.notes.push(format!(
                    "{path} is live from Blender: edits there show here before they are saved"
                ));
            }
        }

        for id in loosened {
            let Some(link) = &mut self.link else {
                return;
            };
            if let Some(live) = link.live.get(&id)
                && live.holders.is_empty()
            {
                let path = live.path.clone();
                link.live.remove(&id);
                self.restore(id);
                self.notes
                    .push(format!("{path} is no longer live; drawing the file again"));
            }
        }
    }

    /// Give back every live asset whose entry a scan deleted or renamed.
    pub(super) fn let_go_moved(&mut self) {
        let Some(link) = &mut self.link else {
            return;
        };
        if link.live.is_empty() {
            return;
        }
        let entries = &self.entries;
        let moved: Vec<(AssetId, String)> = link
            .live
            .iter()
            .filter(|&(&id, live)| {
                !entries
                    .iter()
                    .any(|e| e.id == id && e.on_disk && e.path == live.path)
            })
            .map(|(&id, live)| (id, live.path.clone()))
            .collect();
        for (id, path) in &moved {
            link.live.remove(id);
            self.notes.push(format!(
                "{path} was moved or deleted, so it is no longer live"
            ));
        }
        for (id, _) in moved {
            self.restore(id);
        }
    }

    fn let_go_all(&mut self, mut link: Link) {
        link.stop();
        let mut live: Vec<(AssetId, String)> = link
            .live
            .drain()
            .map(|(id, live)| (id, live.path))
            .collect();
        drop(link);
        live.sort_by(|a, b| a.1.cmp(&b.1));
        for (id, path) in live {
            self.restore(id);
            self.notes
                .push(format!("{path} is no longer live; drawing the file again"));
        }
    }

    /// Bring the artifact for what is on disk back into `id`'s slot.
    fn restore(&mut self, id: AssetId) {
        let drawn = self.meshes.handle(id).is_some_and(|h| h & PENDING == 0);
        if drawn && !self.awaiting.contains(&id) {
            self.queue(id);
        }
    }
}

fn refused(path: &str, reason: &str) -> String {
    match path {
        "" => format!("the live link refused a frame: {reason}"),
        path => format!("the live link refused a mesh for {path}: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefMut;
    use std::io::{ErrorKind, Read, Write};
    use std::time::Instant;

    use super::wire::tests::quad;
    use super::*;
    use crate::asset_db::import::tests::{png, triangle_bin, triangle_gltf};
    use crate::asset_db::tests::{Dir, WAIT, drawing, handle, open, update};
    use crate::asset_db::{CacheKey, Status};
    use crate::gfx::HeadlessBackend;

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let started = Instant::now();
        while !done() {
            assert!(started.elapsed() < WAIT, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// The disk's triangle, as `triangle_gltf` draws it.
    fn on_disk() -> Aabb {
        Aabb::from_points([glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y])
    }

    /// The quad `quad(_, offset)` sends, at `scale`.
    fn streamed(offset: f32, scale: f32) -> Aabb {
        Aabb::from_points([
            glam::Vec3::new(offset, offset, offset) * scale,
            glam::Vec3::new(offset + 1.0, offset + 1.0, offset) * scale,
        ])
    }

    fn strings(mut payload: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        while let Some((len, rest)) = payload.split_first_chunk::<4>() {
            let (text, rest) = rest.split_at(u32::from_le_bytes(*len) as usize);
            out.push(String::from_utf8(text.to_vec()).unwrap());
            payload = rest;
        }
        out
    }

    struct Client(TcpStream);

    impl Client {
        fn raw(port: u16) -> TcpStream {
            let stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
            stream.set_read_timeout(Some(WAIT)).unwrap();
            stream
        }

        fn send(&mut self, frame: &[u8]) {
            self.0.write_all(frame).unwrap();
        }

        fn next(&mut self) -> (u8, Vec<u8>) {
            let mut payload = Vec::new();
            let kind = wire::read_frame(&mut self.0, &mut payload).expect("a frame");
            (kind, payload)
        }

        /// The next frame, which must be an error: its path and message.
        fn error(&mut self) -> (String, String) {
            let (kind, payload) = self.next();
            assert_eq!(kind, wire::ERROR, "{:?}", strings(&payload));
            let [path, message] = <[String; 2]>::try_from(strings(&payload)).unwrap();
            (path, message)
        }

        fn live(&mut self) -> String {
            let (kind, payload) = self.next();
            assert_eq!(kind, wire::LIVE, "{:?}", strings(&payload));
            strings(&payload).remove(0)
        }

        /// Whether the engine has closed its end, having sent nothing more.
        fn closed(&mut self) -> bool {
            let mut rest = Vec::new();
            let read = self.0.read_to_end(&mut rest);
            rest.is_empty()
                && match read {
                    Ok(_) => true,
                    // Closed with what it had not read still in its buffer.
                    Err(error) => error.kind() == ErrorKind::ConnectionReset,
                }
        }
    }

    /// A project drawing the glTF tree, its sidecar at `scale`, loaded, with
    /// the link listening on a port the OS chose.
    struct Rig {
        dir: Dir,
        world: World,
        slot: MeshHandle,
        backend: HeadlessBackend,
        port: u16,
    }

    impl Rig {
        fn new(name: &str, scale: f32) -> Self {
            let dir = Dir::new(name);
            dir.write("models/tree.gltf", triangle_gltf("").as_bytes());
            dir.write("models/tri angle.bin", &triangle_bin());
            dir.write(
                "models/tree.gltf.meta",
                format!(
                    "id = \"5f0c3a4e-0000-4000-8000-0000000000b1\"\n\n[import]\nscale = {scale:?}\n"
                )
                .as_bytes(),
            );
            dir.write("rocks/rock.png", &png(1, 1, |_, _| [9; 4]));
            let mut db = open(&dir);
            db.set_link(Some(0)).unwrap();
            let port = db.link_status().unwrap().port;
            assert_ne!(port, 0);
            let id = db.find("models/tree.gltf").unwrap().id;
            let (world, entity) = drawing(db, id);
            let mut rig = Self {
                dir,
                world,
                slot: MeshHandle(0),
                backend: HeadlessBackend::new(),
                port,
            };
            rig.frame();
            rig.settle();
            rig.slot = handle(&rig.world, entity);
            assert_eq!(rig.bounds(), on_disk().scaled(scale));
            rig
        }

        fn db(&self) -> RefMut<'_, AssetDatabase> {
            self.world.resource_mut::<AssetDatabase>()
        }

        fn frame(&mut self) -> Vec<String> {
            update(&self.world, &mut self.backend)
        }

        /// Wait for the import thread, then take what it finished.
        fn settle(&mut self) -> Vec<String> {
            assert!(self.db().wait_idle(WAIT));
            self.frame()
        }

        fn received(&self) -> u64 {
            let db = self.db();
            let link = db.link.as_ref().expect("the link is on");
            link.shared.received.load(Ordering::Relaxed)
        }

        fn peers(&self) -> usize {
            self.db().link_status().map_or(0, |status| status.peers)
        }

        /// Connect, handshake, and wait until the database has heard of it.
        fn connect(&self) -> Client {
            let before = self.peers();
            let mut client = Client(Client::raw(self.port));
            client.send(&wire::preamble(VERSION));
            let (kind, payload) = client.next();
            assert_eq!(kind, wire::HELLO);
            let version = u32::from_le_bytes(payload[..4].try_into().unwrap());
            assert_eq!(version, VERSION);
            let assets = std::path::absolute(self.dir.path()).unwrap();
            assert_eq!(strings(&payload[4..]), [assets.to_string_lossy()]);
            wait_until("the connection to be counted", || {
                self.peers() == before + 1
            });
            client
        }

        /// Send `frames` and wait until the connection has handed each over.
        fn send(&self, client: &mut Client, frames: &[Vec<u8>]) {
            let before = self.received();
            for frame in frames {
                client.send(frame);
            }
            let after = before + frames.len() as u64;
            wait_until("the frames to arrive", || self.received() >= after);
        }

        fn bounds(&self) -> Aabb {
            self.backend.mesh_bounds(self.slot).unwrap()
        }

        fn live(&self) -> Vec<String> {
            self.db().link_status().map(|s| s.live).unwrap_or_default()
        }

        fn replacements(&self) -> u32 {
            self.backend.mesh_replacements()
        }
    }

    trait Scaled {
        fn scaled(self, scale: f32) -> Self;
    }

    impl Scaled for Aabb {
        fn scaled(self, scale: f32) -> Self {
            Aabb {
                min: self.min * scale,
                max: self.max * scale,
            }
        }
    }

    #[test]
    fn a_peer_that_opens_with_http_is_dropped_having_been_told_nothing() {
        let dir = Dir::new("link-http");
        let mut db = open(&dir);
        db.set_link(Some(0)).unwrap();
        let port = db.link_status().unwrap().port;
        let mut browser = Client(Client::raw(port));
        browser.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(browser.closed());
        assert_eq!(db.link_status().unwrap().peers, 0);
    }

    #[test]
    fn a_peer_of_another_version_is_told_to_update_the_add_on_then_dropped() {
        let dir = Dir::new("link-version");
        let mut db = open(&dir);
        db.set_link(Some(0)).unwrap();
        let mut old = Client(Client::raw(db.link_status().unwrap().port));
        old.send(&wire::preamble(VERSION + 1));
        let (path, message) = old.error();
        assert_eq!(path, "");
        assert_eq!(
            message,
            format!(
                "this add-on speaks live link protocol {} and the editor speaks {VERSION}; \
                 update the add-on from the editor's Settings ▸ Blender",
                VERSION + 1
            )
        );
        assert!(old.closed());
    }

    #[test]
    fn a_streamed_mesh_replaces_the_drawn_slot_at_the_sidecar_scale() {
        let mut rig = Rig::new("link-stream", 2.0);
        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 1.0)]);
        let errors = rig.frame();
        assert!(errors.is_empty(), "{errors:?}");

        let expected = streamed(1.0, 2.0);
        assert_eq!(rig.bounds(), expected);
        let table = rig.world.resource::<MeshBounds>();
        assert_eq!(table.get(rig.slot), Some(expected));
        let geometry = table.geometry(rig.slot).unwrap();
        assert_eq!(geometry.positions[2], glam::Vec3::new(4.0, 4.0, 2.0));
        assert_eq!(geometry.triangles, [[0, 1, 2], [0, 2, 3]]);
        drop(table);
        let id = rig.db().find("models/tree.gltf").unwrap().id;
        assert_eq!(rig.db().state(id), LoadState::Ready);
        assert_eq!(rig.live(), ["models/tree.gltf"]);
        assert_eq!(blender.live(), "models/tree.gltf");
        let notes = rig.db().take_notes();
        assert!(
            notes
                .iter()
                .any(|n| n.starts_with("models/tree.gltf is live")),
            "{notes:?}"
        );
    }

    #[test]
    fn three_meshes_before_one_update_cost_one_replace() {
        let mut rig = Rig::new("link-latest", 1.0);
        let mut blender = rig.connect();
        let before = rig.replacements();
        rig.send(
            &mut blender,
            &[1.0, 2.0, 3.0].map(|offset| quad("models/tree.gltf", offset)),
        );
        rig.frame();
        assert_eq!(rig.replacements(), before + 1);
        assert_eq!(rig.bounds(), streamed(3.0, 1.0), "the newest");
        rig.frame();
        assert_eq!(rig.replacements(), before + 1, "and nothing is left over");
    }

    /// An unknown path, a path that is not a mesh and a mesh that does not
    /// decode are each answered and replace nothing; the connection survives
    /// all three, and a reason repeated is not sent again.
    #[test]
    fn a_refused_mesh_is_answered_with_its_reason_and_replaces_nothing() {
        let mut rig = Rig::new("link-refused", 1.0);
        let mut blender = rig.connect();
        let before = rig.replacements();
        let mut cut = quad("models/tree.gltf", 1.0);
        cut.pop();
        let len = (cut.len() - wire::HEADER) as u32;
        cut[..4].copy_from_slice(&len.to_le_bytes());
        rig.send(
            &mut blender,
            &[
                quad("models/nothing.blend", 1.0),
                quad("rocks/rock.png", 1.0),
                cut,
            ],
        );
        rig.frame();
        let mut answers = [blender.error(), blender.error(), blender.error()];
        answers.sort();
        assert_eq!(
            answers,
            [
                (
                    "models/nothing.blend".to_owned(),
                    "the project has no mesh asset at that path; save the file under the \
                     project's assets folder"
                        .to_owned()
                ),
                (
                    "models/tree.gltf".to_owned(),
                    "its counts (4 vertices, 6 indices) describe 152 bytes of geometry and the \
                     frame carries 151"
                        .to_owned()
                ),
                (
                    "rocks/rock.png".to_owned(),
                    "that asset is a texture, not a mesh".to_owned()
                ),
            ]
        );
        assert_eq!(rig.replacements(), before);
        assert_eq!(rig.bounds(), on_disk());
        assert!(rig.live().is_empty());
        assert_eq!(rig.db().take_warnings().len(), 3);

        rig.send(
            &mut blender,
            &[
                quad("models/nothing.blend", 2.0),
                quad("models/tree.gltf", 1.0),
            ],
        );
        rig.frame();
        assert_eq!(blender.live(), "models/tree.gltf", "no second error first");
        assert!(rig.db().take_warnings().is_empty(), "once per change");
    }

    #[test]
    fn an_oversized_frame_ends_that_connection_alone() {
        let mut rig = Rig::new("link-oversized", 1.0);
        let mut greedy = rig.connect();
        let mut header = (MAX_FRAME + 1).to_le_bytes().to_vec();
        header.push(wire::MESH);
        greedy.send(&header);
        let (path, message) = greedy.error();
        assert_eq!(path, "");
        assert!(
            message.starts_with("a frame of 268435457 bytes is over"),
            "{message}"
        );
        assert!(greedy.closed());
        wait_until("the connection to go", || rig.peers() == 0);

        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 1.0)]);
        rig.frame();
        assert_eq!(rig.bounds(), streamed(1.0, 1.0));
        let warnings = rig.db().take_warnings();
        assert!(
            warnings
                .iter()
                .any(|w| w.starts_with("a frame of 268435457 bytes")),
            "{warnings:?}"
        );
    }

    /// The import of a save lands after edits streamed since it; it updates
    /// the entry and leaves the slot to the stream, and so does one that fails.
    #[test]
    fn a_disk_import_finishing_while_live_leaves_the_slot_the_streams() {
        let mut rig = Rig::new("link-save", 1.0);
        let key = |rig: &Rig| match rig.db().find("models/tree.gltf").unwrap().status {
            Status::Imported { key, .. } => key,
            ref other => panic!("{other:?}"),
        };
        let saved: CacheKey = key(&rig);
        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 5.0)]);
        rig.frame();
        let replaced = rig.replacements();

        let moved = triangle_gltf(r#", "translation": [0, 0, 9]"#);
        let path = rig.dir.write("models/tree.gltf", moved.as_bytes());
        rig.db().notice(path);
        rig.db().apply_changes();
        let errors = rig.settle();
        assert!(errors.is_empty(), "{errors:?}");
        assert_ne!(key(&rig), saved, "the entry has the new import");
        assert_eq!(rig.bounds(), streamed(5.0, 1.0));
        assert_eq!(rig.replacements(), replaced);

        let path = rig.dir.write("models/tree.gltf", b"{ not a gltf");
        rig.db().notice(path);
        rig.db().apply_changes();
        let errors = rig.settle();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].ends_with("Still drawing what Blender is streaming"),
            "{}",
            errors[0]
        );
        assert_eq!(rig.bounds(), streamed(5.0, 1.0), "never the placeholder");
        let id = rig.db().find("models/tree.gltf").unwrap().id;
        assert_eq!(rig.db().state(id), LoadState::Ready);
    }

    #[test]
    fn a_release_or_a_disconnect_brings_the_disk_artifact_back() {
        let mut rig = Rig::new("link-let-go", 2.0);
        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 1.0)]);
        rig.frame();
        assert_eq!(rig.bounds(), streamed(1.0, 2.0));

        rig.send(&mut blender, &[wire::release("models/tree.gltf")]);
        rig.frame();
        assert!(rig.live().is_empty());
        rig.settle();
        assert_eq!(rig.bounds(), on_disk().scaled(2.0));

        rig.send(&mut blender, &[quad("models/tree.gltf", 3.0)]);
        rig.frame();
        assert_eq!(rig.bounds(), streamed(3.0, 2.0));
        drop(blender);
        wait_until("the connection to go", || rig.peers() == 0);
        rig.frame();
        assert!(rig.live().is_empty());
        rig.settle();
        assert_eq!(rig.bounds(), on_disk().scaled(2.0));
        let notes = rig.db().take_notes();
        assert!(
            notes
                .iter()
                .any(|n| n == "models/tree.gltf is no longer live; drawing the file again"),
            "{notes:?}"
        );
    }

    #[test]
    fn a_live_asset_renamed_goes_back_to_its_file() {
        let mut rig = Rig::new("link-rename", 1.0);
        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 1.0)]);
        rig.frame();
        assert_eq!(rig.bounds(), streamed(1.0, 1.0));
        rig.db().rename("models/tree.gltf", "oak").unwrap();
        assert!(rig.live().is_empty());
        rig.settle();
        assert_eq!(rig.bounds(), on_disk());
    }

    #[test]
    fn turning_the_link_off_gives_everything_back_and_frees_the_port() {
        let mut rig = Rig::new("link-off", 1.0);
        let mut blender = rig.connect();
        rig.send(&mut blender, &[quad("models/tree.gltf", 1.0)]);
        rig.frame();
        assert_eq!(blender.live(), "models/tree.gltf");

        let started = Instant::now();
        rig.db().set_link(None).unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
        assert!(rig.db().link_status().is_none());
        assert!(blender.closed());
        assert!(TcpListener::bind((Ipv4Addr::LOCALHOST, rig.port)).is_ok());
        rig.settle();
        assert_eq!(rig.bounds(), on_disk());
    }

    #[test]
    fn a_pack_has_no_link() {
        let dir = Dir::new("link-pack");
        let (mut db, _) = AssetDatabase::open_packed(dir.path(), crate::asset_db::Target::Rgba8);
        assert_eq!(
            db.set_link(Some(0)),
            Err("an exported game has no live link".to_owned())
        );
    }

    /// A grid of `side` x `side` quads in the XZ plane, `2 x side^2`
    /// triangles, with shared vertices as a smooth-shaded grid has them.
    fn grid(path: &str, side: u32, lift: f32) -> Vec<u8> {
        let row = side + 1;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        for z in 0..row {
            for x in 0..row {
                positions.extend([x as f32, lift, z as f32]);
                uvs.extend([x as f32 / side as f32, z as f32 / side as f32]);
            }
        }
        let normals = [0.0, 1.0, 0.0].repeat((row * row) as usize);
        let mut indices = Vec::new();
        for z in 0..side {
            for x in 0..side {
                let at = z * row + x;
                indices.extend([at, at + row, at + 1, at + 1, at + row, at + row + 1]);
            }
        }
        wire::mesh(path, &positions, &normals, &uvs, &indices)
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    /// What a 100k-triangle mesh costs: decoding it on its connection's
    /// thread, and installing it on the frame's. Run with `--release`.
    #[test]
    #[ignore = "a measurement; run with --release --nocapture"]
    fn report_what_a_streamed_mesh_costs() {
        const RUNS: usize = 31;
        let frame = grid("models/tree.gltf", 224, 0.0);
        let decoded = wire::decode_mesh(&frame[wire::HEADER..]).unwrap();
        let (vertices, triangles) = (decoded.mesh.vertices.len(), decoded.mesh.indices.len() / 3);
        let decode = median(
            (0..RUNS)
                .map(|_| {
                    let started = Instant::now();
                    std::hint::black_box(wire::decode_mesh(&frame[wire::HEADER..]).unwrap());
                    started.elapsed()
                })
                .collect(),
        );
        let positions: Vec<glam::Vec3> = decoded
            .mesh
            .vertices
            .iter()
            .map(|v| glam::Vec3::from(v.position))
            .collect();
        let normals: Vec<glam::Vec3> = decoded
            .mesh
            .vertices
            .iter()
            .map(|v| glam::Vec3::from(v.normal))
            .collect();
        let uvs: Vec<glam::Vec2> = decoded
            .mesh
            .vertices
            .iter()
            .map(|v| glam::Vec2::from(v.uv))
            .collect();
        let tangents = median(
            (0..RUNS)
                .map(|_| {
                    let started = Instant::now();
                    std::hint::black_box(crate::scene::model::derive_tangents(
                        &positions,
                        &normals,
                        &uvs,
                        &decoded.mesh.indices,
                    ));
                    started.elapsed()
                })
                .collect(),
        );

        let mut rig = Rig::new("link-cost", 1.0);
        let mut blender = rig.connect();
        let installs = median(
            (0..RUNS)
                .map(|run| {
                    rig.send(&mut blender, &[grid("models/tree.gltf", 224, run as f32)]);
                    let started = Instant::now();
                    rig.frame();
                    started.elapsed()
                })
                .collect(),
        );
        assert_eq!(rig.bounds().min.y, (RUNS - 1) as f32);
        let mut scaled = Rig::new("link-cost-scaled", 2.0);
        let mut blender = scaled.connect();
        let scaled_installs = median(
            (0..RUNS)
                .map(|run| {
                    scaled.send(&mut blender, &[grid("models/tree.gltf", 224, run as f32)]);
                    let started = Instant::now();
                    scaled.frame();
                    started.elapsed()
                })
                .collect(),
        );
        let mut table = MeshBounds::default();
        let geometry = median(
            (0..RUNS)
                .map(|_| {
                    let started = Instant::now();
                    table.insert_geometry(MeshHandle(0), &decoded.mesh);
                    started.elapsed()
                })
                .collect(),
        );
        let idle = median(
            (0..RUNS)
                .map(|_| {
                    let started = Instant::now();
                    rig.frame();
                    started.elapsed()
                })
                .collect(),
        );
        eprintln!(
            "a {triangles}-triangle mesh, {vertices} vertices, {} bytes on the wire; medians of \
             {RUNS}:\n  decode on the connection thread  {decode:?} (of which tangents \
             {tangents:?})\n  update installing it, scale 1    {installs:?}\n  update installing \
             it, scale 2    {scaled_installs:?}\n  of which MeshBounds' collider copy \
             {geometry:?}\n  update with the link on and idle {idle:?}",
            frame.len()
        );
    }
}
