"""The live link's wire and the threads that speak it.

The protocol is defined once, in the `//!` header of the engine's
`crates/core/src/asset_db/link.rs`; this is its client half, standard library
only, so it runs in any Blender's Python and in none of Blender's threads.

One thread connects, retrying every `RETRY` seconds while the client runs, so
it does not matter whether Blender or the editor started first, and then
writes; a second reads, because the engine drops a client that leaves a frame
unread for five seconds. Blender's main thread only ever swaps a frame into the
outgoing slot and copies the status out, each under one short lock.

**Latest wins.** The outgoing mesh is one slot, not a queue: a network slower
than the edits drops the meshes in between rather than falling behind them.
A release is not dropped, and drops a mesh of its path still waiting.
"""

import socket
import struct
import threading

MAGIC = b"ORRNLINK"
VERSION = 1
DEFAULT_PORT = 47103
MAX_FRAME = 256 << 20

HELLO, ERROR, LIVE, MESH, RELEASE = 1, 2, 3, 4, 5

HEADER = struct.Struct("<IB")
U32 = struct.Struct("<I")

# Seconds between attempts to reach the editor.
RETRY = 2.0
# Seconds the editor has to answer the preamble, as it gives a client.
HANDSHAKE = 5.0


def string(text):
    data = text.encode("utf-8")
    return U32.pack(len(data)) + data


def frame(kind, *parts):
    """A frame as the buffers to write, its payload left in `parts`."""
    size = sum(memoryview(part).nbytes for part in parts)
    return [HEADER.pack(size, kind), *parts]


def mesh_frame(path, positions, normals, uvs, indices):
    """A `mesh` frame over the arrays themselves, which are sent without a
    copy: float32 and uint32, little-endian, contiguous."""
    head = string(path) + struct.pack("<II", len(positions), len(indices))
    return frame(MESH, head, positions.data, normals.data, uvs.data, indices.data)


def release_frame(path):
    return frame(RELEASE, string(path))


def payload_size(parts):
    return sum(memoryview(part).nbytes for part in parts[1:])


class Status:
    """What the client knows, copied out for the main thread."""

    __slots__ = ("connected", "connection", "assets", "live", "errors", "problem")

    def __init__(self):
        self.connected = False
        # Counts connections, so a new one is noticed even between two looks.
        self.connection = 0
        # The editor's assets directory, as its `hello` carries it.
        self.assets = None
        self.live = set()
        # The editor's last word on each path it refused.
        self.errors = {}
        # Why there is no connection, or what the editor said ended one.
        self.problem = None

    def copy(self):
        out = Status()
        out.connected = self.connected
        out.connection = self.connection
        out.assets = self.assets
        out.live = set(self.live)
        out.errors = dict(self.errors)
        out.problem = self.problem
        return out


class Client:
    def __init__(self, port, host="127.0.0.1"):
        self.address = (host, port)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._stopping = False
        self._socket = None
        self._dead = False
        self._mesh = None
        self._control = []
        self._status = Status()
        self._thread = None
        # Meshes written whole to the socket, for a test to wait on.
        self.sent = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="orrin-link", daemon=True)
        self._thread.start()

    def stop(self, timeout=2.0):
        """Close the connection and wait for both threads. A release queued
        before it is sent first, if the editor takes it within `timeout`."""
        with self._lock:
            self._stopping = True
            self._wake.notify_all()
        if self._thread is None:
            return
        self._thread.join(timeout)
        if self._thread.is_alive():
            # A write the editor stopped reading: shutting the socket down is
            # what returns it.
            with self._lock:
                if self._socket is not None:
                    try:
                        self._socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            self._thread.join(timeout)
        self._thread = None

    def status(self):
        with self._lock:
            return self._status.copy()

    def send_mesh(self, path, parts):
        """Replace whatever mesh is waiting; dropped while not connected."""
        with self._lock:
            if self._socket is None:
                return
            self._mesh = (path, parts)
            self._wake.notify_all()

    def release(self, path):
        with self._lock:
            if self._socket is None:
                return
            if self._mesh is not None and self._mesh[0] == path:
                self._mesh = None
            self._control.append(release_frame(path))
            self._status.live.discard(path)
            self._status.errors.pop(path, None)
            self._wake.notify_all()

    def _wait(self, seconds):
        with self._lock:
            if not self._stopping:
                self._wake.wait(seconds)
            return self._stopping

    def _run(self):
        while not self._stopping:
            try:
                sock = socket.create_connection(self.address, timeout=1.0)
            except OSError:
                # The editor is not running, which is no fault to report.
                with self._lock:
                    self._status.problem = None
                if self._wait(RETRY):
                    return
                continue
            try:
                self._serve(sock)
            except (OSError, ValueError) as error:
                with self._lock:
                    if not self._stopping:
                        self._status.problem = f"the connection to the editor failed: {error}"
            finally:
                with self._lock:
                    self._socket = None
                    self._mesh = None
                    self._control.clear()
                    self._status.connected = False
                    self._status.live.clear()
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
            if self._wait(RETRY):
                return

    def _serve(self, sock):
        sock.settimeout(HANDSHAKE)
        sock.sendall(MAGIC + U32.pack(VERSION))
        kind, payload = read_frame(sock)
        if kind == ERROR:
            _, message = strings(payload, 2)
            with self._lock:
                self._status.problem = message
            return
        if kind != HELLO or len(payload) < 4:
            raise OSError("the editor did not answer as an Orrin editor")
        (assets,) = strings(payload[4:], 1)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._lock:
            if self._stopping:
                return
            self._socket = sock
            self._dead = False
            status = self._status
            status.connected = True
            status.connection += 1
            status.assets = assets
            status.live.clear()
            status.errors.clear()
            status.problem = None
        reader = threading.Thread(target=self._read, args=(sock,), name="orrin-link-in", daemon=True)
        reader.start()
        try:
            self._write(sock)
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            reader.join()

    def _write(self, sock):
        while True:
            with self._lock:
                while not (self._stopping or self._dead or self._control or self._mesh):
                    self._wake.wait()
                if self._dead:
                    return
                control, self._control = self._control, []
                mesh, self._mesh = (None, None) if self._stopping else (self._mesh, None)
                stopping = self._stopping
            for parts in control:
                _write_all(sock, parts)
            if mesh is not None:
                _write_all(sock, mesh[1])
                self.sent += 1
            if stopping:
                return

    def _read(self, sock):
        try:
            while True:
                kind, payload = read_frame(sock)
                if kind == LIVE:
                    (path,) = strings(payload, 1)
                    with self._lock:
                        self._status.live.add(path)
                        self._status.errors.pop(path, None)
                elif kind == ERROR:
                    path, message = strings(payload, 2)
                    with self._lock:
                        if path:
                            self._status.errors[path] = message
                            self._status.live.discard(path)
                        else:
                            self._status.problem = message
                # Any other kind is one added since this client, skipped whole.
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._dead = True
                self._wake.notify_all()


def _write_all(sock, parts):
    for part in parts:
        sock.sendall(part)


def read_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    got = 0
    while got < size:
        read = sock.recv_into(view[got:])
        if not read:
            raise OSError("the editor closed the connection")
        got += read
    return bytes(data)


def read_frame(sock):
    size, kind = HEADER.unpack(read_exact(sock, HEADER.size))
    if size > MAX_FRAME:
        raise ValueError(f"the editor announced a frame of {size} bytes")
    return kind, read_exact(sock, size)


def strings(payload, count):
    """The first `count` strings of `payload`."""
    out = []
    at = 0
    for _ in range(count):
        if at + 4 > len(payload):
            raise ValueError("the editor sent a frame cut short")
        (size,) = U32.unpack_from(payload, at)
        at += 4
        out.append(payload[at:at + size].decode("utf-8", "replace"))
        at += size
    return out
