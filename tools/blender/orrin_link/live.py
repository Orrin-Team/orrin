"""The main thread's half of the link: which file streams, when, and what the
panel says about it.

Everything here runs on Blender's main thread, from the handlers a session
installs while it runs and from `tick`, and nothing here touches a socket: the
client's threads do. The add-on's timer calls `tick`; a background script,
where timers never run, calls it itself.

**The file is the unit.** The `.blend` open in Blender streams to its own
asset path, the one it has relative to the assets directory the editor's
`hello` names, both resolved through links first. A file outside that
directory, or never saved, streams nothing, and `status_line` says which.

**Pacing.** A change marks the session dirty; a tick sends when it is dirty, no
more often than `INTERVAL`, and no sooner after the last extraction than `PACE`
times what it took, so a heavy scene costs Blender a third of its main thread
at most rather than all of it. A mesh identical to the last one sent, which a
selection change produces, is not sent again.
"""

import os
import time

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import client, extract

# Seconds between meshes at the most: 30 a second.
INTERVAL = 1 / 30
# The next extraction waits at least this many times the last one's length.
PACE = 3.0
# Seconds between ticks with nothing to send.
IDLE = 0.25


class Session:
    def __init__(self, port):
        self.client = client.Client(port)
        self.dirty = True
        self.next_send = 0.0
        self.extraction = 0.0
        # The path the editor was last sent a mesh for, and that mesh.
        self.streaming = None
        self.last = None
        self.connection = 0
        # Why nothing was sent, when the reason is this side's.
        self.withheld = None
        self.handlers = []

    def start(self):
        """Connect, and follow the scene and the file from now on."""
        handlers = bpy.app.handlers
        # Persistent, or opening a file would drop them.
        self.handlers = [
            (handlers.depsgraph_update_post, persistent(lambda scene, graph: self.mark_dirty())),
            (handlers.load_pre, persistent(lambda *_: self.let_go())),
            (handlers.load_post, persistent(lambda *_: self.file_changed())),
            (handlers.save_post, persistent(lambda *_: self.file_changed())),
        ]
        for handler_list, handler in self.handlers:
            handler_list.append(handler)
        self.client.start()

    def stop(self):
        """Hand the asset back to its file and close the connection."""
        for handler_list, handler in self.handlers:
            if handler in handler_list:
                handler_list.remove(handler)
        self.handlers = []
        self.let_go()
        self.client.stop()

    def let_go(self):
        if self.streaming is not None:
            self.client.release(self.streaming)
        self.streaming = None
        self.last = None
        self.dirty = True

    def mark_dirty(self):
        self.dirty = True

    def file_changed(self):
        """After a file is opened or saved under another name: a path no
        longer this file's is handed back."""
        status = self.client.status()
        if self.streaming is not None and self.streaming != asset_path(status.assets):
            self.let_go()
        self.dirty = True

    def tick(self, now=None):
        """Send what changed, if it is time. Seconds until the next tick."""
        now = time.monotonic() if now is None else now
        status = self.client.status()
        if status.connection != self.connection:
            # A new connection holds nothing; send the scene again.
            self.connection = status.connection
            self.streaming = None
            self.last = None
            self.dirty = True
        if not status.connected:
            return IDLE
        path = asset_path(status.assets)
        if path != self.streaming and self.streaming is not None:
            self.let_go()
        if path is None or not self.dirty:
            return IDLE
        if now < self.next_send:
            return self.next_send - now

        started = time.perf_counter()
        self.dirty = False
        mesh = extract.extract(depsgraph())
        self.extraction = time.perf_counter() - started
        self.next_send = now + max(INTERVAL, PACE * self.extraction)
        if mesh is None:
            # The editor refuses an empty mesh, and drawing the last one would
            # show what is no longer there; the file is the nearer answer.
            if self.streaming is not None:
                self.client.release(self.streaming)
            self.streaming = None
            self.last = None
            self.withheld = "Nothing visible has a face to stream"
            return IDLE
        if self.last is not None and path == self.streaming and same(mesh, self.last):
            return IDLE
        parts = client.mesh_frame(path, mesh.positions, mesh.normals, mesh.uvs, mesh.indices)
        size = client.payload_size(parts)
        if size > client.MAX_FRAME:
            self.withheld = f"The scene is {size >> 20} MB, over the link's {client.MAX_FRAME >> 20} MB"
            return IDLE
        self.withheld = None
        self.client.send_mesh(path, parts)
        self.streaming = path
        self.last = mesh
        return max(INTERVAL, self.next_send - now)

    def status_line(self):
        status = self.client.status()
        if not status.connected:
            return status.problem or "Not connected: is the Orrin editor running?"
        if not bpy.data.filepath:
            return "Connected. Save this file in the project's assets folder to stream it"
        path = asset_path(status.assets)
        if path is None:
            return "Connected. This file is not in the project's assets folder"
        if path in status.errors:
            return status.errors[path]
        if self.withheld:
            return self.withheld
        if path in status.live:
            return f"Streaming {path}"
        return f"Sending {path}"


def asset_path(assets):
    """This file's path relative to `assets`, with `/` separators as the
    editor's index has it; `None` for an unsaved file or one outside it."""
    filepath = bpy.data.filepath
    if not filepath or not assets:
        return None
    root = os.path.realpath(assets)
    real = os.path.realpath(filepath)
    try:
        rel = os.path.relpath(real, root)
    except ValueError:
        # Another drive, on Windows.
        return None
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None
    return rel.replace(os.sep, "/")


def depsgraph():
    """The evaluated depsgraph of the window's view layer. A timer runs with
    no window in its context, so it borrows the first one's."""
    context = bpy.context
    windows = context.window_manager.windows if context.window_manager else ()
    if context.window is None and windows:
        with context.temp_override(window=windows[0]):
            return bpy.context.evaluated_depsgraph_get()
    return context.evaluated_depsgraph_get()


def same(a, b):
    return (
        len(a.indices) == len(b.indices)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.positions, b.positions)
        and np.array_equal(a.normals, b.normals)
        and np.array_equal(a.uvs, b.uvs)
    )
