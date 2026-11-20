"""Orrin Live Link: the `.blend` open in Blender, streamed into the Orrin
editor's viewport as it is edited, Edit Mode included, before it is saved.

`client` speaks the editor's wire protocol on threads of its own, `extract`
reads the scene as the editor's `.blend` import exports it, and `live` decides
when to send what. This module is only Blender's side of it: the preferences,
the sidebar panel, and the timer that drives a `live.Session`.

A Blender started with `--background` registers the add-on but never connects
by itself: a render farm or a batch script is not somebody editing. A script
that wants the link drives a `live.Session` itself.
"""

import traceback

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty

from . import client, live

_session = None
_shown = None


def _prefs():
    return bpy.context.preferences.addons[__package__].preferences


def start():
    global _session
    if _session is not None:
        return
    _session = live.Session(_prefs().port)
    _session.start()
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.0, persistent=True)
    _redraw()


def stop():
    global _session
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _session is not None:
        session, _session = _session, None
        session.stop()
    _redraw()


def status_line():
    if _session is None:
        return "Off"
    return _session.status_line()


def _tick():
    if _session is None:
        return None
    try:
        interval = _session.tick()
    except Exception:
        traceback.print_exc()
        interval = 1.0
    global _shown
    line = _session.status_line()
    if line != _shown:
        _shown = line
        _redraw()
    return interval


def _redraw():
    manager = bpy.context.window_manager
    for window in manager.windows if manager else ():
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _autostart():
    if _session is None and _prefs().connect_on_open:
        start()
    return None


@persistent
def _on_load_post(*_):
    if _session is None and not bpy.app.background and _prefs().connect_on_open:
        start()


def _restart(prefs, context):
    if _session is not None:
        stop()
        start()


def _set_enabled(manager, on):
    if on:
        start()
    else:
        stop()


class OrrinLinkPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    port: IntProperty(
        name="Port",
        description="The port the Orrin editor listens on, as its Settings ▸ Blender page says",
        default=client.DEFAULT_PORT,
        min=1,
        max=65535,
        update=_restart,
    )
    connect_on_open: BoolProperty(
        name="Connect when a file opens",
        description="Start streaming whenever Blender starts or opens a file",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "port")
        layout.prop(self, "connect_on_open")


class VIEW3D_PT_orrin_link(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Orrin"
    bl_label = "Live Link"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.window_manager, "orrin_link", text="Stream to Orrin")
        column = layout.column(align=True)
        # A label does not wrap, and the sidebar is narrow.
        width = max(16, int(context.region.width / (7 * context.preferences.view.ui_scale)))
        for line in _wrap(status_line(), width):
            column.label(text=line)


def _wrap(text, width):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    return lines + [line] if line else lines


CLASSES = (OrrinLinkPreferences, VIEW3D_PT_orrin_link)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.orrin_link = BoolProperty(
        name="Stream to Orrin",
        description="Stream this file to the Orrin editor while it is edited",
        get=lambda manager: _session is not None,
        set=_set_enabled,
    )
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)
    if not bpy.app.background:
        bpy.app.timers.register(_autostart, first_interval=0.1)


def unregister():
    stop()
    if bpy.app.timers.is_registered(_autostart):
        bpy.app.timers.unregister(_autostart)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    del bpy.types.WindowManager.orrin_link
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
