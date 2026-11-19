"""What the live link streams: the scene as `blend::EXPORT` exports it.

The engine's `.blend` import runs Blender's glTF exporter with
`export_apply`, `use_visible` and `export_yup`, and merges every node into one
mesh. This is that, read from the depsgraph in memory, so the stream and the
saved file agree and nothing moves on save or when the link lets go:

- every object the exporter's visibility filter keeps, of a type it converts to
  a mesh, plus what a visible collection, vertex or face instancer instances;
- evaluated, so with its modifiers, and in Edit Mode the edit mesh as the
  viewport draws it, which the depsgraph evaluates without leaving the mode;
- in world space and glTF's conventions: positions and normals to +Y up, V
  flipped, and a mirroring transform's triangles wound back, as the import
  winds a mirroring node's;
- one vertex per distinct (vertex, normal, UV) of each object, as the exporter
  welds them, with its normals rounded to the exporter's four digits first; a
  flat-shaded mesh is the exception `weld` describes.

Everything is `foreach_get` into numpy, and the weld mostly takes no sort: a
pass keeps, per vertex, the corners equal to one chosen corner, so a smooth
surface is welded in one. A 100k-triangle UV sphere took 9.0 ms smooth and
13.5 ms flat on an M5 Pro with Blender 5.2.3.
"""

import numpy as np

# What the exporter turns into a mesh. A metaball is not among them, so it is
# not here either.
CONVERTED = frozenset({"MESH", "CURVE", "SURFACE", "FONT"})

# The instancers the exporter follows. A geometry-nodes instance is left out,
# as the exporter leaves it out without `export_gn_mesh`.
INSTANCERS = frozenset({"COLLECTION", "VERTS", "FACES"})

# io_scene_gltf2's ROUNDING_DIGIT, which its normals are rounded to before
# they are welded.
ROUNDING = 4

# Passes before the weld sorts what is left: enough for a seam's two corners or
# a hard edge's three, not for a pole's one corner a segment.
PASSES = 4


class Mesh:
    """One merged mesh, laid out as a `mesh` frame carries it."""

    __slots__ = ("positions", "normals", "uvs", "indices")

    def __init__(self, positions, normals, uvs, indices):
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.indices = indices

    @property
    def triangles(self):
        return len(self.indices) // 3


def exported(ob):
    """The exporter's `use_visible` filter on an object of the scene."""
    if not ob.visible_get() or ob.hide_viewport:
        return False
    collections = ob.users_collection
    return not collections or not all(c.hide_viewport for c in collections)


def extract(depsgraph):
    """The visible scene of `depsgraph` as one mesh, or `None` when nothing
    visible has a triangle."""
    parts = []
    for instance in depsgraph.object_instances:
        ob = instance.object
        if ob.type not in CONVERTED:
            continue
        if instance.is_instance:
            owner = instance.parent.original
            if owner.instance_type not in INSTANCERS or not exported(owner):
                continue
        else:
            original = ob.original
            if not exported(original):
                continue
            if original.is_instancer and not original.show_instancer_for_render:
                continue
        part = _part(ob, np.array(instance.matrix_world, dtype=np.float32))
        if part is not None:
            parts.append(part)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    base = 0
    indices = []
    for part in parts:
        indices.append(part.indices + np.uint32(base))
        base += len(part.positions)
    return Mesh(
        np.concatenate([p.positions for p in parts]),
        np.concatenate([p.normals for p in parts]),
        np.concatenate([p.uvs for p in parts]),
        np.concatenate(indices),
    )


def _part(ob, matrix):
    mesh = ob.to_mesh()
    try:
        return _read(mesh, matrix) if mesh is not None else None
    finally:
        ob.to_mesh_clear()


def _read(mesh, matrix):
    triangles = len(mesh.loop_triangles)
    if not triangles:
        return None
    corners = len(mesh.loops)
    # Each array in the type Blender stores it as, which `foreach_get` copies
    # whole; any other costs a Python call an element, 5 ms a 100k mesh here.
    tri = np.empty(triangles * 3, np.uint32)
    mesh.loop_triangles.foreach_get("loops", tri)
    vertex_of = np.empty(corners, np.int32)
    mesh.attributes[".corner_vert"].data.foreach_get("value", vertex_of)
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co)
    co = co.reshape(-1, 3)

    # A corner's normal and UV side by side, as the weld compares them, padded
    # to 24 bytes so a row compares as three 64-bit words.
    key = np.zeros((corners, 6), np.float32)
    normals = np.empty(corners * 3, np.float32)
    mesh.corner_normals.foreach_get("vector", normals)
    key[:, :3] = _unit(np.round(normals.reshape(-1, 3), ROUNDING))
    if len(mesh.uv_layers):
        uv = np.empty(corners * 2, np.float32)
        mesh.uv_layers[0].uv.foreach_get("vector", uv)
        key[:, 3:5] = uv.reshape(-1, 2)

    source, out = weld(vertex_of, key, len(co))
    indices = out[tri].astype(np.uint32)

    linear = matrix[:3, :3]
    positions = co[vertex_of[source]] @ linear.T + matrix[:3, 3]
    det = np.linalg.det(linear)
    if det != 0:
        normals = _unit(key[source, :3] @ np.linalg.inv(linear))
    else:
        normals = np.zeros((len(source), 3), np.float32)
        normals[:, 2] = 1.0
    if det < 0:
        indices = indices.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)

    uvs = key[source, 3:5]
    uvs[:, 1] = 1.0 - uvs[:, 1]
    return Mesh(_y_up(positions), _y_up(normals), uvs, indices)


def _unit(v):
    """`v` normalised in place, a zero row made +Z, as the exporter does."""
    length = np.sqrt(np.einsum("ij,ij->i", v, v))[:, None]
    np.divide(v, length, out=v, where=length != 0)
    v[length[:, 0] == 0, 2] = 1.0
    return v


def _y_up(v):
    """Blender's +Z up to glTF's +Y up, as the exporter's `zup2yup`."""
    out = np.empty(v.shape, np.float32)
    out[:, 0] = v[:, 0]
    out[:, 1] = v[:, 2]
    out[:, 2] = -v[:, 1]
    return out


def weld(vertex_of, key, vertices):
    """One output vertex per distinct (vertex, key row): the corner each is
    read from, and each corner's output vertex.

    Rows compare by their bits, so a NaN still equals itself and every pass
    retires at least the corner it chose. A pass costs the same however few
    corners are left, and a UV sphere's pole has a corner for every segment, so
    what a few passes leave is sorted out at once instead.

    A mesh whose first pass leaves most corners is flat shaded, where corners
    that would weld are rare and the passes doubled a 100k-triangle extraction
    for 18 fewer vertices; its remaining corners keep a vertex each.
    """
    bits = np.ascontiguousarray(key).view(np.uint64)
    corners = len(vertex_of)
    out = np.empty(corners, np.int64)
    chosen = np.empty(vertices, np.int64)
    sources = []
    count = 0
    pending = np.arange(corners)
    for done in range(PASSES):
        if not pending.size:
            break
        if done == 1 and pending.size * 2 > corners:
            out[pending] = count + np.arange(pending.size)
            sources.append(pending)
            return np.concatenate(sources), out
        vertex = vertex_of[pending]
        # Whichever pending corner lands last for a vertex is its choice; the
        # read back agrees with the write, whatever numpy's order was.
        chosen[vertex] = pending
        choice = chosen[vertex]
        same = (bits[pending] == bits[choice]).all(axis=1)
        new = pending[choice == pending]
        out[new] = count + np.arange(new.size)
        count += new.size
        sources.append(new)
        out[pending[same]] = out[choice[same]]
        pending = pending[~same]
    if pending.size:
        rows = np.empty((pending.size, 1 + bits.shape[1]), np.uint64)
        rows[:, 0] = vertex_of[pending]
        rows[:, 1:] = bits[pending]
        _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
        out[pending] = count + inverse.reshape(-1)
        sources.append(pending[first])
    return np.concatenate(sources), out
