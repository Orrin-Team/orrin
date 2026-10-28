//! Per-object matrices that live on the GPU between frames.
//!
//! Every geometry pass reads the same two matrices per object — the model and
//! where the object stood last frame. The normal matrix is derived from the
//! model in the vertex shader (`normal_matrix.glsl`) rather than carried. The old shape wrote
//! all three into a fresh streaming buffer every frame, once per *list the object
//! appeared in*: an object the camera can see that also casts into four cascades
//! was written five times, 192 bytes each, whether or not it had moved since the
//! last frame. A static scene paid full upload bandwidth for matrices that were
//! bit-for-bit what the GPU already held.
//!
//! Here the rows are a device-local buffer indexed by *entity slot*, which is
//! stable for as long as the entity lives, so:
//!
//! - an object occupies one row no matter how many lists draw it, and
//! - a row is written only on the frames where its matrices actually changed.
//!
//! What each list carries instead is four bytes per entry — the row number — in
//! [`InstanceLists`], which is still streamed per frame because the lists
//! themselves change every frame. The vertex shaders gained one indirection for
//! it: `objects[instances[push.object_base + gl_InstanceIndex]]` rather than
//! `objects[push.object_base + gl_InstanceIndex]`.
//!
//! # Why a staged copy rather than a mapped write
//!
//! The rows outlive the frame that wrote them, so writing them in place would be
//! writing memory the previous frame may still be reading — `render_frame` only
//! calls `cleanup_finished` and never waits, so the CPU is free to record frame
//! N+1 while the GPU executes frame N. Instead the dirty rows go into a
//! per-frame staging subbuffer, which the allocator recycles only once the GPU
//! is done with it, and a `copy_buffer` recorded at the top of the frame's
//! command buffer moves them across. The copy is a command like any other, so
//! the ordering against the passes that read the rows is the same
//! synchronisation everything else in this renderer relies on.

use bytemuck::Zeroable;
use glam::Mat4;

use crate::gfx::RenderItem;
use crate::gfx::graph::BufferUsage;
use crate::gfx::rhi::{
    BlockKind, BufferCopy, BufferDesc, BufferId, Device, Encoder, MemoryLocation,
};

/// Per-object transforms, indexed through [`InstanceLists`] from a storage
/// buffer. std430 matches this `#[repr(C)]` layout exactly because every field
/// is a 64-byte `mat4` (a multiple of 16).
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuObject {
    model: [[f32; 4]; 4],
    /// Last frame's `model`, for the motion vector the prepass writes. Uploaded
    /// for every pass rather than only the one that reads it: the buffer is
    /// shared, so the row's stride is shared too, and a second layout for the
    /// passes that ignore this field would be two ways for one object row to be
    /// wrong.
    prev_model: [[f32; 4]; 4],
}

/// How many rows a fresh store holds before the first growth. Large enough that
/// the built-in scenes never reallocate and small enough to be noise on a device
/// that only ever draws a handful of objects.
const INITIAL_ROWS: u32 = 1024;

/// What the GPU is believed to hold for one row.
///
/// The two source matrices rather than the packed [`GpuObject`], because these
/// are what the dirty test compares and `normal_matrix` is a function of `model`
/// — if the model matrix is unchanged the inverse-transpose is too, so storing
/// it would be a third of the memory for none of the answer.
///
/// Deliberately *not* keyed by entity generation. A slot reused by a different
/// entity that happens to stand exactly where the last one did needs no upload:
/// the row's contents are fully determined by this pair, so equal pairs mean the
/// row on the GPU is already right, whoever put it there.
#[derive(Clone, Copy, PartialEq)]
struct Resident {
    model: Mat4,
    prev_model: Mat4,
}

/// One entry of a draw order: which object row to draw, and which material to
/// draw it with.
///
/// The material rides here rather than in a push constant because a multi-draw
/// has no per-draw push — every batch of a pipeline variant is one command, and
/// the only thing that varies between them that a shader can read is what the
/// instance points at. The CPU path writes the same pairs so that one vertex
/// shader serves both. See [`cull`](super::cull).
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub(super) struct InstanceEntry {
    pub row: u32,
    pub material: u32,
}

/// This frame's draw-order lists, as row numbers into [`InstanceStore`], and
/// where each list's block begins.
///
/// One buffer for every geometry pass in the frame, which is what
/// `instance_index` says in the graph. The opaque camera list indexes from zero
/// and needs no base.
pub(super) struct InstanceLists {
    pub indices: BufferId,
    /// Where the blended items' indices start.
    pub transparent_base: u32,
    /// Where the refractive items' indices start.
    pub refractive_base: u32,
    pub cascade_bases: [u32; crate::gfx::shadows::MAX_CASCADES],
    pub punctual_bases: [u32; crate::gfx::punctual::MAX_SHADOW_LIGHTS],
}

/// The persistent row buffer, plus the mirror the dirty test reads.
pub(super) struct InstanceStore {
    /// Device-local, `TRANSFER_DST`: written only by the staged copy below.
    rows: BufferId,
    /// How many rows `rows` holds. Kept alongside rather than read back off the
    /// buffer so that growth is a comparison against a `u32` slot.
    capacity: u32,
    /// What the GPU holds, per row, or `None` for a row never written. Indexed
    /// exactly as `rows` is.
    resident: Vec<Option<Resident>>,
    /// Reused each frame for the dirty rows and for the row numbers each list
    /// draws, so a frame that moves nothing allocates nothing.
    staged: Vec<GpuObject>,
    entries: Vec<InstanceEntry>,
}

impl InstanceStore {
    pub(super) fn new(device: &Device) -> Self {
        Self {
            rows: allocate_rows(device, INITIAL_ROWS),
            capacity: INITIAL_ROWS,
            resident: vec![None; INITIAL_ROWS as usize],
            staged: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// The row buffer every geometry pass binds. Valid only after [`Self::sync`]
    /// has run for this frame, which is where a growth would have replaced it.
    pub(super) fn rows(&self) -> BufferId {
        self.rows
    }

    /// Bring the GPU's rows up to date with `items`, recording the copy into
    /// `builder`.
    ///
    /// Returns how many rows had to be written, which is what the perf harness
    /// reads to tell a static scene from a moving one.
    pub(super) fn sync(
        &mut self,
        device: &Device,
        encoder: &mut Encoder,
        items: &[RenderItem],
    ) -> usize {
        // Growth first, because the dirty test below is against the mirror this
        // may have just cleared. `+ 1` since `instance` is an index.
        let needed = items
            .iter()
            .map(|item| item.instance + 1)
            .max()
            .unwrap_or(0);
        if needed > self.capacity {
            // Next power of two, so a scene that spawns steadily reallocates a
            // logarithmic number of times rather than once per growth.
            let capacity = needed.next_power_of_two();
            device.destroy_buffer(self.rows);
            self.rows = allocate_rows(device, capacity);
            self.capacity = capacity;
            // A new buffer holds nothing, so every row is dirty again. Cleared
            // rather than resized: the old contents describe the old buffer.
            self.resident.clear();
            self.resident.resize(capacity as usize, None);
        }

        // The dirty rows, packed into the staging buffer in the order they are
        // found, each remembering which row it is bound for.
        let mut dirty: Vec<(u32, GpuObject)> = Vec::new();
        for item in items {
            let slot = item.instance as usize;
            let current = Resident {
                model: item.model,
                prev_model: item.prev_model,
            };
            if self.resident[slot] == Some(current) {
                continue;
            }
            self.resident[slot] = Some(current);
            dirty.push((
                item.instance,
                GpuObject {
                    model: item.model.to_cols_array_2d(),
                    prev_model: item.prev_model.to_cols_array_2d(),
                },
            ));
        }

        if dirty.is_empty() {
            return 0;
        }

        // Sorted so that rows adjacent on the GPU are adjacent in the staging
        // buffer too, which is what lets the run-coalescing below turn a scene
        // that moved wholesale into one region rather than thousands.
        dirty.sort_unstable_by_key(|&(slot, _)| slot);

        self.staged.clear();
        self.staged.extend(dirty.iter().map(|&(_, row)| row));
        let staging = device.upload_block(BlockKind::Staging, bytemuck::cast_slice(&self.staged));

        // One region per maximal run of consecutive rows. A frame in which
        // nothing moved records no copy at all; one in which everything moved
        // records a single region.
        let row_bytes = std::mem::size_of::<GpuObject>() as u64;
        let mut regions: Vec<BufferCopy> = Vec::new();
        let mut start = 0usize;
        while start < dirty.len() {
            let mut end = start + 1;
            while end < dirty.len() && dirty[end].0 == dirty[end - 1].0 + 1 {
                end += 1;
            }
            regions.push(BufferCopy {
                src_offset: start as u64 * row_bytes,
                dst_offset: u64::from(dirty[start].0) * row_bytes,
                size: (end - start) as u64 * row_bytes,
            });
            start = end;
        }

        encoder.copy_buffer(staging, self.rows, &regions);

        dirty.len()
    }

    /// Write this frame's draw orders as row numbers.
    ///
    /// The opaque `visible` list goes first so the forward and prepass passes
    /// keep indexing from zero; the blended ones follow, then each cascade's
    /// casters, then each punctual light's, and the returned bases say where.
    /// One buffer rather than one per list is what keeps `instance_index` a
    /// single resource in the graph rather than a convenient fiction.
    pub(super) fn upload_lists(
        &mut self,
        device: &Device,
        visible: crate::gfx::DrawList<'_>,
        transparent: crate::gfx::DrawList<'_>,
        refractive: crate::gfx::DrawList<'_>,
        casters: &[crate::gfx::DrawList<'_>],
        punctual: &[crate::gfx::DrawList<'_>],
    ) -> InstanceLists {
        let entries = &mut self.entries;
        entries.clear();
        let push = |entries: &mut Vec<InstanceEntry>, list: &crate::gfx::DrawList<'_>| {
            for i in 0..list.len() {
                let item = list.item(i);
                entries.push(InstanceEntry {
                    row: item.instance,
                    material: item.material.0,
                });
            }
        };

        push(entries, &visible);
        let transparent_base = entries.len() as u32;
        push(entries, &transparent);
        let refractive_base = entries.len() as u32;
        push(entries, &refractive);
        let mut cascade_bases = [0u32; crate::gfx::shadows::MAX_CASCADES];
        for (base, list) in cascade_bases.iter_mut().zip(casters) {
            *base = entries.len() as u32;
            push(entries, list);
        }
        let mut punctual_bases = [0u32; crate::gfx::punctual::MAX_SHADOW_LIGHTS];
        for (base, list) in punctual_bases.iter_mut().zip(punctual) {
            *base = entries.len() as u32;
            push(entries, list);
        }

        // A zero-length buffer is not one a descriptor can bind, and an empty
        // scene still needs one, so it is padded to a single unread entry.
        if entries.is_empty() {
            entries.push(InstanceEntry::zeroed());
        }
        InstanceLists {
            indices: device.upload_block(BlockKind::Storage, bytemuck::cast_slice(entries)),
            transparent_base,
            refractive_base,
            cascade_bases,
            punctual_bases,
        }
    }
}

/// Device-local rows, written only by the staged copy in [`InstanceStore::sync`].
fn allocate_rows(device: &Device, rows: u32) -> BufferId {
    device.create_buffer(BufferDesc {
        size: u64::from(rows) * std::mem::size_of::<GpuObject>() as u64,
        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
        location: MemoryLocation::GpuOnly,
    })
}

#[cfg(test)]
mod tests {
    use glam::{Mat3, Quat, Vec3};

    use super::*;
    use crate::scene::Transform;

    /// `normal_matrix` in `normal_matrix.glsl`, here so its algebra can be
    /// asserted without a GPU. The two are a lock-step pair.
    fn shader_normal_matrix(model: &Mat4) -> Mat3 {
        let m = Mat3::from_mat4(*model);
        let cofactor = Mat3::from_cols(
            m.y_axis.cross(m.z_axis),
            m.z_axis.cross(m.x_axis),
            m.x_axis.cross(m.y_axis),
        );
        cofactor * m.determinant().signum_or_zero()
    }

    trait SignumOrZero {
        fn signum_or_zero(self) -> f32;
    }

    /// GLSL's `sign`, which is zero at zero where Rust's `signum` is one.
    impl SignumOrZero for f32 {
        fn signum_or_zero(self) -> f32 {
            if self == 0.0 { 0.0 } else { self.signum() }
        }
    }

    fn assert_same_direction(a: Vec3, b: Vec3) {
        assert!(
            (a.normalize() - b.normalize()).length() < 1e-5,
            "{a} and {b} point different ways",
        );
    }

    /// The shader's matrix differs from the inverse-transpose by a positive
    /// scale only, which the shader's `normalize` removes — so wherever the
    /// `R * S^-1` closed form was valid, the lit normal is the same.
    #[test]
    fn the_normal_matrix_matches_the_closed_form_for_a_trs() {
        let transform = Transform {
            translation: Vec3::new(3.0, -2.0, 7.0),
            rotation: Quat::from_euler(glam::EulerRot::YXZ, 0.6, -1.1, 0.3),
            scale: Vec3::new(0.5, 4.0, 2.0),
        };
        let oracle =
            Mat3::from_quat(transform.rotation) * Mat3::from_diagonal(transform.scale.recip());
        let derived = shader_normal_matrix(&transform.matrix());
        for normal in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(0.3, -0.8, 0.5)] {
            assert_same_direction(derived * normal, oracle * normal);
        }
    }

    /// The cofactor matrix is the inverse-transpose times the determinant, and
    /// a mirrored transform's determinant is negative — without the sign every
    /// normal on a mirrored object would face inward.
    #[test]
    fn a_mirrored_transform_keeps_its_normals_facing_out() {
        let mirrored = Transform {
            rotation: Quat::from_rotation_y(0.4),
            scale: Vec3::new(-1.0, 2.0, 1.0),
            ..Default::default()
        }
        .matrix();
        let oracle = Mat3::from_mat4(mirrored).inverse().transpose();
        let derived = shader_normal_matrix(&mirrored);
        for normal in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert_same_direction(derived * normal, oracle * normal);
        }
    }

    /// A flattened object has no normal to speak of; the matrix must still be
    /// finite, because NaNs would spread through the shader's lighting math.
    #[test]
    fn a_zero_scale_axis_stays_finite() {
        let transform = Transform {
            scale: Vec3::new(1.0, 0.0, 1.0),
            ..Default::default()
        };
        assert!(
            shader_normal_matrix(&transform.matrix())
                .to_cols_array()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// A shear that no translation/rotation/scale triple can express still
    /// yields correct normals.
    #[test]
    fn the_normal_matrix_handles_a_sheared_matrix() {
        let sheared = Mat4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let derived = shader_normal_matrix(&sheared);
        // A normal is correct when it stays perpendicular to a tangent the same
        // matrix carried: n' . (M t) == 0 for every tangent t of the surface.
        let tangent = Mat3::from_mat4(sheared) * Vec3::new(1.0, 0.0, 0.0);
        let normal = derived * Vec3::new(0.0, 1.0, 0.0);
        assert!(
            normal.dot(tangent).abs() < 1e-5,
            "normal {normal:?} is not perpendicular to tangent {tangent:?}"
        );
    }

    /// Two matrices per row, and std430 lays them out with no padding — the
    /// stride every `Object` declaration in the shaders has to agree with.
    #[test]
    fn an_object_row_is_two_matrices() {
        assert_eq!(std::mem::size_of::<GpuObject>(), 128);
    }
}
