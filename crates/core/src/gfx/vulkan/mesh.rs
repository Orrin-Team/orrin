//! Every mesh's geometry, in three buffers rather than three per mesh.
//!
//! A mesh used to own its position, surface and index buffers, which meant a
//! pass had to rebind all three between one mesh and the next — so a draw could
//! only ever be one mesh, and the draw *order* had to be a CPU decision. That is
//! the constraint that kept [`cull`](super::cull) recording one indirect draw
//! per batch per view: the GPU could count a batch's instances, but only the
//! host could say which mesh the next draw was for.
//!
//! Here all meshes share one position buffer, one surface buffer and one index
//! buffer, and a mesh becomes three numbers into them. A pass binds the three
//! once and every draw after that names its geometry through `firstIndex` and
//! `vertexOffset` — fields `VkDrawIndexedIndirectCommand` already carries, which
//! is what lets one multi-draw cover every batch a view kept.
//!
//! # Growth
//!
//! Meshes are appended and never freed — `load_mesh` pushes and nothing pops —
//! so placement is a bump allocator and a mesh's span is stable for the life of
//! the renderer. When a buffer runs out, a larger one is allocated and the old
//! contents are copied into its front, which leaves every span already handed
//! out still correct. The old buffer stays alive as long as a recorded frame
//! holds it: [`Recorder::bind_vertex_buffers`](super::record::Recorder) keeps
//! what it binds.
//!
//! # Why the two upload paths survive
//!
//! Wide-BAR devices write into the arena directly; narrow-BAR ones stage through
//! host memory, exactly as they did when the buffers were per mesh — the reason
//! for the split is where the memory lives, which sharing one allocation does
//! not change. See [`allocate`].

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{BufferCopy, CopyBufferInfoTyped};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::geom::Aabb;
use crate::gfx::{PositionVertex, SurfaceVertex, Vertex};
use glam::Vec3;

use crate::gfx::rhi::{MeshSpan, Placement, grown};

use super::context::VkContext;
use super::record::Recorder;

/// Vertices and indices a fresh arena holds before the first growth. Small
/// enough that a scene of a few cubes does not reserve megabytes, and a power of
/// two so the growth below never has to round.
const INITIAL_VERTICES: u64 = 1 << 14;
const INITIAL_INDICES: u64 = 1 << 15;

/// The shared geometry every opaque and blended pass binds.
pub(crate) struct MeshArena {
    positions: Subbuffer<[PositionVertex]>,
    surfaces: Subbuffer<[SurfaceVertex]>,
    indices: Subbuffer<[u32]>,
    placement: Placement,
    /// Whether the arena is memory the CPU can write into. Decided once, by the
    /// same BAR-width probe that used to decide it per mesh.
    host_visible: bool,
}

impl MeshArena {
    pub(super) fn new(ctx: &VkContext) -> Self {
        let host_visible = ctx.profile.wide_bar();
        Self {
            positions: allocate(
                ctx,
                BufferUsage::VERTEX_BUFFER,
                INITIAL_VERTICES,
                host_visible,
            ),
            surfaces: allocate(
                ctx,
                BufferUsage::VERTEX_BUFFER,
                INITIAL_VERTICES,
                host_visible,
            ),
            indices: allocate(
                ctx,
                BufferUsage::INDEX_BUFFER,
                INITIAL_INDICES,
                host_visible,
            ),
            placement: Placement::default(),
            host_visible,
        }
    }

    /// Bind everything a pass that shades a surface reads, plus the indices.
    ///
    /// Once per pass rather than once per draw, which is the point of the
    /// arena. Vertex bindings are command-buffer state and a secondary starts
    /// with none, so this belongs at the top of whatever records the draws —
    /// not around them.
    pub(super) fn bind(&self, builder: &mut Recorder) {
        builder
            .bind_vertex_buffers(0, (self.positions.clone(), self.surfaces.clone()))
            .bind_index_buffer(self.indices.clone());
    }

    /// The same for a depth-only pass, which reads position and texture
    /// coordinate and never the surface stream. See [`PositionVertex`].
    pub(super) fn bind_positions(&self, builder: &mut Recorder) {
        builder
            .bind_vertex_buffers(0, self.positions.clone())
            .bind_index_buffer(self.indices.clone());
    }

    /// Put a mesh into the arena and say where it landed.
    ///
    /// Also derives the object-space bounds, because upload is the last place
    /// the vertex data exists on the CPU.
    pub(super) fn upload(
        &mut self,
        ctx: &VkContext,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> (MeshSpan, Aabb) {
        let bounds = Aabb::from_points(vertices.iter().map(|v| Vec3::from(v.position)));

        // De-interleaved here because upload is the last place a vertex exists
        // as the one authored struct, and the only place that has to know the
        // two GPU streams are halves of it.
        let (positions, surfaces): (Vec<_>, Vec<_>) = vertices.iter().map(Vertex::split).unzip();

        self.reserve(ctx, positions.len() as u64, indices.len() as u64);
        let span = self
            .placement
            .place(positions.len() as u32, indices.len() as u32);

        let vertex_offset = span.vertex_offset as u64;
        if self.host_visible {
            write_directly(&self.positions, vertex_offset, &positions);
            write_directly(&self.surfaces, vertex_offset, &surfaces);
            write_directly(&self.indices, u64::from(span.first_index), indices);
        } else {
            let mut builder = Recorder::new(ctx);
            stage(
                ctx,
                &mut builder,
                &self.positions,
                vertex_offset,
                &positions,
            );
            stage(ctx, &mut builder, &self.surfaces, vertex_offset, &surfaces);
            stage(
                ctx,
                &mut builder,
                &self.indices,
                u64::from(span.first_index),
                indices,
            );
            builder.submit_and_wait(ctx);
        }

        (span, bounds)
    }

    /// Make room for one more mesh, growing whichever of the three buffers is
    /// short and copying what is already in it across.
    fn reserve(&mut self, ctx: &VkContext, vertices: u64, indices: u64) {
        let held_vertices = u64::from(self.placement.vertices);
        let held_indices = u64::from(self.placement.indices);
        let positions = grown(self.positions.len(), held_vertices + vertices);
        let surfaces = grown(self.surfaces.len(), held_vertices + vertices);
        let index_capacity = grown(self.indices.len(), held_indices + indices);
        if positions.is_none() && surfaces.is_none() && index_capacity.is_none() {
            return;
        }

        // One command buffer for up to three copies, and one wait: growth is
        // rare, but it happens while a scene is loading, where a round trip per
        // buffer would be three.
        let mut builder = Recorder::new(ctx);
        if let Some(capacity) = positions {
            self.positions = regrow(
                ctx,
                &mut builder,
                &self.positions,
                BufferUsage::VERTEX_BUFFER,
                capacity,
                held_vertices,
                self.host_visible,
            );
        }
        if let Some(capacity) = surfaces {
            self.surfaces = regrow(
                ctx,
                &mut builder,
                &self.surfaces,
                BufferUsage::VERTEX_BUFFER,
                capacity,
                held_vertices,
                self.host_visible,
            );
        }
        if let Some(capacity) = index_capacity {
            self.indices = regrow(
                ctx,
                &mut builder,
                &self.indices,
                BufferUsage::INDEX_BUFFER,
                capacity,
                held_indices,
                self.host_visible,
            );
        }
        builder.submit_and_wait(ctx);
    }
}

/// A larger buffer holding what the old one held.
///
/// The copy is recorded rather than done host-side because the arena may be
/// memory the CPU can write but not usefully read.
#[allow(clippy::too_many_arguments)]
fn regrow<T: BufferContents + Copy>(
    ctx: &VkContext,
    builder: &mut Recorder,
    old: &Subbuffer<[T]>,
    usage: BufferUsage,
    capacity: u64,
    held: u64,
    host_visible: bool,
) -> Subbuffer<[T]> {
    let new = allocate::<T>(ctx, usage, capacity, host_visible);
    if held > 0 {
        builder.copy_buffer(CopyBufferInfoTyped {
            regions: [BufferCopy {
                size: held,
                ..Default::default()
            }]
            .into_iter()
            .collect(),
            ..CopyBufferInfoTyped::buffers(old.clone(), new.clone())
        });
    }
    new
}

/// One `memcpy` into the arena — the wide-BAR path, and the reason a mesh load
/// on such a device still submits nothing.
fn write_directly<T: BufferContents + Copy>(buffer: &Subbuffer<[T]>, offset: u64, data: &[T]) {
    let mut write = buffer.write().expect("the arena is host-visible");
    write[offset as usize..offset as usize + data.len()].copy_from_slice(data);
}

/// Record a copy of `data` into `buffer` at `offset`, through host memory the
/// device can read — the narrow-BAR path.
fn stage<T: BufferContents + Copy>(
    ctx: &VkContext,
    builder: &mut Recorder,
    buffer: &Subbuffer<[T]>,
    offset: u64,
    data: &[T],
) {
    if data.is_empty() {
        return;
    }
    let staging = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data.iter().copied(),
    )
    .expect("failed to allocate mesh staging buffer");

    builder.copy_buffer(CopyBufferInfoTyped {
        regions: [BufferCopy {
            dst_offset: offset,
            size: data.len() as u64,
            ..Default::default()
        }]
        .into_iter()
        .collect(),
        ..CopyBufferInfoTyped::buffers(staging, buffer.clone())
    });
}

/// One of the three arena buffers, put where the GPU can read it many times per
/// frame.
///
/// Static geometry is the one resource in the renderer that is written once and
/// then *re-read* by every pass in the frame — the prepass, the forward pass,
/// each shadow cascade, the atlas — which is what makes where it lives worth a
/// branch. The two vendor guides say the same thing about it from opposite
/// directions: AMD's, that device-local host-visible memory is for data "each
/// byte of which is accessed once by the GPU", and NVIDIA's, to look explicitly
/// for `DEVICE_LOCAL` when picking a memory type. Vertices are neither
/// write-once nor incidentally device-local.
///
/// The reason a branch beats picking one path is that both are right on some
/// machine:
///
/// - **Wide BAR** (Resizable BAR on, or a unified-memory part). The CPU writes
///   straight into video memory. The buffer is device-local *and* the upload is
///   one `memcpy` with no staging copy, no command buffer and no fence. Staging
///   here would be strictly worse: same destination, extra work.
/// - **Narrow BAR.** `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE` reads as *required*
///   host-visible and only *preferred* device-local, so it resolves to the
///   legacy 256 MiB aperture — and when that fills, vulkano falls back to the
///   next type satisfying the requirement, which is ordinary system RAM. The
///   failure is silent and it is the bad one: every cascade then re-reads the
///   scene's geometry across PCIe, every frame, for as long as the scene is
///   open. Staging costs one copy at load and buys VRAM residency.
///
/// This is the branch that a machine with Resizable BAR *cannot* show you. On
/// such a machine the first path is taken throughout; the second exists for the
/// configuration that is still common on NVIDIA desktops and on anything with
/// the option switched off in firmware.
///
/// `TRANSFER_SRC` as well as `TRANSFER_DST` because growth reads the buffer it
/// replaces; the narrow-BAR path needs the latter anyway.
fn allocate<T: BufferContents>(
    ctx: &VkContext,
    usage: BufferUsage,
    len: u64,
    host_visible: bool,
) -> Subbuffer<[T]> {
    let memory_type_filter = if host_visible {
        MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
    } else {
        // No host-access filter, which is the entire point: it leaves the plain
        // `DEVICE_LOCAL` type — the one the aperture does not cover — as the
        // best match.
        MemoryTypeFilter::PREFER_DEVICE
    };
    Buffer::new_slice::<T>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter,
            ..Default::default()
        },
        len,
    )
    .expect("failed to allocate the mesh arena")
}
