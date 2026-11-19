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
//! # Growth and reuse
//!
//! A loaded mesh's span is stable until the mesh is replaced, and then it is
//! retired: once the frames that may still draw from it have retired, later
//! uploads land in its space. Where a mesh goes and when retired space is free
//! again are `rhi::geometry`'s arithmetic, shared with Metal; an upload takes
//! free space that fits and moves the end only when none does.
//!
//! When a buffer runs out, a larger one is allocated and the old contents are
//! copied into its front, which leaves every span already handed out still
//! correct. The old buffer stays alive as long as a recorded frame holds it:
//! [`Recorder::bind_vertex_buffers`](super::record::Recorder) keeps what it
//! binds. The copy is not waited for, which reuse makes a hazard on one path
//! only. A staged upload is a copy too, ordered behind the growth by the
//! barrier every arena copy ends with. A wide-BAR upload is a host write the
//! queue knows nothing of, so one placed below the end — space the copy may
//! still be writing — waits for the copies in flight first; one placed at the
//! end needs nothing, since the end never comes back down.
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
use super::record::{self, InFlight, Pending, Recorder};

/// Vertices and indices a fresh arena holds before the first growth. Small
/// enough that a scene of a few cubes does not reserve megabytes, and a power of
/// two so the growth below never has to round.
const INITIAL_VERTICES: u64 = 1 << 14;
const INITIAL_INDICES: u64 = 1 << 15;

/// Positions and indices are also storage buffers, for the wireframe, which
/// fetches its own vertices so each corner knows which corner it is.
const POSITION_USAGE: BufferUsage = BufferUsage::VERTEX_BUFFER.union(BufferUsage::STORAGE_BUFFER);
const INDEX_USAGE: BufferUsage = BufferUsage::INDEX_BUFFER.union(BufferUsage::STORAGE_BUFFER);

/// The shared geometry every opaque and blended pass binds.
pub(crate) struct MeshArena {
    positions: Subbuffer<[PositionVertex]>,
    surfaces: Subbuffer<[SurfaceVertex]>,
    indices: Subbuffer<[u32]>,
    placement: Placement,
    /// Whether the arena is memory the CPU can write into. Decided once, by the
    /// same BAR-width probe that used to decide it per mesh.
    host_visible: bool,
    /// Copies submitted and not yet known to be finished — growth and staged
    /// uploads, neither of which the CPU waits for. Each holds what it read
    /// until its fence signals: the outgrown buffer, the staging.
    pending: Vec<InFlight>,
}

impl MeshArena {
    pub(super) fn new(ctx: &VkContext, frames_in_flight: u64) -> Self {
        let host_visible = ctx.profile.wide_bar();
        Self {
            positions: allocate(ctx, POSITION_USAGE, INITIAL_VERTICES, host_visible),
            surfaces: allocate(
                ctx,
                BufferUsage::VERTEX_BUFFER,
                INITIAL_VERTICES,
                host_visible,
            ),
            indices: allocate(ctx, INDEX_USAGE, INITIAL_INDICES, host_visible),
            placement: Placement::new(frames_in_flight),
            host_visible,
            pending: Vec::new(),
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

    /// One of the buffers as bytes, for a shader that reads it as storage.
    pub(super) fn storage(&self, stream: crate::gfx::rhi::GeometryStream) -> Subbuffer<[u8]> {
        match stream {
            crate::gfx::rhi::GeometryStream::Positions => self.positions.clone().into_bytes(),
            crate::gfx::rhi::GeometryStream::Indices => self.indices.clone().into_bytes(),
        }
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

        let held = self.placement.ends();
        let span = self
            .placement
            .place(positions.len() as u32, indices.len() as u32);
        self.reserve(ctx, held, self.placement.ends());

        let vertex_offset = span.vertex_offset as u64;
        if self.host_visible {
            let reused = (!positions.is_empty() && (span.vertex_offset as u32) < held.0)
                || (!indices.is_empty() && span.first_index < held.1);
            // Below the end a growth copy still on the queue may be writing.
            if reused {
                self.settle();
            }
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
            self.submit(ctx, builder);
        }

        (span, bounds)
    }

    /// Retire a replaced mesh's span as of `frame`. See [`Placement::retire`].
    pub(super) fn retire(&mut self, span: MeshSpan, vertices: u32, frame: u64) {
        self.placement.retire(span, vertices, frame);
    }

    /// Return to the arena every retired span whose frames are done.
    pub(super) fn collect(&mut self, frame: u64) {
        self.placement.collect(frame);
    }

    #[cfg(test)]
    pub(super) fn ends(&self) -> (u32, u32) {
        self.placement.ends()
    }

    /// Grow whichever of the three buffers is short of `needed` — vertices,
    /// indices — copying the `held` they already hold across.
    fn reserve(&mut self, ctx: &VkContext, held: (u32, u32), needed: (u32, u32)) {
        let held_vertices = u64::from(held.0);
        let held_indices = u64::from(held.1);
        let positions = grown(self.positions.len(), u64::from(needed.0));
        let surfaces = grown(self.surfaces.len(), u64::from(needed.0));
        let index_capacity = grown(self.indices.len(), u64::from(needed.1));
        if positions.is_none() && surfaces.is_none() && index_capacity.is_none() {
            return;
        }

        // One command buffer for up to three copies, and no wait: growth is
        // rare, but it happens while a scene is loading, a few times over. The
        // copies fill `0..held` of the new buffers while the host goes on
        // writing the spans placed past `held`, so the two never touch the same
        // bytes; a span placed below it waits for them first (`upload`).
        let mut builder = Recorder::new(ctx);
        if let Some(capacity) = positions {
            self.positions = regrow(
                ctx,
                &mut builder,
                &self.positions,
                POSITION_USAGE,
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
                INDEX_USAGE,
                capacity,
                held_indices,
                self.host_visible,
            );
        }
        self.submit(ctx, builder);
    }

    /// Submit an arena copy without waiting for it, ending it with the barrier
    /// that orders it before anything later on the queue reads the arena.
    fn submit(&mut self, ctx: &VkContext, mut builder: Recorder) {
        builder.transfer_to_geometry_reads();
        let (commands, keep) = builder.end();
        self.pending.retain(|copy| !copy.is_complete());
        self.pending.push(record::submit_frame(
            ctx,
            &ctx.queue,
            commands,
            keep,
            &[],
            &[],
        ));
    }

    /// Block until every arena copy on the queue has finished.
    fn settle(&mut self) {
        for copy in self.pending.drain(..) {
            copy.retire();
        }
    }
}

impl Drop for MeshArena {
    /// What a copy reads must outlive it, and the device must outlive that.
    fn drop(&mut self) {
        for copy in self.pending.drain(..) {
            copy.retire();
        }
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

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use orrin_ecs::World;

    use super::super::VulkanRenderer;
    use crate::capture::{self, CaptureSettings};
    use crate::gfx::{Material, RenderBackend, Vertex};
    use crate::scene::{Camera, CpuMesh, LocalTransform, MeshHandle, Transform};

    /// A unit square facing the camera, centred `x` along from the middle of
    /// the frame. Every one is the same size, which is the case a streamed
    /// mesh is in between edits.
    fn square(x: f32) -> CpuMesh {
        let corner = |dx: f32, dy: f32| Vertex {
            position: [x + dx, dy, 0.0],
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 3],
            uv: [dx + 0.5, dy + 0.5],
            tangent: [1.0, 0.0, 0.0, 1.0],
        };
        CpuMesh {
            vertices: vec![
                corner(-0.5, -0.5),
                corner(0.5, -0.5),
                corner(0.5, 0.5),
                corner(-0.5, 0.5),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
    }

    fn centre_is_red(frame: &(Vec<u8>, [u32; 2])) -> (bool, [u8; 3]) {
        let (pixels, [width, height]) = frame;
        let at = ((height / 2 * width + width / 2) * 4) as usize;
        let [r, g, b] = [pixels[at], pixels[at + 1], pixels[at + 2]];
        (
            r > g.saturating_add(40) && r > b.saturating_add(40),
            [r, g, b],
        )
    }

    /// A square replaced over and over, alternately across the middle of the
    /// frame and beside it, through the real renderer: the arena stops growing
    /// within a few replaces, later replaces land in space earlier ones held,
    /// and what each frame draws and what `mesh_bounds` says are the new
    /// square's — so nothing draws through a span the replace gave back.
    fn a_replaced_mesh_reuses_its_arena_space(mut renderer: VulkanRenderer) {
        crate::logging::init(crate::logging::LogConfig::for_this_build());
        let settings = CaptureSettings {
            extent: [64, 64],
            frames: 2,
            taa: false,
            camera: Some(Camera {
                position: Vec3::new(0.0, 0.0, 3.0),
                target: Vec3::ZERO,
                ..Camera::default()
            }),
            ..CaptureSettings::default()
        };
        let mut world = World::new();
        capture::prepare(&mut world, &mut renderer, &settings);
        let drawn: Vec<_> = world
            .entities()
            .filter(|&entity| world.has::<MeshHandle>(entity))
            .collect();
        for entity in drawn {
            world.despawn(entity);
        }

        let red = renderer.load_material(&Material {
            base_color: Vec3::new(1.0, 0.0, 0.0),
            metallic: 0.0,
            roughness: 1.0,
            ..Material::default()
        });
        let mesh = renderer.load_mesh(&square(0.0));
        let entity = world.spawn();
        world.insert(entity, LocalTransform::new(Transform::default()));
        world.insert(entity, mesh);
        world.insert(entity, red);
        let (covered, pixel) =
            centre_is_red(&capture::render(&mut world, &mut renderer, &settings));
        assert!(covered, "the square is drawn across the middle: {pixel:?}");

        let loaded = renderer.device.geometry_ends();
        let mut ends = Vec::new();
        let mut spans = Vec::new();
        for replace in 0..24 {
            let x = if replace % 2 == 0 { 2.0 } else { 0.0 };
            let next = square(x);
            renderer.replace_mesh(mesh, &next);
            let (covered, pixel) =
                centre_is_red(&capture::render(&mut world, &mut renderer, &settings));
            assert_eq!(
                covered,
                x == 0.0,
                "replace {replace} draws the square at x = {x}: {pixel:?}"
            );
            assert_eq!(renderer.mesh_bounds(mesh), Some(next.bounds()));
            ends.push(renderer.device.geometry_ends());
            spans.push(renderer.meshes[mesh.0 as usize].span);
        }

        println!("arena ends with the square loaded: {loaded:?}, after each replace: {ends:?}");
        assert!(
            ends[8..].iter().all(|&end| end == ends[8]),
            "the arena kept growing: {ends:?}"
        );
        assert!(
            spans[8..].iter().all(|span| spans[..8].contains(span)),
            "later replaces land where earlier ones did: {spans:?}"
        );
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn a_replaced_mesh_reuses_its_arena_space_on_vulkan() {
        a_replaced_mesh_reuses_its_arena_space(capture::offscreen_renderer([64, 64]));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_replaced_mesh_reuses_its_arena_space_on_metal() {
        a_replaced_mesh_reuses_its_arena_space(
            VulkanRenderer::offscreen_metal([64, 64]).expect("a Metal device"),
        );
    }
}
