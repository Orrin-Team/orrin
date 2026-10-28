//! Deciding what each view draws, on the GPU.
//!
//! The CPU sweep in `systems::extract_geometry` answers the same question by
//! testing every renderable against the camera frustum, then against every
//! cascade, then against every shadowing light — a per-entity cost that is
//! invisible at a hundred objects and is most of the frame at ten thousand.
//! Here the tests are one compute dispatch over (instance, view), and what the
//! CPU keeps is only the part that is genuinely per entity: which rows the
//! dispatch is offered, and which batch each belongs to. That walk is still
//! O(entities) — it is a list build and a cached lookup rather than the
//! frustum, cascade and light tests it replaced, and it is what remains to be
//! made incremental.
//!
//! # What a batch is, and why the draw count does not change
//!
//! A batch is one (mesh, material) pair. It is the unit the passes already drew
//! in — `DrawList::runs` collapses an ordered list into exactly these — so the
//! number of draw calls a frame records is the same before and after this
//! module exists. That is deliberate: the recorded draw count was measured flat
//! at 182 for the demo scene whether 126 or 3197 objects were visible, so there
//! was never a draw-call win here to chase. What moves is the *instance count*
//! of each of those draws, from a number the CPU counted to one the GPU wrote.
//!
//! It also decides the shape. One indirect draw per batch keeps each batch's
//! mesh buffers, its pipeline variant and its material index where they are —
//! bound and pushed by the CPU, which is the only reason this does not also
//! require every mesh in one vertex buffer and every material reachable from a
//! shader. A single multi-draw over all batches would need both.
//!
//! # The layout the three buffers agree on
//!
//! Views are numbered: the camera first, then each cascade, then each punctual
//! face. `instance_index` is that numbering major — view `v` owns the block at
//! `v * stride`, and within it batch `b` owns the slice at `instance_base[b]`,
//! wide enough for every live instance of that batch. Nothing is compacted
//! across batches, so a slice is written by one batch's atomics alone and its
//! draw reads `instance_base[b]` as its base whatever the other batches did.
//!
//! `draw_commands` is numbered the same way, `v * batch_count + b`, so the
//! offset a pass hands `draw_indexed_indirect` is arithmetic rather than a
//! lookup.

use std::collections::HashMap;

use crate::geom::{Aabb, Frustum};
use glam::Mat4;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::DrawIndexedIndirectCommand;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};

use crate::gfx::rhi::{Binding, BindingResource, ComputePipelineDesc, Device, Encoder, PipelineId};
use crate::gfx::shaders::ShaderId;

use super::context::VkContext;
use super::instances::InstanceEntry;
use super::occlusion::OcclusionTest;
use super::record::Arena;
use crate::gfx::rhi::MeshSpan;

/// Invocations per workgroup, in both shaders. The cull dispatch is
/// `local_size_x` instances by one view, so a view is a workgroup row rather
/// than a second axis inside one.
const GROUP: u32 = 64;

/// What one view culls against.
///
/// Two shapes rather than six planes for everything, because a cascade is not a
/// frustum test. See `casts_into` in `cull.comp` and in `systems.rs`.
#[derive(Clone, Copy, Debug)]
pub(super) enum CullView {
    /// The camera, and each face of a punctual light: a closed volume, and a box
    /// outside it cannot be seen from inside it.
    Frustum(Frustum),
    /// A cascade, which keeps anything that reaches its box when swept toward
    /// the light — so it has no near plane, and an object between the sun and
    /// the cascade is exactly what casts into it.
    Cascade {
        light_view: Mat4,
        half_extent: f32,
        depth_range: f32,
    },
}

const KIND_FRUSTUM: u32 = 0;
const KIND_CASCADE: u32 = 1;

/// Layout-frozen against `View` in `cull.comp`.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuView {
    light_view: [[f32; 4]; 4],
    planes: [[f32; 4]; 6],
    kind: u32,
    half_extent: f32,
    depth_range: f32,
    base: u32,
}

/// Layout-frozen against `Batch` in both shaders.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuBatch {
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    instance_base: u32,
    /// Written into every instance entry this batch keeps, because the shader
    /// that draws it has no per-draw push constant to read it from.
    material: u32,
    /// std430 aligns a `vec4` to sixteen bytes and the five `u32`s above do not
    /// land on one. Named rather than left to the compiler: the shader's struct
    /// has the same hole, and one side changing it silently is what this
    /// prevents.
    _padding: [u32; 3],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
}

/// Layout-frozen against `Instance` in `cull.comp`.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuInstance {
    row: u32,
    batch: u32,
}

/// One batch as the passes need it: which geometry it draws, beside where its
/// instances will be.
#[derive(Clone, Copy, Debug)]
pub(super) struct Batch {
    pub material: u32,
    /// Where this batch's slice begins inside a view's block.
    pub instance_base: u32,
    /// Where its mesh sits in the shared buffers. Zero-length for a batch whose
    /// mesh has not been uploaded, which draws nothing rather than being a
    /// special case in each of the four passes.
    span: MeshSpan,
    /// The mesh's object-space bounds. A batch is one mesh, so this is where
    /// the box is stored once rather than once per instance.
    bounds: Aabb,
}

/// What one frame's `prepare` worked out, and what the passes read back off it.
pub(super) struct CullFrame {
    /// Every batch, the ones drawn by the plain pipeline first. That order is
    /// what makes a pipeline variant a contiguous range of `draw_commands` and
    /// therefore one multi-draw — see [`CullFrame::regions`].
    pub batches: Vec<Batch>,
    /// How many of them are plain, which is where the masked ones begin.
    pub plain: usize,
    /// How wide one view's block of `instance_index` is.
    pub stride: u32,
    pub views: u32,
}

impl CullFrame {
    /// Where batch `batch` of view `view` starts in `instance_index`, which is
    /// the `firstInstance` its command carries.
    #[allow(dead_code, reason = "states the layout the culling shaders index by")]
    pub fn object_base(&self, view: u32, batch: usize) -> u32 {
        view * self.stride + self.batches[batch].instance_base
    }

    /// Where that batch's draw command sits in `draw_commands`.
    #[allow(dead_code, reason = "states the layout the culling shaders index by")]
    fn command_index(&self, view: u32, batch: usize) -> u64 {
        view as u64 * self.batches.len() as u64 + batch as u64
    }

    /// The two command ranges one view draws, plain first.
    ///
    /// A range covers every batch of its variant, including the ones this view
    /// culled to nothing: their commands were reset to zero instances and cost
    /// the command processor a skipped draw rather than the host a decision.
    /// That is the whole trade — the count is known before the dispatch runs,
    /// so recording does not have to wait for it.
    fn ranges(&self, view: u32) -> [(bool, std::ops::Range<u64>); 2] {
        let base = view as u64 * self.batches.len() as u64;
        let plain = self.plain as u64;
        let total = self.batches.len() as u64;
        [
            (false, base..base + plain),
            (true, base + plain..base + total),
        ]
    }
}

/// What one geometry pass draws: an ordered list the CPU culled, or the batches
/// a dispatch culled for one view.
///
/// The two are one type so that a pass takes one argument and asks it for both
/// shapes. Exactly one of [`Draws::units`] and [`Draws::regions`] ever yields
/// anything, and which it is is the culling path, so a pass writes the loop for
/// each and neither knows which it is in.
#[derive(Clone, Copy)]
pub(super) enum Draws<'a> {
    Cpu {
        list: crate::gfx::DrawList<'a>,
        /// Where this list's block of `instance_index` begins, which the CPU
        /// path passes through as each run's `firstInstance`.
        base: u32,
    },
    Gpu {
        frame: &'a CullFrame,
        view: u32,
        commands: &'a Subbuffer<[DrawIndexedIndirectCommand]>,
    },
}

/// One CPU-culled draw: which geometry, and how many instances of it.
pub(super) struct DrawUnit<'a> {
    pub mesh: u32,
    pub material: u32,
    /// The `firstInstance` of the draw. `gl_InstanceIndex` counts from it, and
    /// the entry found there says which object row and which material.
    pub object_base: u32,
    pub instances: u32,
    /// The items this unit draws, for a caller that wants to narrow it further
    /// — which the punctual atlas does, per face.
    pub run: Option<(crate::gfx::DrawList<'a>, std::ops::Range<usize>)>,
}

/// One multi-draw: every command of one pipeline variant, for one view.
pub(super) struct DrawRegion {
    /// Which of the two pipeline variants records this range. A variant is a
    /// contiguous range precisely because batch numbering puts the plain ones
    /// first — see [`BatchIds`].
    pub masked: bool,
    pub commands: Subbuffer<[DrawIndexedIndirectCommand]>,
}

impl<'a> Draws<'a> {
    /// The CPU path's runs, one at a time. Empty under GPU culling.
    ///
    /// An iterator rather than a collected `Vec` because a pass records this
    /// once per frame and there are up to four of them: collecting would be an
    /// allocation per pass per frame in the middle of the frame loop, which is
    /// the one thing recording is not allowed to do.
    pub(super) fn units(&self) -> Units<'a> {
        match *self {
            Draws::Cpu { list, base } => Units::Cpu {
                runs: list.runs(),
                list,
                base,
            },
            Draws::Gpu { .. } => Units::None,
        }
    }

    /// The compute path's two multi-draws. Empty under CPU culling.
    pub(super) fn regions(&self) -> impl Iterator<Item = DrawRegion> + use<'a> {
        let ranges = match *self {
            Draws::Cpu { .. } => None,
            Draws::Gpu {
                frame,
                view,
                commands,
            } => Some((frame.ranges(view), commands.clone())),
        };
        ranges
            .into_iter()
            .flat_map(|(ranges, commands)| {
                ranges.map(move |(masked, range)| (masked, range, commands.clone()))
            })
            .filter(|(_, range, _)| !range.is_empty())
            .map(|(masked, range, commands)| DrawRegion {
                masked,
                commands: commands.slice(range),
            })
    }
}

/// The draws of one CPU-culled pass, one at a time.
pub(super) enum Units<'a> {
    Cpu {
        runs: crate::gfx::Runs<'a>,
        list: crate::gfx::DrawList<'a>,
        base: u32,
    },
    None,
}

impl<'a> Iterator for Units<'a> {
    type Item = DrawUnit<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Units::Cpu { runs, list, base } => {
                let run = runs.next()?;
                let item = list.item(run.start);
                Some(DrawUnit {
                    mesh: item.mesh.0,
                    material: item.material.0,
                    object_base: *base + run.start as u32,
                    instances: run.len() as u32,
                    run: Some((*list, run)),
                })
            }
            Units::None => None,
        }
    }
}

/// Which batch a (mesh, material) pair is.
///
/// A batch number is *packed*: the top bit says which pipeline variant draws it,
/// the rest is its index within that variant. The number is what persists — a
/// pair asked for twice gets the same one for the life of the renderer, which
/// is what makes the per-row cache below sound — while the batch's *slot*, the
/// row of `draw_commands` it owns, is derived per frame by [`slot`] because it
/// moves whenever a new plain batch appears.
///
/// Splitting by variant at all is what lets a pass draw a whole view with two
/// multi-draws: a variant's commands are contiguous, so a range of them is a
/// range of one pipeline's work. The variant is captured when the batch is
/// first seen and never revisited, which is sound because materials are only
/// ever appended — `load_material` pushes and nothing edits.
#[derive(Default)]
struct BatchIds {
    ids: HashMap<(u32, u32), u32>,
    /// The pairs of each variant in index order, plain first. Indexed by
    /// [`class`].
    infos: [Vec<(u32, u32)>; 2],
}

/// The bit of a packed batch number that says "masked".
const MASKED_BIT: u32 = 1 << 31;

fn class(packed: u32) -> usize {
    usize::from(packed & MASKED_BIT != 0)
}

fn index(packed: u32) -> usize {
    (packed & !MASKED_BIT) as usize
}

/// The row of `draw_commands` a packed batch number owns this frame, given
/// where the masked batches begin. The shaders do this arithmetic themselves;
/// this is the CPU-side statement of it the tests check against.
#[cfg(test)]
fn slot(packed: u32, plain: usize) -> usize {
    index(packed) + class(packed) * plain
}

impl BatchIds {
    /// The number for this pair, assigning one if it is new.
    fn id(&mut self, mesh: u32, material: u32, masked: bool) -> u32 {
        if let Some(&id) = self.ids.get(&(mesh, material)) {
            return id;
        }
        let class = usize::from(masked);
        let id = self.infos[class].len() as u32 | (class as u32 * MASKED_BIT);
        self.infos[class].push((mesh, material));
        self.ids.insert((mesh, material), id);
        id
    }

    /// How many batches the plain pipeline draws, which is where the masked
    /// ones begin.
    fn plain(&self) -> usize {
        self.infos[0].len()
    }

    fn len(&self) -> usize {
        self.infos[0].len() + self.infos[1].len()
    }
}

pub(super) struct CullPass {
    reset_pipeline: PipelineId,
    cull_pipeline: PipelineId,
    /// Device-local and indirect: written by the two dispatches, read by the
    /// draw-indirect stage of every opaque geometry pass.
    commands: Subbuffer<[DrawIndexedIndirectCommand]>,
    /// The compacted per-view lists. This is the frame's `instance_index`, which
    /// under CPU culling is written by the host instead — see
    /// `instances::InstanceStore::upload_lists`.
    indices: Subbuffer<[InstanceEntry]>,
    /// Per-frame scratch for the three tables the dispatches read. An
    /// [`Arena`] rather than the allocator itself because a `SubbufferAllocator`
    /// is `!Sync`, and a pass holding one directly cannot be recorded from a
    /// worker at all.
    scratch: Arena,
    /// Batch numbers, kept across frames so that a scene which spawns nothing
    /// assigns none. Keyed by (mesh, material), which is what a batch is.
    ids: BatchIds,
    /// The last answer this row got, so the map above is consulted only for a
    /// row whose mesh or material actually changed.
    ///
    /// Not an optimisation of last resort: without it this walk is a hash
    /// lookup per renderable per frame over *every* opaque object rather than
    /// the visible ones, which measured at +0.7 ms on a 40k scene — more than
    /// the sweep it replaced gave back.
    cached: Vec<Option<(u32, u32, u32)>>,
    /// Live instances and their batch, rebuilt each frame from the sweep.
    instances: Vec<GpuInstance>,
    /// How many instances each batch has this frame, per variant, indexed the
    /// way [`BatchIds::infos`] is.
    counts: [Vec<u32>; 2],
}

impl CullPass {
    pub(super) fn new(ctx: &VkContext, device: &Device) -> Self {
        let scratch = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        Self {
            reset_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::CullResetComp,
            }),
            cull_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::CullComp,
            }),
            commands: allocate_commands(ctx, 1),
            indices: allocate_indices(ctx, 1),
            scratch,
            ids: BatchIds::default(),
            cached: Vec::new(),
            instances: Vec::new(),
            counts: [Vec::new(), Vec::new()],
        }
    }

    /// What the geometry passes bind as `instance_index`.
    pub(super) fn indices(&self) -> &Subbuffer<[InstanceEntry]> {
        &self.indices
    }

    pub(super) fn commands(&self) -> &Subbuffer<[DrawIndexedIndirectCommand]> {
        &self.commands
    }

    /// Sort this frame's renderables into batches and size the two persistent
    /// buffers for them.
    ///
    /// One pass over `items` rather than the sort the CPU path takes, because
    /// the batch a renderable belongs to is a property of its mesh and material
    /// and not of any ordering — and because this walk is the last per-entity
    /// cost the frame has left.
    pub(super) fn prepare(
        &mut self,
        ctx: &VkContext,
        items: crate::gfx::DrawList<'_>,
        mesh: impl Fn(u32) -> Option<(MeshSpan, Aabb)>,
        masked: impl Fn(u32) -> bool,
        views: u32,
    ) -> CullFrame {
        self.instances.clear();
        self.counts[0].clear();
        self.counts[1].clear();

        for slot in 0..items.len() {
            let item = items.item(slot);
            let key = (item.mesh.0, item.material.0);
            let row = item.instance as usize;
            if row >= self.cached.len() {
                self.cached.resize(row + 1, None);
            }
            let id = match self.cached[row] {
                Some((mesh, material, id)) if (mesh, material) == key => id,
                _ => {
                    let id = self.ids.id(key.0, key.1, masked(key.1));
                    self.cached[row] = Some((key.0, key.1, id));
                    id
                }
            };
            let counts = &mut self.counts[class(id)];
            if index(id) >= counts.len() {
                counts.resize(index(id) + 1, 0);
            }
            counts[index(id)] += 1;
            self.instances.push(GpuInstance {
                row: item.instance,
                batch: id,
            });
        }

        // A batch number outlives the frames in which its batch has members, so
        // a scene that stops drawing a mesh leaves a gap. Gaps are kept rather
        // than compacted: renumbering would move every other batch's slice, and
        // an empty batch costs one command whose instance count stays zero.
        let plain = self.ids.plain();
        let mut batches: Vec<Batch> = Vec::with_capacity(self.ids.len());
        let mut base = 0u32;
        for (infos, counts) in self.ids.infos.iter().zip(&self.counts) {
            for (index, &(mesh_index, material)) in infos.iter().enumerate() {
                // An unuploaded mesh draws nothing, and its bounds stay the
                // inverted box `Aabb::EMPTY` is — which `cull.comp` reads as
                // unmeasurable and therefore never culls. Neither matters while
                // the index count is zero, and stating both keeps that from
                // being the only reason.
                let (span, bounds) = mesh(mesh_index).unwrap_or((MeshSpan::default(), Aabb::EMPTY));
                batches.push(Batch {
                    material,
                    instance_base: base,
                    span,
                    bounds,
                });
                base += counts.get(index).copied().unwrap_or(0);
            }
        }
        let stride = base.max(1);

        let commands = views.max(1) as u64 * batches.len().max(1) as u64;
        if self.commands.len() < commands {
            self.commands = allocate_commands(ctx, commands.next_power_of_two());
        }
        let indices = views.max(1) as u64 * stride as u64;
        if self.indices.len() < indices {
            self.indices = allocate_indices(ctx, indices.next_power_of_two());
        }

        CullFrame {
            batches,
            plain,
            stride,
            views,
        }
    }

    pub(super) fn record_reset(&self, device: &Device, builder: &mut Encoder, frame: &CullFrame) {
        let Device::Vulkan(driver) = device;
        let set = device.create_transient_bind_group(
            self.reset_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(
                        driver.import_buffer(self.upload_batches(frame).into_bytes()),
                    ),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(
                        driver.import_buffer(self.commands.clone().into_bytes()),
                    ),
                },
            ],
        );

        let count = frame.views * frame.batches.len() as u32;
        builder.bind_pipeline(self.reset_pipeline);
        builder.bind_group(0, set);
        // `stride` is how wide one view's block of `instance_index` is, which is
        // what turns a command's index into the view base its `firstInstance`
        // starts from.
        builder.push_constants(&[frame.batches.len() as u32, count, frame.stride]);
        builder.dispatch([count.div_ceil(GROUP).max(1), 1, 1]);
    }

    /// One invocation per (instance, view). The view axis is the dispatch's `y`
    /// so that a workgroup stays within one view — every invocation in it then
    /// takes the same branch through the two view kinds and reads the same
    /// `View`.
    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub(super) fn record_cull(
        &self,
        device: &Device,
        builder: &mut Encoder,
        rows: &Subbuffer<[super::instances::GpuObject]>,
        frame: &CullFrame,
        views: &[CullView],
        occlusion: &OcclusionTest,
        prev_view_proj: Mat4,
    ) {
        let Device::Vulkan(driver) = device;
        let storage = |binding, buffer: Subbuffer<[u8]>| Binding {
            binding,
            resource: BindingResource::StorageBuffer(driver.import_buffer(buffer)),
        };
        let set = device.create_transient_bind_group(
            self.cull_pipeline,
            0,
            &[
                storage(0, rows.clone().into_bytes()),
                storage(1, self.upload_instances().into_bytes()),
                storage(2, self.upload_batches(frame).into_bytes()),
                storage(3, self.upload_views(frame, views).into_bytes()),
                storage(4, self.commands.clone().into_bytes()),
                storage(5, self.indices.clone().into_bytes()),
                Binding {
                    binding: 6,
                    resource: BindingResource::SampledTexture {
                        view: driver.import_view(occlusion.pyramid.clone()),
                        sampler: driver.import_sampler(occlusion.sampler.clone()),
                    },
                },
            ],
        );

        builder.bind_pipeline(self.cull_pipeline);
        builder.bind_group(0, set);
        // `prev_view_proj` is the camera the bound pyramid was rasterised
        // through — the *previous* frame's, because that is the frame whose depth
        // survived to be reduced. `occlusion` is whether the depth test runs at
        // all: a push rather than a second pipeline because the branch is uniform
        // across the dispatch, so what it would buy is a shader compile.
        let mut words = [0u32; 21];
        words[0] = self.instances.len() as u32;
        words[1] = frame.views;
        words[2] = frame.batches.len() as u32;
        // Where the masked batches begin, which is what turns the class and index
        // an instance carries into the batch's slot. See [`slot`].
        words[3] = frame.plain as u32;
        for (word, float) in words[4..20]
            .iter_mut()
            .zip(prev_view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = float.to_bits();
        }
        words[20] = u32::from(occlusion.live);
        builder.push_constants(&words);
        // One invocation per (instance, view). The view axis is the dispatch's
        // `y` so that a workgroup stays within one view — every invocation in it
        // then takes the same branch through the two view kinds and reads the
        // same `View`.
        builder.dispatch([
            (self.instances.len() as u32).div_ceil(GROUP).max(1),
            frame.views.max(1),
            1,
        ]);
    }

    /// The tables are allocated at length one when empty, because
    /// `allocate_slice` rejects a zero length and an empty scene still has to
    /// bind something for the dispatch that will read none of it.
    fn upload_batches(&self, frame: &CullFrame) -> Subbuffer<[GpuBatch]> {
        let buffer = self
            .scratch
            .allocate_slice::<GpuBatch>(frame.batches.len().max(1) as u64);
        {
            let mut write = buffer.write().unwrap();
            for (slot, batch) in frame.batches.iter().enumerate() {
                write[slot] = GpuBatch {
                    index_count: batch.span.index_count,
                    first_index: batch.span.first_index,
                    vertex_offset: batch.span.vertex_offset,
                    instance_base: batch.instance_base,
                    material: batch.material,
                    _padding: [0; 3],
                    bounds_min: batch.bounds.min.extend(0.0).to_array(),
                    bounds_max: batch.bounds.max.extend(0.0).to_array(),
                };
            }
        }
        buffer
    }

    fn upload_instances(&self) -> Subbuffer<[GpuInstance]> {
        let buffer = self
            .scratch
            .allocate_slice::<GpuInstance>(self.instances.len().max(1) as u64);
        buffer
            .write()
            .unwrap()
            .get_mut(..self.instances.len())
            .expect("the scratch slice is at least as long as the instances")
            .copy_from_slice(&self.instances);
        buffer
    }

    fn upload_views(&self, frame: &CullFrame, views: &[CullView]) -> Subbuffer<[GpuView]> {
        let buffer = self
            .scratch
            .allocate_slice::<GpuView>(views.len().max(1) as u64);
        {
            let mut write = buffer.write().unwrap();
            for (index, view) in views.iter().enumerate() {
                let base = index as u32 * frame.stride;
                write[index] = match view {
                    CullView::Frustum(frustum) => GpuView {
                        light_view: Mat4::IDENTITY.to_cols_array_2d(),
                        planes: frustum.planes().map(|plane| plane.to_array()),
                        kind: KIND_FRUSTUM,
                        half_extent: 0.0,
                        depth_range: 0.0,
                        base,
                    },
                    CullView::Cascade {
                        light_view,
                        half_extent,
                        depth_range,
                    } => GpuView {
                        light_view: light_view.to_cols_array_2d(),
                        planes: [[0.0; 4]; 6],
                        kind: KIND_CASCADE,
                        half_extent: *half_extent,
                        depth_range: *depth_range,
                        base,
                    },
                };
            }
        }
        buffer
    }
}

fn allocate_commands(ctx: &VkContext, len: u64) -> Subbuffer<[DrawIndexedIndirectCommand]> {
    Buffer::new_slice::<DrawIndexedIndirectCommand>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        len,
    )
    .expect("failed to allocate the draw command buffer")
}

fn allocate_indices(ctx: &VkContext, len: u64) -> Subbuffer<[InstanceEntry]> {
    Buffer::new_slice::<InstanceEntry>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        len,
    )
    .expect("failed to allocate the instance index buffer")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pipeline variant has to be a contiguous range of the command buffer,
    /// or a multi-draw would cover batches the bound pipeline cannot draw.
    #[test]
    fn plain_batches_number_before_masked() {
        let mut ids = BatchIds::default();
        let plain = ids.id(0, 0, false);
        let masked = ids.id(0, 1, true);
        let second_plain = ids.id(1, 0, false);

        let split = ids.plain();
        assert_eq!(ids.len(), 3);
        assert_eq!(split, 2);
        assert!(slot(plain, split) < split);
        assert!(slot(second_plain, split) < split);
        assert!(slot(masked, split) >= split);
    }

    /// Batch numbers outlive the frame that created them: the per-row cache is
    /// only sound if asking twice gives the same answer.
    #[test]
    fn a_pair_keeps_its_id() {
        let mut ids = BatchIds::default();
        let first = ids.id(7, 3, false);
        let other = ids.id(7, 4, false);
        let again = ids.id(7, 3, false);

        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(ids.len(), 2);
    }

    /// Every batch is drawn exactly once, by the pipeline its material asked
    /// for. A range that overlapped or fell short would draw a cutout through
    /// the plain pipeline, or draw it twice.
    #[test]
    fn the_two_ranges_partition_a_view() {
        let frame = CullFrame {
            batches: vec![batch(0), batch(1), batch(2)],
            plain: 2,
            stride: 8,
            views: 2,
        };

        let [(plain_masked, plain), (masked_masked, masked)] = frame.ranges(1);
        assert!(!plain_masked && masked_masked);
        // View 1's block starts after view 0's three commands.
        assert_eq!(plain, 3..5);
        assert_eq!(masked, 5..6);
    }

    /// A frame with nothing masked records one multi-draw, not two: an empty
    /// range is not a legal `vkCmdDrawIndexedIndirect`.
    #[test]
    fn an_empty_variant_records_nothing() {
        let frame = CullFrame {
            batches: vec![batch(0)],
            plain: 1,
            stride: 4,
            views: 1,
        };

        let ranges = frame.ranges(0);
        let drawn: Vec<_> = ranges.iter().filter(|(_, r)| !r.is_empty()).collect();
        assert_eq!(drawn.len(), 1);
        assert_eq!(drawn[0].1, 0..1);
    }

    fn batch(material: u32) -> Batch {
        Batch {
            material,
            instance_base: 0,
            span: MeshSpan::default(),
            bounds: Aabb::EMPTY,
        }
    }

    /// A masked batch's slot moves when a plain batch appears after it, which is
    /// why nothing persistent may hold a slot.
    #[test]
    fn a_masked_slot_follows_the_split() {
        let mut ids = BatchIds::default();
        let masked = ids.id(0, 1, true);
        assert_eq!(slot(masked, ids.plain()), 0);

        ids.id(0, 0, false);
        assert_eq!(slot(masked, ids.plain()), 1);
    }
}
