//! Deciding what each view draws, on the GPU.
//!
//! The CPU sweep in `systems::extract_geometry` used to answer the same
//! question by testing every renderable against the camera frustum, then
//! against every cascade, then against every shadowing light — a per-entity
//! cost that is invisible at a hundred objects and is most of the frame at ten
//! thousand. That path is gone for opaque geometry: the tests are one compute
//! dispatch over (instance, view), and what the CPU keeps is only the part that
//! is genuinely per entity: which rows the
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
use std::ops::Range;

use crate::geom::{Aabb, Frustum};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use crate::gfx::graph::BufferUsage;
use crate::gfx::rhi::{
    Binding, BindingResource, BlockKind, BufferDesc, BufferId, ComputePipelineDesc, Device,
    Encoder, MemoryLocation, PipelineId,
};
use crate::gfx::shaders::ShaderId;

use super::instances::InstanceEntry;
use super::occlusion::OcclusionTest;
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
#[derive(Pod, Zeroable, Clone, Copy)]
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
#[derive(Pod, Zeroable, Clone, Copy)]
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
#[derive(Pod, Zeroable, Clone, Copy)]
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
    /// Every batch, in [`Variant::class`] order. That order is what makes a
    /// pipeline class a contiguous range of `draw_commands` and therefore one
    /// multi-draw — see [`Draws::regions`].
    pub batches: Vec<Batch>,
    /// Where each class's batches begin, indexed by [`Variant::class`], for the
    /// first `classes` classes and one past them: `bases[classes]` is the batch
    /// count. Inline rather than a `Vec` so a frame allocates nothing for it.
    pub bases: [u32; CLASSES + 1],
    /// How many classes have ever held a batch: one past the highest.
    pub classes: usize,
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

    /// The commands of classes `first..end` for one view, clipped to the
    /// classes that exist.
    ///
    /// A range covers every batch of its classes, including the ones this view
    /// culled to nothing: their commands were reset to zero instances and cost
    /// the command processor a skipped draw rather than the host a decision.
    /// That is the whole trade — the count is known before the dispatch runs,
    /// so recording does not have to wait for it.
    fn range(&self, view: u32, first: usize, end: usize) -> Range<u32> {
        let base = view * self.batches.len() as u32;
        let at = |class: usize| self.bases[class.min(self.classes)];
        base + at(first)..base + at(end.max(first))
    }

    /// The class bases as `cull.comp` reads them: one word per class that
    /// exists, and never none, because a zero-length buffer cannot be bound.
    fn class_bases(&self) -> &[u32] {
        &self.bases[..self.classes.max(1)]
    }
}

/// Which shader reads a batch's surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Surface {
    /// `shading.glsl` with the optional lobes compiled out.
    Flat,
    /// The build that can take every lobe.
    Lobed,
    /// A loaded surface shader, by its slot (`gfx::vulkan::surface`), which is
    /// below [`MAX_SURFACE_SHADERS`].
    Custom(u32),
}

impl Surface {
    fn index(self) -> usize {
        match self {
            Surface::Flat => 0,
            Surface::Lobed => 1,
            Surface::Custom(slot) => BUILT_IN_SURFACES + slot as usize,
        }
    }

    fn of_index(index: usize) -> Self {
        match index {
            0 => Surface::Flat,
            1 => Surface::Lobed,
            custom => Surface::Custom((custom - BUILT_IN_SURFACES) as u32),
        }
    }
}

/// Which pipeline draws a batch: whether its material cuts holes, and which
/// shader reads its surface. The two built-in surfaces are a split because the
/// shader that can take the optional lobes (`gfx::vulkan::forward::
/// material_flags::LOBES`) costs a quarter more than one that cannot, for every
/// material, by their code being there at all — measured, not reasoned (see
/// `ForwardPipelines`). A surface shader is always compiled against every lobe,
/// so it has no flat twin. The depth-only shadow passes care only about
/// `masked`; the prepass cares about `masked` and whether the surface is a
/// loaded one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Variant {
    pub masked: bool,
    pub surface: Surface,
}

/// How many of a packed batch number's bits say its class. Eight, because the
/// other twenty-four still count sixteen million batches *per class* — a batch
/// is a (mesh, material) pair ever drawn in that class, gaps included, so even
/// a scene of forty thousand distinct pairs uses a four-hundredth of it — and
/// eight is what leaves room for a project's surface shaders. Lock-stepped with
/// `CLASS_SHIFT` in `cull.comp`.
const CLASS_BITS: u32 = 8;
const CLASS_SHIFT: u32 = 32 - CLASS_BITS;
const INDEX_MASK: u32 = (1 << CLASS_SHIFT) - 1;

/// Every class a packed number can name: unmasked surfaces first, then masked.
pub(super) const CLASSES: usize = 1 << CLASS_BITS;
/// Surfaces per mask.
const SURFACES: usize = CLASSES / 2;
const BUILT_IN_SURFACES: usize = 2;
/// How many surface shaders may be loaded at once. Loading one more is an error
/// rather than a silent fallback, because nothing else could tell its author
/// why the material looks standard.
pub const MAX_SURFACE_SHADERS: usize = SURFACES - BUILT_IN_SURFACES;

impl Variant {
    /// What a batch whose material is unknown draws as.
    pub(super) const STANDARD: Variant = Variant {
        masked: false,
        surface: Surface::Flat,
    };

    /// The position of this variant's batches: by mask first, then by surface
    /// with flat before lobed, so a mask's classes are adjacent — the shadow
    /// passes draw each mask as one range — and the two built-in ones adjacent
    /// within it, which is what lets the prepass draw them as one.
    pub(super) fn class(self) -> usize {
        usize::from(self.masked) * SURFACES + self.surface.index()
    }

    fn of_class(class: usize) -> Self {
        Self {
            masked: class >= SURFACES,
            surface: Surface::of_index(class % SURFACES),
        }
    }
}

/// What one opaque geometry pass draws: the batches the dispatch culled for one
/// view, as commands in `commands` starting at that view's block.
#[derive(Clone, Copy)]
pub(super) struct Draws<'a> {
    pub frame: &'a CullFrame,
    pub view: u32,
    pub commands: BufferId,
}

/// One multi-draw: every command of one pipeline class, for one view.
pub(super) struct DrawRegion {
    /// Which pipeline records this range. A class is a contiguous range
    /// precisely because batch numbering orders the classes — see
    /// [`BatchIds`]. Where a method merges classes, its docs say which fields
    /// still mean something.
    pub variant: Variant,
    pub commands: BufferId,
    /// Which of its commands this region draws.
    pub range: Range<u32>,
}

impl<'a> Draws<'a> {
    fn region(&self, variant: Variant, first: usize, end: usize) -> Option<DrawRegion> {
        let range = self.frame.range(self.view, first, end);
        (!range.is_empty()).then_some(DrawRegion {
            variant,
            commands: self.commands,
            range,
        })
    }

    /// The view's multi-draws, one per class that has any batches — for the
    /// forward pass, which has a pipeline for each.
    pub(super) fn regions(self) -> impl Iterator<Item = DrawRegion> + 'a {
        (0..self.frame.classes)
            .filter_map(move |class| self.region(Variant::of_class(class), class, class + 1))
    }

    /// The view's multi-draws with every surface merged, one per mask — for the
    /// shadow passes, which take no surface shader and whose pipelines differ
    /// only in whether they cut holes. `surface` is meaningless.
    pub(super) fn regions_by_mask(self) -> impl Iterator<Item = DrawRegion> + 'a {
        [false, true].into_iter().filter_map(move |masked| {
            let first = usize::from(masked) * SURFACES;
            let variant = Variant {
                masked,
                surface: Surface::Flat,
            };
            self.region(variant, first, first + SURFACES)
        })
    }

    /// The view's multi-draws for the prepass: per mask, the two built-in
    /// surfaces as one range, whose shader takes no lobe, then one range per
    /// surface shader. A built-in region says `Surface::Flat`.
    pub(super) fn regions_by_surface_shader(self) -> impl Iterator<Item = DrawRegion> + 'a {
        [false, true].into_iter().flat_map(move |masked| {
            let first = usize::from(masked) * SURFACES;
            let built_in = self.region(
                Variant {
                    masked,
                    surface: Surface::Flat,
                },
                first,
                first + BUILT_IN_SURFACES,
            );
            let custom = (first + BUILT_IN_SURFACES..(first + SURFACES).min(self.frame.classes))
                .filter_map(move |class| self.region(Variant::of_class(class), class, class + 1));
            built_in.into_iter().chain(custom)
        })
    }
}

/// Which batch a (mesh, material) pair is.
///
/// A batch number is *packed*: the top [`CLASS_BITS`] say which pipeline class
/// draws it ([`Variant::class`]), the rest is its index within that class. The
/// number is keyed by the pair *and* its class, and what persists is that key's
/// number — asked for again it is the same one for the life of the renderer —
/// while the batch's *slot*, the row of `draw_commands` it owns, is derived per
/// frame by [`slot`] because it moves whenever a batch of an earlier class
/// appears.
///
/// Splitting by class at all is what lets a pass draw a whole view with one
/// multi-draw per pipeline: a class's commands are contiguous, so a range of
/// them is a range of one pipeline's work. A material does change class —
/// `update_material` can mask it or give it a lobe, and attaching a surface
/// shader always does — and then its pairs get numbers in the new class, the
/// old ones staying behind as gaps. Who asks with which class is
/// [`Batching`]'s concern.
#[derive(Default)]
struct BatchIds {
    ids: HashMap<(u32, u32, u32), u32>,
    /// The pairs of each class in index order, indexed by [`Variant::class`] and
    /// as long as the highest class that has ever held one.
    infos: Vec<Vec<(u32, u32)>>,
}

fn class(packed: u32) -> usize {
    (packed >> CLASS_SHIFT) as usize
}

fn index(packed: u32) -> usize {
    (packed & INDEX_MASK) as usize
}

/// The row of `draw_commands` a packed batch number owns this frame, given
/// where each class begins. The shaders do this arithmetic themselves; this is
/// the CPU-side statement of it the tests check against.
#[cfg(test)]
fn slot(packed: u32, bases: &[u32]) -> usize {
    bases[class(packed)] as usize + index(packed)
}

impl BatchIds {
    /// The number for this pair in this class, assigning one if it is new.
    fn id(&mut self, mesh: u32, material: u32, variant: Variant) -> u32 {
        let class = variant.class();
        if let Some(&id) = self.ids.get(&(mesh, material, class as u32)) {
            return id;
        }
        if self.infos.len() <= class {
            self.infos.resize_with(class + 1, Vec::new);
        }
        let id = self.infos[class].len() as u32 | ((class as u32) << CLASS_SHIFT);
        self.infos[class].push((mesh, material));
        self.ids.insert((mesh, material, class as u32), id);
        id
    }

    /// Where each class's batches begin this frame, then the total.
    fn bases(&self) -> [u32; CLASSES + 1] {
        let mut bases = [0; CLASSES + 1];
        let mut start = 0;
        for (class, infos) in self.infos.iter().enumerate() {
            bases[class] = start;
            start += infos.len() as u32;
        }
        bases[self.infos.len()] = start;
        bases
    }

    fn len(&self) -> usize {
        self.infos.iter().map(Vec::len).sum()
    }
}

/// The batch each renderable's row belongs to, remembered per row.
///
/// The cache is what makes the per-entity walk a comparison rather than a hash
/// lookup: without it the walk looks up every opaque object every frame, which
/// measured at +0.7 ms on a 40k scene — more than the sweep it replaced gave
/// back. Its answer is a function of the row's (mesh, material) and of that
/// material's class, and a row compares only the first two. So the invariant
/// is: **every change to any material's class is followed by
/// [`forget_classes`](Self::forget_classes)** before the next `prepare`. The
/// renderer keeps it by comparing each material's class before and after
/// anything that can move it, so the cost of a change is one frame of lookups
/// and the steady state pays nothing.
#[derive(Default)]
struct Batching {
    ids: BatchIds,
    cached: Vec<Option<(u32, u32, u32)>>,
}

impl Batching {
    fn batch_of(
        &mut self,
        row: usize,
        mesh: u32,
        material: u32,
        variant: impl Fn(u32) -> Variant,
    ) -> u32 {
        if row >= self.cached.len() {
            self.cached.resize(row + 1, None);
        }
        match self.cached[row] {
            Some((cached_mesh, cached_material, id))
                if (cached_mesh, cached_material) == (mesh, material) =>
            {
                id
            }
            _ => {
                let id = self.ids.id(mesh, material, variant(material));
                self.cached[row] = Some((mesh, material, id));
                id
            }
        }
    }

    fn forget_classes(&mut self) {
        self.cached.clear();
    }
}

pub(super) struct CullPass {
    reset_pipeline: PipelineId,
    cull_pipeline: PipelineId,
    /// Device-local and indirect: written by the two dispatches, read by the
    /// draw-indirect stage of every opaque geometry pass.
    commands: BufferId,
    commands_len: u64,
    /// The compacted per-view lists. This is the frame's `instance_index`, which
    /// under CPU culling is written by the host instead — see
    /// `instances::InstanceStore::upload_lists`.
    indices: BufferId,
    indices_len: u64,
    /// Batch numbers and the per-row answers, kept across frames so that a
    /// scene which spawns nothing assigns none and looks nothing up.
    batching: Batching,
    /// Live instances and their batch, rebuilt each frame from the sweep.
    instances: Vec<GpuInstance>,
    /// How many instances each batch has this frame, per class, indexed the
    /// way [`BatchIds::infos`] is.
    counts: Vec<Vec<u32>>,
}

impl CullPass {
    pub(super) fn new(device: &Device) -> Self {
        Self {
            reset_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::CullResetComp.into(),
            }),
            cull_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::CullComp.into(),
            }),
            commands: allocate_commands(device, 1),
            commands_len: 1,
            indices: allocate_indices(device, 1),
            indices_len: 1,
            batching: Batching::default(),
            instances: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// What the geometry passes bind as `instance_index`.
    pub(super) fn indices(&self) -> BufferId {
        self.indices
    }

    pub(super) fn commands(&self) -> BufferId {
        self.commands
    }

    /// Sort this frame's renderables into batches and size the two persistent
    /// buffers for them.
    ///
    /// One pass over `items` rather than a sort, because
    /// the batch a renderable belongs to is a property of its mesh and material
    /// and not of any ordering — and because this walk is the last per-entity
    /// cost the frame has left.
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        items: crate::gfx::DrawList<'_>,
        mesh: impl Fn(u32) -> Option<(MeshSpan, Aabb)>,
        variant: impl Fn(u32) -> Variant,
        views: u32,
    ) -> CullFrame {
        self.instances.clear();
        for counts in &mut self.counts {
            counts.clear();
        }

        for slot in 0..items.len() {
            let item = items.item(slot);
            let id = self.batching.batch_of(
                item.instance as usize,
                item.mesh.0,
                item.material.0,
                &variant,
            );
            if self.counts.len() <= class(id) {
                self.counts.resize_with(class(id) + 1, Vec::new);
            }
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
        let ids = &self.batching.ids;
        let bases = ids.bases();
        let mut batches: Vec<Batch> = Vec::with_capacity(ids.len());
        let mut base = 0u32;
        for (class, infos) in ids.infos.iter().enumerate() {
            let counts = self.counts.get(class).map_or(&[][..], Vec::as_slice);
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
        if self.commands_len < commands {
            device.destroy_buffer(self.commands);
            self.commands_len = commands.next_power_of_two();
            self.commands = allocate_commands(device, self.commands_len);
        }
        let indices = views.max(1) as u64 * stride as u64;
        if self.indices_len < indices {
            device.destroy_buffer(self.indices);
            self.indices_len = indices.next_power_of_two();
            self.indices = allocate_indices(device, self.indices_len);
        }

        CullFrame {
            batches,
            bases,
            classes: ids.infos.len(),
            stride,
            views,
        }
    }

    /// Drop every row's remembered batch, because some material's class
    /// changed. See [`Batching`] for why this is the caller's to call.
    pub(super) fn forget_classes(&mut self) {
        self.batching.forget_classes();
    }

    pub(super) fn record_reset(&self, device: &Device, builder: &mut Encoder, frame: &CullFrame) {
        let set = device.create_transient_bind_group(
            self.reset_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(self.upload_batches(device, frame)),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(self.commands),
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
        rows: BufferId,
        frame: &CullFrame,
        views: &[CullView],
        occlusion: &OcclusionTest,
        prev_view_proj: Mat4,
    ) {
        let storage = |binding, buffer| Binding {
            binding,
            resource: BindingResource::StorageBuffer(buffer),
        };
        let set = device.create_transient_bind_group(
            self.cull_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(rows),
                },
                storage(1, self.upload_instances(device)),
                storage(2, self.upload_batches(device, frame)),
                storage(3, self.upload_views(device, frame, views)),
                storage(4, self.commands),
                storage(5, self.indices),
                Binding {
                    binding: 6,
                    resource: BindingResource::SampledTexture {
                        view: occlusion.pyramid,
                        sampler: occlusion.sampler,
                    },
                },
                // Where each class's batches begin, which is what turns the
                // class and index an instance carries into the batch's slot. See
                // [`slot`]. A buffer rather than the push constants they once
                // were, because their number grows with the surface shaders.
                storage(
                    7,
                    device.upload_block(
                        BlockKind::Storage,
                        bytemuck::cast_slice(frame.class_bases()),
                    ),
                ),
            ],
        );

        builder.bind_pipeline(self.cull_pipeline);
        builder.bind_group(0, set);
        // `prev_view_proj` is the camera the bound pyramid was rasterised
        // through — the *previous* frame's, because that is the frame whose depth
        // survived to be reduced. `occlusion` is whether the depth test runs at
        // all: a push rather than a second pipeline because the branch is uniform
        // across the dispatch, so what it would buy is a shader compile.
        let mut words = [0u32; 20];
        words[0] = self.instances.len() as u32;
        words[1] = frame.views;
        words[2] = frame.batches.len() as u32;
        words[3] = u32::from(occlusion.live);
        for (word, float) in words[4..20]
            .iter_mut()
            .zip(prev_view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = float.to_bits();
        }
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

    fn upload_batches(&self, device: &Device, frame: &CullFrame) -> BufferId {
        let mut table: Vec<GpuBatch> = frame
            .batches
            .iter()
            .map(|batch| GpuBatch {
                index_count: batch.span.index_count,
                first_index: batch.span.first_index,
                vertex_offset: batch.span.vertex_offset,
                instance_base: batch.instance_base,
                material: batch.material,
                _padding: [0; 3],
                bounds_min: batch.bounds.min.extend(0.0).to_array(),
                bounds_max: batch.bounds.max.extend(0.0).to_array(),
            })
            .collect();
        upload_table(device, &mut table)
    }

    fn upload_instances(&self, device: &Device) -> BufferId {
        if self.instances.is_empty() {
            return upload_table::<GpuInstance>(device, &mut Vec::new());
        }
        device.upload_block(BlockKind::Storage, bytemuck::cast_slice(&self.instances))
    }

    fn upload_views(&self, device: &Device, frame: &CullFrame, views: &[CullView]) -> BufferId {
        let mut table: Vec<GpuView> = views
            .iter()
            .enumerate()
            .map(|(index, view)| {
                let base = index as u32 * frame.stride;
                match view {
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
                }
            })
            .collect();
        upload_table(device, &mut table)
    }
}

/// An empty table is padded to one zeroed row, because a zero-length buffer is
/// not one a descriptor can bind and an empty scene still dispatches.
fn upload_table<T: Pod>(device: &Device, table: &mut Vec<T>) -> BufferId {
    if table.is_empty() {
        table.push(T::zeroed());
    }
    device.upload_block(BlockKind::Storage, bytemuck::cast_slice(table))
}

fn allocate_commands(device: &Device, len: u64) -> BufferId {
    device.create_buffer(BufferDesc {
        size: len * std::mem::size_of::<[u32; 5]>() as u64,
        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
        location: MemoryLocation::GpuOnly,
    })
}

fn allocate_indices(device: &Device, len: u64) -> BufferId {
    device.create_buffer(BufferDesc {
        size: len * std::mem::size_of::<InstanceEntry>() as u64,
        usage: BufferUsage::STORAGE_BUFFER,
        location: MemoryLocation::GpuOnly,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn variant(masked: bool, surface: Surface) -> Variant {
        Variant { masked, surface }
    }
    const PLAIN: Variant = variant(false, Surface::Flat);
    const LOBED: Variant = variant(false, Surface::Lobed);
    const MASKED: Variant = variant(true, Surface::Flat);
    const MASKED_LOBED: Variant = variant(true, Surface::Lobed);
    const ROCK: Variant = variant(false, Surface::Custom(0));
    const MOSS: Variant = variant(false, Surface::Custom(3));
    const MASKED_ROCK: Variant = variant(true, Surface::Custom(0));

    /// A CullFrame over `ids` as `prepare` would build it, minus the geometry.
    fn frame(ids: &BatchIds, views: u32) -> CullFrame {
        CullFrame {
            batches: (0..ids.len() as u32).map(batch).collect(),
            bases: ids.bases(),
            classes: ids.infos.len(),
            stride: 8,
            views,
        }
    }

    fn batch(material: u32) -> Batch {
        Batch {
            material,
            instance_base: 0,
            span: MeshSpan::default(),
            bounds: Aabb::EMPTY,
        }
    }

    fn draws(frame: &CullFrame, view: u32) -> Draws<'_> {
        Draws {
            frame,
            view,
            commands: BufferId(0),
        }
    }

    /// A pipeline class has to be a contiguous range of the command buffer, or
    /// a multi-draw would cover batches the bound pipeline cannot draw — and the
    /// classes run mask first, then flat, lobed, and each surface shader, so a
    /// depth-only pass still finds what it merges side by side.
    #[test]
    fn batches_number_by_class_in_draw_order() {
        let mut ids = BatchIds::default();
        let masked_rock = ids.id(0, 6, MASKED_ROCK);
        let moss = ids.id(0, 5, MOSS);
        let masked_lobed = ids.id(0, 4, MASKED_LOBED);
        let lobed = ids.id(0, 2, LOBED);
        let plain = ids.id(0, 0, PLAIN);
        let rock = ids.id(1, 7, ROCK);
        let masked = ids.id(0, 1, MASKED);
        let second_plain = ids.id(1, 0, PLAIN);

        let bases = ids.bases();
        assert_eq!(ids.len(), 8);
        let order = [
            plain,
            second_plain,
            lobed,
            rock,
            moss,
            masked,
            masked_lobed,
            masked_rock,
        ];
        let slots = order.map(|id| slot(id, &bases));
        assert!(slots[0] < 2 && slots[1] < 2);
        assert_eq!(&slots[2..], &[2, 3, 4, 5, 6, 7]);
    }

    /// Batch numbers outlive the frame that created them: the per-row cache is
    /// only sound if asking twice gives the same answer.
    #[test]
    fn a_pair_keeps_its_id() {
        let mut ids = BatchIds::default();
        let first = ids.id(7, 3, PLAIN);
        let other = ids.id(7, 4, PLAIN);
        let again = ids.id(7, 3, PLAIN);

        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(ids.len(), 2);
    }

    /// `update_material` can make a material masked after its batches were
    /// numbered. The pair must then be drawn by the pipeline of the class it is
    /// in now, or a cutout is drawn solid.
    #[test]
    fn a_pair_whose_class_changed_draws_in_its_new_class() {
        let mut ids = BatchIds::default();
        let before = ids.id(0, 3, PLAIN);
        let after = ids.id(0, 3, MASKED);
        assert_eq!(class(before), PLAIN.class());
        assert_eq!(class(after), MASKED.class());
    }

    /// The same through the per-row cache, which is where a frame actually
    /// asks: a row whose material moved class keeps its stale batch until the
    /// classes are forgotten, and is in the new class from the frame after.
    #[test]
    fn a_row_follows_its_material_into_a_new_class_once_classes_are_forgotten() {
        let mut batching = Batching::default();
        let first = batching.batch_of(4, 0, 3, |_| PLAIN);
        assert_eq!(class(first), PLAIN.class());
        assert_eq!(
            batching.batch_of(4, 0, 3, |_| ROCK),
            first,
            "the cache is trusted until it is told otherwise"
        );

        batching.forget_classes();
        let attached = batching.batch_of(4, 0, 3, |_| ROCK);
        assert_eq!(class(attached), ROCK.class());

        batching.forget_classes();
        assert_eq!(batching.batch_of(4, 0, 3, |_| PLAIN), first);
    }

    /// Every batch is drawn exactly once, by the pipeline its material asked
    /// for. A range that overlapped or fell short would draw a cutout through
    /// the plain pipeline, or a surface shader's material through the standard
    /// one.
    #[test]
    fn the_classes_partition_a_view() {
        let mut ids = BatchIds::default();
        for (material, variant) in [
            (0, PLAIN),
            (1, PLAIN),
            (2, LOBED),
            (3, ROCK),
            (4, MOSS),
            (5, MOSS),
            (6, MASKED),
            (7, MASKED_ROCK),
        ] {
            ids.id(0, material, variant);
        }
        let frame = frame(&ids, 2);

        let regions: Vec<_> = draws(&frame, 1)
            .regions()
            .map(|r| (r.variant, r.range))
            .collect();
        // View 1's block starts after view 0's eight commands.
        assert_eq!(
            regions,
            vec![
                (PLAIN, 8..10),
                (LOBED, 10..11),
                (ROCK, 11..12),
                (MOSS, 12..14),
                (MASKED, 14..15),
                (MASKED_ROCK, 15..16),
            ]
        );
    }

    /// The shadows draw every opaque batch with one pipeline per mask, so every
    /// surface of a mask is one range — one multi-draw each, however many
    /// surface shaders the scene has.
    #[test]
    fn a_shadow_pass_draws_one_range_per_mask() {
        let mut ids = BatchIds::default();
        for (material, variant) in [
            (0, PLAIN),
            (1, MOSS),
            (2, LOBED),
            (3, MASKED_LOBED),
            (4, ROCK),
            (5, MASKED_ROCK),
            (6, MASKED),
        ] {
            ids.id(0, material, variant);
        }
        let frame = frame(&ids, 1);
        let merged: Vec<_> = draws(&frame, 0)
            .regions_by_mask()
            .map(|r| (r.variant.masked, r.range))
            .collect();
        assert_eq!(merged, vec![(false, 0..4), (true, 4..7)]);
    }

    /// The prepass has one shader for both built-in surfaces and one per surface
    /// shader: flat and lobed are one range per mask, each surface shader its
    /// own.
    #[test]
    fn the_prepass_merges_the_built_in_surfaces_and_splits_the_loaded_ones() {
        let mut ids = BatchIds::default();
        for (material, variant) in [
            (0, PLAIN),
            (1, LOBED),
            (2, ROCK),
            (3, MOSS),
            (4, MASKED),
            (5, MASKED_LOBED),
            (6, MASKED_ROCK),
        ] {
            ids.id(0, material, variant);
        }
        let frame = frame(&ids, 1);
        let regions: Vec<_> = draws(&frame, 0)
            .regions_by_surface_shader()
            .map(|r| (r.variant, r.range))
            .collect();
        assert_eq!(
            regions,
            vec![
                (PLAIN, 0..2),
                (ROCK, 2..3),
                (MOSS, 3..4),
                (MASKED, 4..6),
                (MASKED_ROCK, 6..7),
            ]
        );
    }

    /// A frame with nothing masked and no surface shader records one
    /// multi-draw, not one per class: an empty range is not a legal
    /// `vkCmdDrawIndexedIndirect`.
    #[test]
    fn an_empty_class_records_nothing() {
        let mut ids = BatchIds::default();
        ids.id(0, 0, PLAIN);
        let frame = frame(&ids, 1);
        let drawn: Vec<_> = draws(&frame, 0).regions().map(|r| r.range).collect();
        assert_eq!(drawn, vec![0..1]);
        assert_eq!(draws(&frame, 0).regions_by_mask().count(), 1);
        assert_eq!(draws(&frame, 0).regions_by_surface_shader().count(), 1);
        assert_eq!(frame.class_bases(), &[0]);

        let empty = BatchIds::default();
        let frame = super::tests::frame(&empty, 1);
        assert_eq!(draws(&frame, 0).regions().count(), 0);
        assert_eq!(draws(&frame, 0).regions_by_mask().count(), 0);
        assert_eq!(draws(&frame, 0).regions_by_surface_shader().count(), 0);
        assert_eq!(
            frame.class_bases().len(),
            1,
            "a buffer of no words cannot be bound"
        );
    }

    /// A batch's slot moves when one of an earlier class appears after it,
    /// which is why nothing persistent may hold a slot.
    #[test]
    fn a_slot_follows_the_classes_before_it() {
        let mut ids = BatchIds::default();
        let masked = ids.id(0, 1, MASKED);
        assert_eq!(slot(masked, &ids.bases()), 0);

        ids.id(0, 0, PLAIN);
        assert_eq!(slot(masked, &ids.bases()), 1);
        ids.id(0, 2, MOSS);
        assert_eq!(slot(masked, &ids.bases()), 2);
    }

    /// The bound on surface shaders is what the class bits leave: the last
    /// slot, masked, is the last class, and the class and index of a packed
    /// number at either extreme come back out unchanged.
    #[test]
    fn the_last_surface_shader_fits_the_class_bits() {
        let last = variant(true, Surface::Custom(MAX_SURFACE_SHADERS as u32 - 1));
        assert_eq!(last.class(), CLASSES - 1);
        assert_eq!(Variant::of_class(last.class()), last);
        assert_eq!(MAX_SURFACE_SHADERS, 126);
        for class_number in [0, SURFACES, CLASSES - 1] {
            let variant = Variant::of_class(class_number);
            assert_eq!(variant.class(), class_number);
            let packed = INDEX_MASK | ((class_number as u32) << CLASS_SHIFT);
            assert_eq!(class(packed), class_number);
            assert_eq!(index(packed), INDEX_MASK as usize);
        }
    }
}
