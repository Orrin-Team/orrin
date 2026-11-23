//! The scene as something a ray can be traced against.
//!
//! Two levels, which is the API's shape and not a choice: one bottom-level
//! structure per mesh span, built once from the arena's positions and indices,
//! and one top-level structure over this frame's instances of them. The driver's
//! own BVH rather than one of ours, because it is the one the hardware
//! traverses.
//!
//! # A mesh's structure
//!
//! Queued when the mesh is uploaded and built by a later frame, a bounded
//! number of triangles per frame ([`BUILD_BUDGET`]) so a scene load spreads its
//! builds instead of stalling one frame on all of them. Until it is built the
//! mesh is simply absent from the scene: a ray passes through it for those
//! frames, which is the same thing the rasteriser shows of a mesh still
//! streaming in.
//!
//! Each is then compacted. The size a build has to be given is the driver's
//! worst case, and the size it turns out to need is typically half; the
//! difference is asked for with a query the frame after the build, and once
//! that frame has retired the structure is copied into one of exactly that
//! size and the first is retired. The query waits a frame because a pass may
//! not write a barrier and one is needed between a build and a read of what it
//! built: across frames the queue's order is the barrier.
//!
//! A structure is keyed by where its span starts, and leaves with the span when
//! the mesh is replaced, on the pools' delay.
//!
//! # The scene's structure
//!
//! Rebuilt every frame rather than refitted: a refit keeps the tree's shape,
//! which degrades as instances move apart, and a full build over a few thousand
//! instances is cheap enough that there is nothing to trade it for. Built for
//! build speed, where a mesh's is built for trace speed, for the same reason.
//!
//! Two of them, alternating by frame, because a frame split across two queues
//! has its compute tail still tracing while the next frame's head builds — the
//! reason `GraphImages` keeps two sets of the graph's images.
//!
//! What goes in is the caller's list. An instance whose mesh has no structure
//! yet is left out, and the list's order is the order of the instances, so a
//! hit's instance index means nothing: the 24-bit id an instance carries is
//! what a shader reads.
//!
//! # Not here
//!
//! Deforming meshes, which need a refit per frame of their own structure, and
//! any geometry but triangles.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use vulkano::acceleration_structure::{
    AccelerationStructure, AccelerationStructureBuildGeometryInfo,
    AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
    AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
    AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
    AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
    GeometryFlags, GeometryInstanceFlags,
};
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::format::Format;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::Sharing;
use vulkano::{DeviceSize, Packed24_8};

use crate::gfx::PositionVertex;
use crate::gfx::rhi::{MeshSpan, RayInstance, has_retired};

use super::context::VkContext;
use super::mesh::MeshArena;
use super::record::{Arena, Recorder, create_acceleration_structure};

/// Triangles a frame builds mesh structures for, at most. A mesh larger than
/// this still builds, alone: a budget that could refuse one would never build
/// it.
const BUILD_BUDGET: u64 = 400_000;

/// Instances a scene structure has room for before it first grows.
const INITIAL_INSTANCES: u32 = 1024;

/// An instance array's address is a multiple of this.
const INSTANCE_ALIGNMENT: DeviceSize = 16;

/// Where a span starts in the arena's two address spaces, which no two live
/// spans share.
type SpanKey = (u32, i32);

fn key(span: MeshSpan) -> SpanKey {
    (span.first_index, span.vertex_offset)
}

struct Waiting {
    span: MeshSpan,
    vertices: u32,
}

/// The structures one frame built, until they are compacted.
struct Batch {
    entries: Vec<(SpanKey, Arc<AccelerationStructure>)>,
    pool: Arc<QueryPool>,
    built: u64,
    /// The frame whose commands ask the sizes, once one has.
    queried: Option<u64>,
}

struct Scene {
    structure: Arc<AccelerationStructure>,
    scratch: Subbuffer<[u8]>,
    capacity: u32,
}

/// What `prepare` decided this frame records, taken by the two passes.
#[derive(Default)]
struct Work {
    builds: Vec<(
        AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo,
    )>,
    queries: Vec<(Vec<Arc<AccelerationStructure>>, Arc<QueryPool>)>,
    compactions: Vec<(Arc<AccelerationStructure>, Arc<AccelerationStructure>)>,
    scene: Option<(
        AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo,
    )>,
}

pub(super) struct SceneAccel {
    meshes: HashMap<SpanKey, Arc<AccelerationStructure>>,
    waiting: VecDeque<Waiting>,
    batches: VecDeque<Batch>,
    work: Work,
    scenes: [Option<Scene>; 2],
    /// Which of `scenes` the frame being recorded builds and traces.
    current: usize,
    instances: Arena,
    retired: Vec<(u64, Arc<AccelerationStructure>)>,
    scratch_alignment: DeviceSize,
    frames_in_flight: u64,
    mesh_bytes: u64,
}

impl SceneAccel {
    pub(super) fn new(ctx: &VkContext, frames_in_flight: u64) -> Self {
        let scratch_alignment = ctx
            .device
            .physical_device()
            .properties()
            .min_acceleration_structure_scratch_offset_alignment
            .map_or(256, DeviceSize::from);
        Self {
            meshes: HashMap::new(),
            waiting: VecDeque::new(),
            batches: VecDeque::new(),
            work: Work::default(),
            scenes: [None, None],
            current: 0,
            instances: Arena::new(
                ctx.memory_allocator.clone(),
                SubbufferAllocatorCreateInfo {
                    buffer_usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            ),
            retired: Vec::new(),
            scratch_alignment,
            frames_in_flight,
            mesh_bytes: 0,
        }
    }

    /// Queue a structure for a mesh just uploaded. A mesh with no triangles
    /// has none.
    pub(super) fn add(&mut self, span: MeshSpan, vertices: u32) {
        if span.index_count >= 3 && vertices > 0 {
            self.waiting.push_back(Waiting { span, vertices });
        }
    }

    /// Let go of a replaced mesh's structure as of `frame`.
    pub(super) fn retire(&mut self, span: MeshSpan, frame: u64) {
        let key = key(span);
        self.waiting
            .retain(|waiting| self::key(waiting.span) != key);
        if let Some(structure) = self.meshes.remove(&key) {
            self.mesh_bytes -= structure.size();
            self.retired.push((frame, structure));
        }
    }

    /// The structure this frame traces, once `prepare` has run for it.
    pub(super) fn scene(&self) -> Arc<AccelerationStructure> {
        self.scenes[self.current]
            .as_ref()
            .expect("a pass traced the scene in a frame that did not prepare it")
            .structure
            .clone()
    }

    /// Bytes of video memory the structures hold: every mesh's, and both of
    /// the scene's.
    pub(super) fn bytes(&self) -> u64 {
        let scenes: u64 = self
            .scenes
            .iter()
            .flatten()
            .map(|scene| scene.structure.size() + scene.scratch.size())
            .sum();
        self.mesh_bytes + scenes
    }

    /// Decide what frame `frame` builds: the next mesh structures under the
    /// budget, the compactions whose sizes have arrived, and the scene's own
    /// over `instances`. Returns how many of them the scene took.
    pub(super) fn prepare(
        &mut self,
        ctx: &VkContext,
        arena: &MeshArena,
        frame: u64,
        instances: &[RayInstance],
    ) -> u32 {
        let depth = self.frames_in_flight;
        self.retired
            .retain(|&(retired, _)| !has_retired(retired, depth, frame));
        self.work = Work::default();
        self.current = (frame % 2) as usize;

        self.compact(ctx, frame);
        self.build_meshes(ctx, arena, frame);
        self.build_scene(ctx, instances)
    }

    /// Record the mesh structures' share of this frame's work.
    pub(super) fn record_meshes(&mut self, recorder: &mut Recorder) {
        for (structures, pool) in self.work.queries.drain(..) {
            recorder.write_compacted_sizes(&structures, pool);
        }
        for (src, dst) in self.work.compactions.drain(..) {
            recorder.compact_acceleration_structure(src, dst);
        }
        for (info, range) in self.work.builds.drain(..) {
            recorder.build_acceleration_structure(info, range);
        }
    }

    /// Record the scene structure's build.
    pub(super) fn record_scene(&mut self, recorder: &mut Recorder) {
        if let Some((info, range)) = self.work.scene.take() {
            recorder.build_acceleration_structure(info, range);
        }
    }

    fn compact(&mut self, ctx: &VkContext, frame: u64) {
        let depth = self.frames_in_flight;
        while let Some(batch) = self.batches.front() {
            let Some(queried) = batch.queried else {
                break;
            };
            if !has_retired(queried, depth, frame) {
                break;
            }
            let batch = self.batches.pop_front().expect("peeked above");
            let mut sizes = vec![0u64; batch.entries.len()];
            let ready = batch
                .pool
                .get_results(
                    0..batch.entries.len() as u32,
                    &mut sizes,
                    QueryResultFlags::empty(),
                )
                .expect("failed to read the compacted sizes");
            // The frame that asked has retired, so the answer is there. One
            // that is not leaves its structures as built, which is only larger.
            if !ready {
                continue;
            }
            for ((key, built), size) in batch.entries.into_iter().zip(sizes) {
                let current = self
                    .meshes
                    .get(&key)
                    .is_some_and(|structure| Arc::ptr_eq(structure, &built));
                if !current || size == 0 || size >= built.size() {
                    continue;
                }
                let compact = create_acceleration_structure(
                    ctx,
                    AccelerationStructureType::BottomLevel,
                    storage(ctx, size),
                );
                self.mesh_bytes = self.mesh_bytes - built.size() + compact.size();
                self.meshes.insert(key, compact.clone());
                self.retired.push((frame, built.clone()));
                self.work.compactions.push((built, compact));
            }
        }

        // Asked a frame after the build, never in it. See the module docs.
        for batch in &mut self.batches {
            if batch.queried.is_none() && batch.built < frame {
                batch.queried = Some(frame);
                self.work.queries.push((
                    batch
                        .entries
                        .iter()
                        .map(|(_, structure)| structure.clone())
                        .collect(),
                    batch.pool.clone(),
                ));
            }
        }
    }

    fn build_meshes(&mut self, ctx: &VkContext, arena: &MeshArena, frame: u64) {
        let mut chosen = Vec::new();
        let mut triangles = 0u64;
        while let Some(next) = self.waiting.front() {
            let count = u64::from(next.span.index_count / 3);
            if !chosen.is_empty() && triangles + count > BUILD_BUDGET {
                break;
            }
            triangles += count;
            chosen.push(self.waiting.pop_front().expect("peeked above"));
        }
        if chosen.is_empty() {
            return;
        }

        // Sized first, so one scratch allocation serves the batch: each build
        // takes a slice of its own, and builds that share no scratch need no
        // barrier between them.
        let sized: Vec<_> = chosen
            .iter()
            .map(|waiting| {
                let (positions, indices) = arena.build_inputs(waiting.span, waiting.vertices);
                let mut triangles =
                    AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT);
                // Opaque unless an instance says otherwise: whether a triangle
                // blocks a ray is the material's to say, and one mesh is drawn
                // with several.
                triangles.flags = GeometryFlags::OPAQUE;
                triangles.vertex_data = Some(positions.into_bytes());
                triangles.vertex_stride = std::mem::size_of::<PositionVertex>() as u32;
                triangles.max_vertex = waiting.vertices - 1;
                triangles.index_data = Some(IndexBuffer::U32(indices));
                let mut info = AccelerationStructureBuildGeometryInfo::new(
                    AccelerationStructureGeometries::Triangles(vec![triangles]),
                );
                info.flags = BuildAccelerationStructureFlags::PREFER_FAST_TRACE
                    | BuildAccelerationStructureFlags::ALLOW_COMPACTION;
                let count = waiting.span.index_count / 3;
                let sizes = ctx
                    .device
                    .acceleration_structure_build_sizes(
                        AccelerationStructureBuildType::Device,
                        &info,
                        &[count],
                    )
                    .expect("failed to size a mesh's acceleration structure");
                (info, count, sizes)
            })
            .collect();

        let align = self.scratch_alignment;
        let scratch_bytes: DeviceSize = sized
            .iter()
            .map(|(_, _, sizes)| sizes.build_scratch_size.next_multiple_of(align))
            .sum();
        let scratch = scratch(ctx, scratch_bytes, align);

        let mut entries = Vec::with_capacity(chosen.len());
        let mut offset = 0;
        for (waiting, (mut info, count, sizes)) in chosen.iter().zip(sized) {
            let structure = create_acceleration_structure(
                ctx,
                AccelerationStructureType::BottomLevel,
                storage(ctx, sizes.acceleration_structure_size),
            );
            let end = offset + sizes.build_scratch_size.next_multiple_of(align);
            info.dst_acceleration_structure = Some(structure.clone());
            info.scratch_data = Some(scratch.clone().slice(offset..end));
            offset = end;

            self.mesh_bytes += structure.size();
            self.meshes.insert(key(waiting.span), structure.clone());
            entries.push((key(waiting.span), structure));
            self.work.builds.push((
                info,
                AccelerationStructureBuildRangeInfo {
                    primitive_count: count,
                    ..Default::default()
                },
            ));
        }

        let pool = QueryPool::new(
            ctx.device.clone(),
            QueryPoolCreateInfo {
                query_count: entries.len() as u32,
                ..QueryPoolCreateInfo::query_type(QueryType::AccelerationStructureCompactedSize)
            },
        )
        .expect("failed to create a compaction query pool");
        self.batches.push_back(Batch {
            entries,
            pool,
            built: frame,
            queried: None,
        });
    }

    fn build_scene(&mut self, ctx: &VkContext, instances: &[RayInstance]) -> u32 {
        let resolved: Vec<AccelerationStructureInstance> = instances
            .iter()
            .filter_map(|instance| {
                let structure = self.meshes.get(&key(instance.span))?;
                let flags = if instance.opaque {
                    GeometryInstanceFlags::FORCE_OPAQUE
                } else {
                    GeometryInstanceFlags::FORCE_NO_OPAQUE
                };
                Some(AccelerationStructureInstance {
                    transform: instance.transform,
                    instance_custom_index_and_mask: Packed24_8::new(instance.id, 0xff),
                    instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                        0,
                        flags.into(),
                    ),
                    acceleration_structure_reference: structure.device_address().get(),
                })
            })
            .collect();
        let count = resolved.len() as u32;

        // Never empty: a buffer of no bytes has no address to build from, and
        // a scene of nothing is still a structure a ray can miss.
        let bytes = std::mem::size_of_val(resolved.as_slice()).max(INSTANCE_ALIGNMENT as usize);
        let data = by_address(
            self.instances
                .allocate_slice::<u8>(bytes as DeviceSize + INSTANCE_ALIGNMENT),
            bytes as DeviceSize,
            INSTANCE_ALIGNMENT,
        );
        if !resolved.is_empty() {
            data.write()
                .expect("the instance buffer is host-visible")
                .copy_from_slice(bytemuck::cast_slice(&resolved));
        }

        let geometry = |data| {
            let mut info = AccelerationStructureBuildGeometryInfo::new(
                AccelerationStructureGeometries::Instances(
                    AccelerationStructureGeometryInstancesData::new(
                        AccelerationStructureGeometryInstancesDataType::Values(data),
                    ),
                ),
            );
            info.flags = BuildAccelerationStructureFlags::PREFER_FAST_BUILD;
            info
        };

        let slot = &mut self.scenes[self.current];
        if slot.as_ref().is_none_or(|scene| scene.capacity < count) {
            let capacity = count.max(INITIAL_INSTANCES).next_power_of_two();
            let sizes = ctx
                .device
                .acceleration_structure_build_sizes(
                    AccelerationStructureBuildType::Device,
                    &geometry(None),
                    &[capacity],
                )
                .expect("failed to size the scene's acceleration structure");
            // The one outgrown is dropped with the last frame that named it:
            // every command that traced it holds it.
            *slot = Some(Scene {
                structure: create_acceleration_structure(
                    ctx,
                    AccelerationStructureType::TopLevel,
                    storage(ctx, sizes.acceleration_structure_size),
                ),
                scratch: scratch(ctx, sizes.build_scratch_size, self.scratch_alignment),
                capacity,
            });
        }
        let scene = slot.as_ref().expect("allocated above");

        let mut info = geometry(Some(data.reinterpret()));
        info.dst_acceleration_structure = Some(scene.structure.clone());
        info.scratch_data = Some(scene.scratch.clone());
        self.work.scene = Some((
            info,
            AccelerationStructureBuildRangeInfo {
                primitive_count: count,
                ..Default::default()
            },
        ));
        count
    }
}

/// Storage for one acceleration structure, reachable from both queues where
/// there are two: the frame's compute tail traces it.
fn storage(ctx: &VkContext, size: DeviceSize) -> Subbuffer<[u8]> {
    allocate(
        ctx,
        BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
        size,
    )
}

/// A build's working memory, at an address the device accepts.
fn scratch(ctx: &VkContext, size: DeviceSize, alignment: DeviceSize) -> Subbuffer<[u8]> {
    let size = size.max(alignment);
    by_address(
        allocate(
            ctx,
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            size + alignment,
        ),
        size,
        alignment,
    )
}

fn allocate(ctx: &VkContext, usage: BufferUsage, size: DeviceSize) -> Subbuffer<[u8]> {
    let families = ctx.shared_queue_families();
    Buffer::new_slice::<u8>(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            sharing: if families.is_empty() {
                Sharing::Exclusive
            } else {
                Sharing::Concurrent(families.into_iter().collect())
            },
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .expect("failed to allocate acceleration structure memory")
}

/// The first `size` bytes of `buffer` whose device address is a multiple of
/// `alignment`. By address and not by offset, because a buffer's own address
/// need not be aligned to anything a build asks for.
fn by_address(buffer: Subbuffer<[u8]>, size: DeviceSize, alignment: DeviceSize) -> Subbuffer<[u8]> {
    let address = buffer
        .device_address()
        .expect("the buffer was created with a device address")
        .get();
    let skip = address.next_multiple_of(alignment) - address;
    buffer.slice(skip..skip + size)
}
