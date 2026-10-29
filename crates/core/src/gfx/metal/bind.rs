//! Bind groups as argument buffers.
//!
//! A descriptor set is one argument buffer per stage, bound at the buffer index
//! of the set's number. A stage's buffer is written by the argument encoder of
//! *that stage's function*, which is what keeps its layout correct by
//! construction (trap T3): Metal packs a struct's members in the order of their
//! `[[id]]`s, so the offset of a binding depends on which other bindings the
//! function happens to declare, and the vertex and fragment stages of one
//! pipeline rarely declare the same ones. The binding map's `index` is the
//! `[[id]]` (fixed by `build.rs`); what a stage declares is read from its MSL.
//!
//! A resource reached only through an argument buffer is invisible to the GPU
//! unless the encoder says it is in use (trap T1), so each stage's buffer
//! carries the list of what it holds and the encoder declares it when the group
//! is bound.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLDevice, MTLFunction, MTLResource, MTLResourceOptions,
    MTLResourceUsage,
};

use super::pipeline::Pipe;
use super::{MtlDriver, Shared};
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, PipelineId, SamplerId, TextureViewId,
};
use crate::gfx::shaders::{Resource, ResourceKind, ShaderId};

/// A set's argument buffer, and what it holds. One buffer for every stage: the
/// translated shaders declare the same slots (see `build.rs`), so the layout is
/// the set's and not the reading stage's.
pub(super) struct Group {
    pub buffer: Shared<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub offset: u64,
    pub uses: Vec<Use>,
}

/// A group as an encoder binds it.
pub(super) type GroupParts = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    u64,
    Vec<(Retained<ProtocolObject<dyn MTLResource>>, MTLResourceUsage)>,
);

pub(super) struct Use {
    pub resource: Shared<Retained<ProtocolObject<dyn MTLResource>>>,
    pub usage: MTLResourceUsage,
}

/// The `[[id]]`s a shader's translated source declares as real resources in one
/// descriptor set's struct, not counting the stand-ins that fill the ids it does
/// not use. Writing a stand-in as a resource would be a type error in Metal's
/// eyes, and an id the translator proved unused is one of them. Each id says
/// whether the stage may write it: a buffer the translator emitted `const` is
/// only read, and declaring it written makes tracked encoders wait on each
/// other for nothing.
fn declared_ids(msl: &str, set: u32) -> HashMap<u32, bool> {
    let marker = format!("struct spvDescriptorSetBuffer{set}\n");
    let Some(body) = msl.split(&marker).nth(1) else {
        return HashMap::new();
    };
    let body = body.split("};").next().unwrap_or("");
    body.lines()
        .filter(|line| !line.contains("_unused"))
        .filter_map(|line| {
            let at = line.find("[[id(")? + 5;
            let id = line[at..at + line[at..].find(')')?].parse().ok()?;
            Some((id, !line.contains("const device")))
        })
        .collect()
}

fn find_in(shader: ShaderId, set: u32, binding: u32) -> Option<Resource> {
    shader
        .blob()
        .bindings
        .resources
        .iter()
        .find(|resource| resource.set == set && resource.binding == binding)
        .copied()
}

/// One stage's argument encoder over a set's buffer, and the ids it declares.
struct StageWriter {
    shader: ShaderId,
    declared: HashMap<u32, bool>,
    encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
}

/// What a (shader, set) pair costs to work out: creating an argument encoder is
/// about 30 microseconds and the set is rebuilt many times a frame.
pub(super) type EncoderCache =
    HashMap<(ShaderId, u32), Option<(HashMap<u32, bool>, super::Held<dyn MTLArgumentEncoder>)>>;

impl MtlDriver {
    fn shaders_of(&self, pipeline: PipelineId) -> Vec<ShaderId> {
        self.with_pipe(pipeline, |pipe| match pipe {
            Pipe::Render(render) => vec![render.vertex, render.fragment],
            Pipe::Compute(compute) => vec![compute.shader],
        })
    }

    /// Write `bindings` into a fresh argument buffer for each stage of
    /// `pipeline` that reads `set`. A set no stage reads gets an empty group, so
    /// binding it is harmless.
    fn build_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
        persistent: bool,
    ) -> BindGroupId {
        let shaders = self.shaders_of(pipeline);
        // Held for the whole build: an argument encoder is not thread safe and
        // the cached ones are shared.
        let mut cache = self.encoders.lock().unwrap();
        let writers: Vec<StageWriter> = shaders
            .iter()
            .copied()
            .filter_map(|shader| {
                let entry = cache.entry((shader, set)).or_insert_with(|| {
                    let declared = declared_ids(shader.blob().msl, set);
                    if declared.is_empty() {
                        return None;
                    }
                    let function = self.function(shader);
                    // SAFETY: `set` is a buffer index this function declares an
                    // argument buffer at, which `declared_ids` just established.
                    let encoder =
                        unsafe { function.newArgumentEncoderWithBufferIndex(set as usize) };
                    Some((declared, super::Shared(encoder)))
                });
                let (declared, encoder) = entry.as_ref()?;
                Some(StageWriter {
                    shader,
                    declared: declared.clone(),
                    encoder: encoder.0.clone(),
                })
            })
            .collect();
        if writers.is_empty() {
            return self.insert_group(self.empty_group(), persistent);
        }
        // Every stage's struct has the same slots, so any encoder's length is
        // the buffer's.
        let length = writers[0].encoder.encodedLength().max(8) as u64;
        let (buffer, offset) = if persistent {
            let buffer = self
                .device
                .newBufferWithLength_options(length as usize, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate an argument buffer");
            (buffer, 0)
        } else {
            self.frame_arguments(length)
        };
        for writer in &writers {
            // SAFETY: the buffer is at least `encodedLength` bytes from `offset`,
            // at the alignment the arena and a fresh buffer both give.
            unsafe {
                writer
                    .encoder
                    .setArgumentBuffer_offset(Some(&buffer), offset as usize)
            };
        }
        // The stage whose struct declares `id` as a real resource, which is the
        // one whose encoder knows what type to write there.
        let writer_of = |id: u32| {
            writers
                .iter()
                .find(|writer| writer.declared.contains_key(&id))
        };

        let mut uses = Vec::new();
        // Textures at one binding are an array, filled in the order given.
        let mut next_in_array: Vec<(u32, u32)> = Vec::new();
        for binding in bindings {
            let Some(resource) = writers
                .iter()
                .find_map(|writer| find_in(writer.shader, set, binding.binding))
            else {
                continue;
            };
            let slot = |counts: &mut Vec<(u32, u32)>| -> u32 {
                match counts.iter_mut().find(|(b, _)| *b == binding.binding) {
                    Some((_, n)) => {
                        *n += 1;
                        resource.index + *n
                    }
                    None => {
                        counts.push((binding.binding, 0));
                        resource.index
                    }
                }
            };
            match binding.resource {
                BindingResource::SampledTexture { view, sampler } => {
                    let index = slot(&mut next_in_array);
                    if let Some(writer) = writer_of(resource.index) {
                        self.write_texture(
                            &writer.encoder,
                            index,
                            view,
                            MTLResourceUsage::Read,
                            &mut uses,
                        );
                    }
                    let at = resource.index + resource.count;
                    if resource.kind == ResourceKind::CombinedImageSampler
                        && let Some(writer) = writer_of(at)
                    {
                        self.write_sampler(&writer.encoder, at, sampler);
                    }
                }
                BindingResource::Texture(view) => {
                    let index = slot(&mut next_in_array);
                    if let Some(writer) = writer_of(resource.index) {
                        self.write_texture(
                            &writer.encoder,
                            index,
                            view,
                            MTLResourceUsage::Read,
                            &mut uses,
                        );
                    }
                }
                BindingResource::StorageTexture(view) => {
                    let index = slot(&mut next_in_array);
                    if let Some(writer) = writer_of(resource.index) {
                        self.write_texture(
                            &writer.encoder,
                            index,
                            view,
                            MTLResourceUsage::Read | MTLResourceUsage::Write,
                            &mut uses,
                        );
                    }
                }
                BindingResource::Sampler(sampler) => {
                    if let Some(writer) = writer_of(resource.index) {
                        self.write_sampler(&writer.encoder, resource.index, sampler);
                    }
                }
                BindingResource::StorageBuffer(id) | BindingResource::UniformBuffer(id) => {
                    let Some(writer) = writer_of(resource.index) else {
                        continue;
                    };
                    let buf = self.buffer(id);
                    let written = writers
                        .iter()
                        .any(|writer| writer.declared.get(&resource.index) == Some(&true));
                    let usage = if written
                        && matches!(binding.resource, BindingResource::StorageBuffer(_))
                    {
                        MTLResourceUsage::Read | MTLResourceUsage::Write
                    } else {
                        MTLResourceUsage::Read
                    };
                    // SAFETY: a live buffer and an offset inside it.
                    unsafe {
                        writer.encoder.setBuffer_offset_atIndex(
                            Some(&buf.buffer.0),
                            buf.offset as usize,
                            resource.index as usize,
                        );
                    }
                    uses.push(Use {
                        resource: Shared(ProtocolObject::from_retained(buf.buffer.0.clone())),
                        usage,
                    });
                }
            }
        }
        // The samplers the layout baked in, which no caller writes.
        for (immutable_set, binding, sampler) in self.immutable_of(pipeline) {
            if immutable_set != set {
                continue;
            }
            let Some(resource) = writers
                .iter()
                .find_map(|writer| find_in(writer.shader, set, binding))
            else {
                continue;
            };
            if let Some(writer) = writer_of(resource.index) {
                self.write_sampler(&writer.encoder, resource.index, sampler);
            }
        }
        self.insert_group(
            Group {
                buffer: Shared(buffer),
                offset,
                uses,
            },
            persistent,
        )
    }

    fn write_texture(
        &self,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        index: u32,
        view: TextureViewId,
        usage: MTLResourceUsage,
        uses: &mut Vec<Use>,
    ) {
        let texture = self.texture(view);
        // SAFETY: a live texture at an index the stage declares.
        unsafe { encoder.setTexture_atIndex(Some(&texture), index as usize) };
        uses.push(Use {
            resource: Shared(ProtocolObject::from_retained(texture)),
            usage,
        });
    }

    fn write_sampler(
        &self,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
        index: u32,
        sampler: SamplerId,
    ) {
        let sampler = self.sampler(sampler);
        // SAFETY: a live sampler built for argument buffers.
        unsafe { encoder.setSamplerState_atIndex(Some(&sampler), index as usize) };
    }

    fn empty_group(&self) -> Group {
        let buffer = self
            .device
            .newBufferWithLength_options(8, MTLResourceOptions::StorageModeShared)
            .expect("failed to allocate an argument buffer");
        Group {
            buffer: Shared(buffer),
            offset: 0,
            uses: Vec::new(),
        }
    }

    fn insert_group(&self, group: Group, persistent: bool) -> BindGroupId {
        let mut groups = self.bind_groups.lock().unwrap();
        let id = groups.insert(group);
        if !persistent {
            groups.remove(id, self.frame.load(Ordering::Relaxed));
        }
        BindGroupId(id)
    }

    /// Argument buffers that live one frame are suballocated from the frame's
    /// arena like any other per-frame block.
    fn frame_arguments(&self, length: u64) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u64) {
        let slot = (self.frame.load(Ordering::Relaxed) % super::FRAME_SLOTS) as usize;
        let mut arenas = self.arenas.lock().unwrap();
        arenas[5][slot].allocate(&self.device, length)
    }

    pub fn create_persistent_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        self.build_group(pipeline, set, bindings, true)
    }

    /// A group valid for the frame being recorded. Metal has no descriptor pool
    /// to cache against, and writing one is a handful of stores into an arena,
    /// so cached and transient are the same operation here.
    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        self.build_group(pipeline, set, bindings, false)
    }

    pub fn create_transient_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        self.build_group(pipeline, set, bindings, false)
    }

    /// The material texture table: `textures` as the array at binding 0 and
    /// `sampler` at binding 1, padded to the shader's array length with the
    /// first texture, which is the white default.
    pub fn create_texture_array_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> BindGroupId {
        let shaders = self.shaders_of(pipeline);
        let length = shaders
            .iter()
            .find_map(|shader| find_in(*shader, set, 0))
            .map_or(textures.len(), |r| r.count as usize);
        let mut bindings: Vec<Binding> = (0..length)
            .map(|i| Binding {
                binding: 0,
                resource: BindingResource::Texture(textures.get(i).copied().unwrap_or(textures[0])),
            })
            .collect();
        bindings.push(Binding {
            binding: 1,
            resource: BindingResource::Sampler(sampler),
        });
        self.build_group(pipeline, set, &bindings, true)
    }

    pub fn destroy_bind_group(&self, group: BindGroupId) {
        self.bind_groups
            .lock()
            .unwrap()
            .remove(group.0, self.frame.load(Ordering::Relaxed));
    }

    pub(super) fn group(&self, group: BindGroupId) -> GroupParts {
        let groups = self.bind_groups.lock().unwrap();
        let found = groups
            .get(group.0)
            .unwrap_or_else(|| panic!("bind group {} is not live", group.0));
        (
            found.buffer.0.clone(),
            found.offset,
            found
                .uses
                .iter()
                .map(|u| (u.resource.0.clone(), u.usage))
                .collect(),
        )
    }
}

#[cfg(test)]
mod declared_tests {
    use super::declared_ids;

    #[test]
    fn a_const_buffer_is_declared_read_only_and_a_pad_is_not_declared() {
        let msl = "struct spvDescriptorSetBuffer0\n{\n    const device A* a [[id(0)]];\n    device B* b [[id(1)]];\n    constant uint* _unused2 [[id(2)]];\n};\n";
        let ids = declared_ids(msl, 0);
        assert_eq!(ids.get(&0), Some(&false));
        assert_eq!(ids.get(&1), Some(&true));
        assert!(!ids.contains_key(&2));
    }
}
