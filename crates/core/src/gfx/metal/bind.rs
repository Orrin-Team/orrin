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

use std::collections::HashSet;
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
use crate::gfx::shaders::{Resource, ResourceKind, ShaderId, Stage};

/// One stage's argument buffer, and what it holds.
pub(super) struct StageGroup {
    pub stage: Stage,
    pub buffer: Shared<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub offset: u64,
    pub uses: Vec<Use>,
}

pub(super) struct Group {
    pub stages: Vec<StageGroup>,
}

/// A group as an encoder binds it.
pub(super) type GroupParts = Vec<(
    Stage,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    u64,
    Vec<(Retained<ProtocolObject<dyn MTLResource>>, MTLResourceUsage)>,
)>;

pub(super) struct Use {
    pub resource: Shared<Retained<ProtocolObject<dyn MTLResource>>>,
    pub usage: MTLResourceUsage,
}

/// The `[[id]]`s a shader's translated source declares in one descriptor set's
/// struct. A resource the translator proved unused is in the binding map and not
/// here, and writing it would be out of the argument buffer's range.
fn declared_ids(msl: &str, set: u32) -> HashSet<u32> {
    let marker = format!("struct spvDescriptorSetBuffer{set}\n");
    let Some(body) = msl.split(&marker).nth(1) else {
        return HashSet::new();
    };
    let body = body.split("};").next().unwrap_or("");
    body.match_indices("[[id(")
        .filter_map(|(at, _)| {
            let digits = &body[at + 5..];
            digits[..digits.find(')')?].parse().ok()
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

/// The stages of a pipeline that declare `set`, with the declared ids of each.
fn stages_of(shaders: &[ShaderId], set: u32) -> Vec<(ShaderId, HashSet<u32>)> {
    shaders
        .iter()
        .copied()
        .map(|shader| (shader, declared_ids(shader.blob().msl, set)))
        .filter(|(_, ids)| !ids.is_empty())
        .collect()
}

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
        let mut stages = Vec::new();
        for (shader, declared) in stages_of(&shaders, set) {
            let function = self.function(shader);
            // SAFETY: `set` is a buffer index this function declares an
            // argument buffer at, which `declared_ids` just established.
            let encoder = unsafe { function.newArgumentEncoderWithBufferIndex(set as usize) };
            let length = encoder.encodedLength().max(8) as u64;
            let (buffer, offset) = if persistent {
                let buffer = self
                    .device
                    .newBufferWithLength_options(
                        length as usize,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("failed to allocate an argument buffer");
                (buffer, 0)
            } else {
                self.frame_arguments(length)
            };
            // SAFETY: the buffer is at least `encodedLength` bytes from `offset`,
            // at the alignment the arena and a fresh buffer both give.
            unsafe { encoder.setArgumentBuffer_offset(Some(&buffer), offset as usize) };

            let mut uses = Vec::new();
            // Textures at one binding are an array, filled in the order given.
            let mut next_in_array: Vec<(u32, u32)> = Vec::new();
            for binding in bindings {
                let Some(resource) = find_in(shader, set, binding.binding) else {
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
                        if declared.contains(&resource.index) {
                            self.write_texture(
                                &encoder,
                                index,
                                view,
                                MTLResourceUsage::Read,
                                &mut uses,
                            );
                        }
                        let at = resource.index + resource.count;
                        if resource.kind == ResourceKind::CombinedImageSampler
                            && declared.contains(&at)
                        {
                            self.write_sampler(&encoder, at, sampler);
                        }
                    }
                    BindingResource::Texture(view) => {
                        let index = slot(&mut next_in_array);
                        if declared.contains(&resource.index) {
                            self.write_texture(
                                &encoder,
                                index,
                                view,
                                MTLResourceUsage::Read,
                                &mut uses,
                            );
                        }
                    }
                    BindingResource::StorageTexture(view) => {
                        let index = slot(&mut next_in_array);
                        if declared.contains(&resource.index) {
                            self.write_texture(
                                &encoder,
                                index,
                                view,
                                MTLResourceUsage::Read | MTLResourceUsage::Write,
                                &mut uses,
                            );
                        }
                    }
                    BindingResource::Sampler(sampler) => {
                        if declared.contains(&resource.index) {
                            self.write_sampler(&encoder, resource.index, sampler);
                        }
                    }
                    BindingResource::StorageBuffer(id) | BindingResource::UniformBuffer(id) => {
                        if !declared.contains(&resource.index) {
                            continue;
                        }
                        let buf = self.buffer(id);
                        let usage = if matches!(binding.resource, BindingResource::StorageBuffer(_))
                        {
                            MTLResourceUsage::Read | MTLResourceUsage::Write
                        } else {
                            MTLResourceUsage::Read
                        };
                        // SAFETY: a live buffer and an offset inside it.
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
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
            stages.push(StageGroup {
                stage: shader.blob().stage,
                buffer: Shared(buffer),
                offset,
                uses,
            });
        }
        self.insert_group(Group { stages }, persistent)
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
        found
            .stages
            .iter()
            .map(|stage| {
                (
                    stage.stage,
                    stage.buffer.0.clone(),
                    stage.offset,
                    stage
                        .uses
                        .iter()
                        .map(|u| (u.resource.0.clone(), u.usage))
                        .collect(),
                )
            })
            .collect()
    }
}
