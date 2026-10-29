//! Bind groups as argument buffers.
//!
//! A descriptor set is one argument buffer bound at the buffer index of its set
//! number. What goes in it is found from the shader's own binding map: a
//! resource declared at `(set, binding)` lives at that map's `index`, which is
//! the same for every shader (see `build.rs`), so one group serves a pipeline's
//! vertex and fragment stages alike. The argument encoder writes it, which is
//! what keeps its layout correct by construction (trap T3).
//!
//! A resource reached only through an argument buffer is invisible to the GPU
//! unless the encoder says it is in use (trap T1), so each group carries the
//! list of what it holds and the encoder declares it when the group is bound.

use std::sync::atomic::Ordering;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLFunction, MTLResource, MTLResourceID, MTLResourceOptions,
    MTLResourceUsage, MTLSamplerState, MTLTexture,
};

use super::pipeline::Pipe;
use super::{MtlDriver, Shared};
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, PipelineId, SamplerId, TextureViewId,
};
use crate::gfx::shaders::{Resource, ResourceKind, ShaderId};

/// An argument buffer being written: eight bytes a slot, at the index the
/// binding map gave.
struct Slots(*mut u8);

impl Slots {
    fn write_id(&self, index: u32, id: MTLResourceID) {
        // SAFETY: the slot lies inside the buffer this was made over, and a
        // resource id is eight bytes.
        unsafe {
            self.0
                .add(index as usize * 8)
                .cast::<MTLResourceID>()
                .write(id)
        };
    }

    fn write_address(&self, index: u32, address: u64) {
        // SAFETY: as above.
        unsafe { self.0.add(index as usize * 8).cast::<u64>().write(address) };
    }
}

pub(super) struct Group {
    pub buffer: Shared<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub offset: u64,
    pub uses: Vec<Use>,
}

/// A group as an encoder binds it: the argument buffer, where in it the group
/// begins, and what it holds.
pub(super) type GroupParts = (
    Retained<ProtocolObject<dyn MTLBuffer>>,
    u64,
    Vec<(Retained<ProtocolObject<dyn MTLResource>>, MTLResourceUsage)>,
);

pub(super) struct Use {
    pub resource: Shared<Retained<ProtocolObject<dyn MTLResource>>>,
    pub usage: MTLResourceUsage,
}

/// Where the groups of one pipeline's set come from.
struct Layout {
    shaders: Vec<ShaderId>,
    function: Option<Retained<ProtocolObject<dyn MTLFunction>>>,
}

fn find(shaders: &[ShaderId], set: u32, binding: u32) -> Option<Resource> {
    shaders.iter().find_map(|shader| {
        shader
            .blob()
            .bindings
            .resources
            .iter()
            .find(|resource| resource.set == set && resource.binding == binding)
            .copied()
    })
}

impl MtlDriver {
    fn layout(&self, pipeline: PipelineId, set: u32) -> Layout {
        let shaders = self.with_pipe(pipeline, |pipe| match pipe {
            Pipe::Render(render) => vec![render.vertex, render.fragment],
            Pipe::Compute(compute) => vec![compute.shader],
        });
        // The function that declares this set's struct, which is the one whose
        // argument encoder describes it.
        let marker = format!("struct spvDescriptorSetBuffer{set}\n");
        let declaring = shaders
            .iter()
            .copied()
            .find(|shader| shader.blob().msl.contains(&marker));
        let function = declaring.map(|shader| self.function(shader));
        Layout { shaders, function }
    }

    /// Write `bindings` into a fresh argument buffer for `set` of `pipeline`.
    /// A set no stage reads gets an empty one, so binding it is harmless.
    fn build_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
        persistent: bool,
    ) -> BindGroupId {
        let layout = self.layout(pipeline, set);
        if layout.function.is_none() {
            return self.insert_group(self.empty_group(), persistent);
        }
        // Every resource of the set takes eight bytes at its index, whichever
        // stage reads it: the layout is fixed by `build.rs`, so the buffer is
        // written directly and one group serves vertex and fragment alike. (An
        // argument encoder describes only the one function it was made from,
        // and so cannot write a binding another stage declares.)
        let length = layout
            .shaders
            .iter()
            .flat_map(|shader| shader.blob().bindings.resources)
            .filter(|resource| resource.set == set)
            .map(|resource| {
                let width = if resource.kind == ResourceKind::CombinedImageSampler {
                    2
                } else {
                    1
                };
                (resource.index + resource.count * width) as u64 * 8
            })
            .max()
            .unwrap_or(8);

        let (buffer, offset) = if persistent {
            let buffer = self
                .device
                .newBufferWithLength_options(length as usize, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate an argument buffer");
            (buffer, 0)
        } else {
            self.frame_arguments(length)
        };
        // SAFETY: a shared buffer with `length` bytes from `offset`, which is
        // eight-byte aligned.
        let base = unsafe { buffer.contents().as_ptr().cast::<u8>().add(offset as usize) };
        let encoder = Slots(base);

        let mut uses = Vec::new();
        // Textures at one binding are an array, filled in the order given.
        let mut next_in_array: Vec<(u32, u32)> = Vec::new();
        for binding in bindings {
            let Some(resource) = find(&layout.shaders, set, binding.binding) else {
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
                    self.write_texture(
                        &encoder,
                        slot(&mut next_in_array),
                        view,
                        MTLResourceUsage::Read,
                        &mut uses,
                    );
                    if resource.kind == ResourceKind::CombinedImageSampler {
                        self.write_sampler(&encoder, resource.index + resource.count, sampler);
                    }
                }
                BindingResource::Texture(view) => {
                    self.write_texture(
                        &encoder,
                        slot(&mut next_in_array),
                        view,
                        MTLResourceUsage::Read,
                        &mut uses,
                    );
                }
                BindingResource::StorageTexture(view) => {
                    self.write_texture(
                        &encoder,
                        slot(&mut next_in_array),
                        view,
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                        &mut uses,
                    );
                }
                BindingResource::Sampler(sampler) => {
                    self.write_sampler(&encoder, resource.index, sampler);
                }
                BindingResource::StorageBuffer(id) | BindingResource::UniformBuffer(id) => {
                    let buf = self.buffer(id);
                    let usage = if matches!(binding.resource, BindingResource::StorageBuffer(_)) {
                        MTLResourceUsage::Read | MTLResourceUsage::Write
                    } else {
                        MTLResourceUsage::Read
                    };
                    encoder.write_address(resource.index, buf.buffer.0.gpuAddress() + buf.offset);
                    uses.push(Use {
                        resource: Shared(ProtocolObject::from_retained(buf.buffer.0.clone())),
                        usage,
                    });
                }
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
        encoder: &Slots,
        index: u32,
        view: TextureViewId,
        usage: MTLResourceUsage,
        uses: &mut Vec<Use>,
    ) {
        let texture = self.texture(view);
        encoder.write_id(index, texture.gpuResourceID());
        uses.push(Use {
            resource: Shared(ProtocolObject::from_retained(texture)),
            usage,
        });
    }

    fn write_sampler(&self, encoder: &Slots, index: u32, sampler: SamplerId) {
        let sampler = self.sampler(sampler);
        encoder.write_id(index, sampler.gpuResourceID());
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
        let layout = self.layout(pipeline, set);
        let length = find(&layout.shaders, set, 0).map_or(textures.len(), |r| r.count as usize);
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

    #[cfg(test)]
    pub(crate) fn group_for_test(&self, group: BindGroupId) -> GroupParts {
        self.group(group)
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
