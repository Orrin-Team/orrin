//! Pipelines from the embedded MSL.
//!
//! Each shader's translated source is compiled to a function on first use and
//! kept; a pipeline is that function plus the fixed-function state the RHI
//! describes. Vertex inputs are matched to the shader by *name*, as the Vulkan
//! driver does: the MSL names its stage-in attributes after the GLSL inputs,
//! and the translator numbered them by location, so the name is what ties a
//! `VertexStream` attribute to its `[[attribute(n)]]`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, ns_string};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction, MTLCompileOptions,
    MTLComputePipelineState, MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState,
    MTLDevice, MTLFunction, MTLLibrary, MTLPrimitiveType, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLSize, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};
use std::collections::HashMap;

use super::convert::pixel_format;
use super::{MtlDriver, Shared};
use crate::gfx::format::SampleCount;
use crate::gfx::rhi::{
    Blend, ComputePipelineDesc, CullMode, DepthCompare, GraphicsPipelineDesc, PipelineId,
    SamplerId, Topology, VertexFormat, VertexStream,
};
use crate::gfx::shaders::{ShaderId, VERTEX_STREAM_BASE};

pub(super) enum Pipe {
    Render(RenderPipe),
    Compute(ComputePipe),
}

impl Pipe {
    /// The most push-constant bytes any stage declares.
    pub(super) fn push_constant_bytes(&self) -> usize {
        (match self {
            Pipe::Render(render) => render
                .vertex
                .blob()
                .bindings
                .push_constant_bytes
                .max(render.fragment.blob().bindings.push_constant_bytes),
            Pipe::Compute(compute) => compute.shader.blob().bindings.push_constant_bytes,
        }) as usize
    }
}

pub(super) struct RenderPipe {
    pub state: Shared<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub depth: Option<Shared<Retained<ProtocolObject<dyn MTLDepthStencilState>>>>,
    pub cull: MTLCullMode,
    pub topology: MTLPrimitiveType,
    pub vertex: ShaderId,
    pub fragment: ShaderId,
    /// Samplers Vulkan bakes into the set layout, which a bind group here has to
    /// write itself because nothing else will: `(set, binding, sampler)`.
    pub immutable: Vec<(u32, u32, SamplerId)>,
}

pub(super) struct ComputePipe {
    pub state: Shared<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub shader: ShaderId,
    pub threads: MTLSize,
    pub immutable: Vec<(u32, u32, SamplerId)>,
}

/// Compiled functions, one per shader.
pub(super) type Functions = HashMap<ShaderId, Shared<Retained<ProtocolObject<dyn MTLFunction>>>>;

impl MtlDriver {
    /// The shader's entry function, compiling its library the first time.
    pub(super) fn function(&self, shader: ShaderId) -> Retained<ProtocolObject<dyn MTLFunction>> {
        let mut functions = self.functions.lock().unwrap();
        if let Some(function) = functions.get(&shader) {
            return function.0.clone();
        }
        let blob = shader.blob();
        let source = NSString::from_str(blob.msl);
        // `invariant gl_Position` becomes `[[position, invariant]]`, which Metal
        // honours only when told to: without it the prepass and the forward pass
        // can disagree on a depth by a bit, and the forward pass's `Equal` test
        // then rejects every fragment of the opaque queue.
        let options = MTLCompileOptions::new();
        options.setPreserveInvariance(true);
        let library = self
            .device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .unwrap_or_else(|error| panic!("{} does not compile as MSL: {error}", blob.name));
        // The translator's name for a GLSL `main`.
        let function = library
            .newFunctionWithName(ns_string!("main0"))
            .unwrap_or_else(|| panic!("{} has no entry point", blob.name));
        functions.insert(shader, Shared(function.clone()));
        function
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        let vertex = self.function(desc.vertex);
        let fragment = self.function(desc.fragment);

        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        if !desc.vertex_streams.is_empty() {
            descriptor
                .setVertexDescriptor(Some(&vertex_descriptor(desc.vertex, desc.vertex_streams)));
        }
        let attachments = descriptor.colorAttachments();
        for (index, &format) in desc.color_formats.iter().enumerate() {
            let attachment = unsafe { attachments.objectAtIndexedSubscript(index) };
            attachment.setPixelFormat(pixel_format(format));
            let blend = desc
                .attachment_blends
                .get(index)
                .copied()
                .unwrap_or(desc.blend);
            if let Some(factors) = blend_factors(blend) {
                attachment.setBlendingEnabled(true);
                attachment.setSourceRGBBlendFactor(factors.src_color);
                attachment.setDestinationRGBBlendFactor(factors.dst_color);
                attachment.setRgbBlendOperation(factors.op);
                attachment.setSourceAlphaBlendFactor(factors.src_alpha);
                attachment.setDestinationAlphaBlendFactor(factors.dst_alpha);
                attachment.setAlphaBlendOperation(factors.op);
            }
            // The shader declares one output, and an undeclared one left
            // unmasked is undefined data under every blur tap near a silhouette.
            if desc.mask_extra_attachments && index > 0 {
                attachment.setWriteMask(MTLColorWriteMask::empty());
            }
        }
        if let Some(format) = desc.depth_format {
            descriptor.setDepthAttachmentPixelFormat(pixel_format(format));
        }
        descriptor.setRasterSampleCount(match desc.samples {
            SampleCount::Sample1 => 1,
            SampleCount::Sample2 => 2,
            SampleCount::Sample4 => 4,
            SampleCount::Sample8 => 8,
        });
        descriptor.setAlphaToCoverageEnabled(desc.alpha_to_coverage);

        let state = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .unwrap_or_else(|error| panic!("a render pipeline is invalid: {error}"));

        let depth = desc.depth_test.map(|test| {
            let stencil = MTLDepthStencilDescriptor::new();
            stencil.setDepthCompareFunction(match test.compare {
                DepthCompare::Less => MTLCompareFunction::Less,
                DepthCompare::Equal => MTLCompareFunction::Equal,
                DepthCompare::LessOrEqual => MTLCompareFunction::LessEqual,
            });
            stencil.setDepthWriteEnabled(test.write);
            Shared(
                self.device
                    .newDepthStencilStateWithDescriptor(&stencil)
                    .expect("failed to create a depth state"),
            )
        });

        // What this pipeline's own description bakes in, then what the pipeline
        // whose layout it borrows does: a pipeline built to share another's sets
        // is bound against groups written for them, and reads the same samplers.
        let mut immutable: Vec<_> = desc.immutable_samplers.to_vec();
        let inherited = desc
            .layout_from
            .map(|source| (source, u32::MAX))
            .or(desc.layout_prefix_from);
        if let Some((source, sets)) = inherited {
            immutable.extend(
                self.immutable_of(source)
                    .into_iter()
                    .filter(|(set, _, _)| *set < sets),
            );
        }
        let pipe = Pipe::Render(RenderPipe {
            state: Shared(state),
            depth,
            cull: match desc.cull {
                CullMode::None => MTLCullMode::None,
                CullMode::Back => MTLCullMode::Back,
            },
            topology: match desc.topology {
                Topology::Triangles => MTLPrimitiveType::Triangle,
                Topology::Lines => MTLPrimitiveType::Line,
            },
            vertex: desc.vertex,
            fragment: desc.fragment,
            immutable,
        });
        PipelineId(self.pipelines.lock().unwrap().insert(pipe))
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> PipelineId {
        let function = self.function(desc.shader);
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|error| panic!("a compute pipeline is invalid: {error}"));
        let [x, y, z] = desc
            .shader
            .blob()
            .bindings
            .workgroup_size
            .expect("a compute shader declares its workgroup size");
        let pipe = Pipe::Compute(ComputePipe {
            state: Shared(state),
            shader: desc.shader,
            threads: MTLSize {
                width: x as usize,
                height: y as usize,
                depth: z as usize,
            },
            immutable: Vec::new(),
        });
        PipelineId(self.pipelines.lock().unwrap().insert(pipe))
    }

    /// A comparison or fixed sampler is an ordinary argument-buffer sampler
    /// here, so the layouts Vulkan needs baked in need nothing.
    pub fn create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> PipelineId {
        let id = self.create_compute_pipeline(desc);
        if let Pipe::Compute(compute) = self
            .pipelines
            .lock()
            .unwrap()
            .get_mut(id.0)
            .expect("just created")
        {
            compute.immutable = immutable_samplers.to_vec();
        }
        id
    }

    /// The samplers a pipeline has baked into its layout.
    pub(super) fn immutable_of(&self, id: PipelineId) -> Vec<(u32, u32, SamplerId)> {
        self.with_pipe(id, |pipe| match pipe {
            Pipe::Render(render) => render.immutable.clone(),
            Pipe::Compute(compute) => compute.immutable.clone(),
        })
    }

    pub(super) fn with_pipe<R>(&self, id: PipelineId, f: impl FnOnce(&Pipe) -> R) -> R {
        let pipelines = self.pipelines.lock().unwrap();
        f(pipelines
            .get(id.0)
            .unwrap_or_else(|| panic!("pipeline {} is not live", id.0)))
    }
}

struct Factors {
    src_color: MTLBlendFactor,
    dst_color: MTLBlendFactor,
    src_alpha: MTLBlendFactor,
    dst_alpha: MTLBlendFactor,
    op: MTLBlendOperation,
}

fn blend_factors(blend: Blend) -> Option<Factors> {
    let same = |src, dst, op| Factors {
        src_color: src,
        dst_color: dst,
        src_alpha: src,
        dst_alpha: dst,
        op,
    };
    use MTLBlendFactor as F;
    use MTLBlendOperation as Op;
    match blend {
        Blend::None => None,
        Blend::Alpha => Some(Factors {
            src_color: F::SourceAlpha,
            dst_color: F::OneMinusSourceAlpha,
            src_alpha: F::One,
            dst_alpha: F::OneMinusSourceAlpha,
            op: Op::Add,
        }),
        Blend::Premultiplied => Some(same(F::One, F::OneMinusSourceAlpha, Op::Add)),
        Blend::Max => Some(same(F::One, F::One, Op::Max)),
        Blend::Additive => Some(same(F::One, F::One, Op::Add)),
        Blend::Revealage => Some(Factors {
            src_color: F::Zero,
            dst_color: F::OneMinusSourceColor,
            src_alpha: F::Zero,
            dst_alpha: F::OneMinusSourceAlpha,
            op: Op::Add,
        }),
        Blend::PremultipliedCoverage => Some(Factors {
            src_color: F::One,
            dst_color: F::OneMinusSourceAlpha,
            src_alpha: F::OneMinusDestinationAlpha,
            dst_alpha: F::One,
            op: Op::Add,
        }),
    }
}

/// The `[[attribute(n)]]` each named input of a vertex shader's MSL was given.
pub(super) fn stage_in_attributes(msl: &str) -> Vec<(&str, u32)> {
    let Some(body) = msl.split("struct main0_in\n").nth(1) else {
        return Vec::new();
    };
    let body = body.split("};").next().unwrap_or("");
    body.lines()
        .filter_map(|line| {
            let (declaration, rest) = line.split_once("[[attribute(")?;
            let index = rest.split(')').next()?.parse().ok()?;
            let name = declaration.split_whitespace().last()?;
            Some((name, index))
        })
        .collect()
}

fn vertex_descriptor(shader: ShaderId, streams: &[VertexStream]) -> Retained<MTLVertexDescriptor> {
    let attributes_of = stage_in_attributes(shader.blob().msl);
    let descriptor = MTLVertexDescriptor::new();
    for (slot, stream) in streams.iter().enumerate() {
        let buffer = (VERTEX_STREAM_BASE as usize) + slot;
        let layout = unsafe { descriptor.layouts().objectAtIndexedSubscript(buffer) };
        unsafe { layout.setStride(stream.stride as usize) };
        layout.setStepFunction(if stream.per_instance {
            MTLVertexStepFunction::PerInstance
        } else {
            MTLVertexStepFunction::PerVertex
        });
        for attribute in stream.attributes {
            // An input the shader never reads is absent from its translated
            // stage-in, and a stream may carry more than one shader uses.
            let Some((_, location)) = attributes_of
                .iter()
                .find(|(name, _)| *name == attribute.name)
            else {
                continue;
            };
            let described = unsafe {
                descriptor
                    .attributes()
                    .objectAtIndexedSubscript(*location as usize)
            };
            described.setFormat(match attribute.format {
                VertexFormat::Float2 => MTLVertexFormat::Float2,
                VertexFormat::Float3 => MTLVertexFormat::Float3,
                VertexFormat::Float4 => MTLVertexFormat::Float4,
                VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
            });
            unsafe { described.setOffset(attribute.offset as usize) };
            unsafe { described.setBufferIndex(buffer) };
        }
    }
    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vertex streams' attribute names are what bind them to the shader, so
    /// every one a stream declares has to be an input of the shader it is used
    /// with. Device-free: it reads the translated source.
    #[test]
    fn the_stage_in_names_of_the_prepass_are_found() {
        let names: Vec<_> = stage_in_attributes(ShaderId::PrepassVert.blob().msl)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert!(names.contains(&"position"), "{names:?}");
    }

    #[test]
    fn a_shader_without_vertex_inputs_has_none() {
        assert!(stage_in_attributes(ShaderId::FullscreenVert.blob().msl).is_empty());
    }
}
