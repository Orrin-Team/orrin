//! Pipelines from the embedded MSL, or a loaded shader's.
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
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use super::convert::pixel_format;
use super::{Held, MtlDriver, Shared, bind};
use crate::gfx::format::SampleCount;
use crate::gfx::rhi::{
    Blend, ComputePipelineDesc, CullMode, DepthCompare, GraphicsPipelineDesc, PipelineId,
    SamplerId, ShaderModuleId, ShaderRef, Topology, VertexFormat, VertexStream,
};
use crate::gfx::shaders::{CompiledShader, ShaderId, ShaderView, VERTEX_STREAM_BASE};

/// A shader as a pipeline holds it. A loaded one is held by the pipelines built
/// from it rather than looked up by handle, so retiring the handle leaves them
/// whole; and it carries its own compiled function and argument encoders,
/// because in the driver-wide caches, keyed by an id the pool recycles, they
/// would be found again by the next shader to land in its slot.
#[derive(Clone)]
pub(super) enum MtlShader {
    BuiltIn(ShaderId),
    Loaded(Arc<Loaded>),
}

impl MtlShader {
    pub(super) fn view(&self) -> ShaderView<'_> {
        match self {
            MtlShader::BuiltIn(id) => id.blob().view(),
            MtlShader::Loaded(loaded) => loaded.shader.view(),
        }
    }
}

/// A loaded shader's MSL and binding map; its SPIR-V is dropped at load.
pub(super) struct Loaded {
    shader: CompiledShader,
    function: Mutex<Option<Held<dyn MTLFunction>>>,
    pub(super) encoders: Mutex<HashMap<u32, bind::Encoding>>,
}

impl Loaded {
    pub(super) fn new(shader: &CompiledShader) -> Self {
        Self {
            shader: CompiledShader {
                spirv: Vec::new(),
                ..shader.clone()
            },
            function: Mutex::new(None),
            encoders: Mutex::new(HashMap::new()),
        }
    }
}

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
                .view()
                .push_constant_bytes
                .max(render.fragment.view().push_constant_bytes),
            Pipe::Compute(compute) => compute.shader.view().push_constant_bytes,
        }) as usize
    }
}

pub(super) struct RenderPipe {
    pub state: Shared<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub depth: Option<Shared<Retained<ProtocolObject<dyn MTLDepthStencilState>>>>,
    pub cull: MTLCullMode,
    pub topology: MTLPrimitiveType,
    pub vertex: MtlShader,
    pub fragment: MtlShader,
    /// Samplers Vulkan bakes into the set layout, which a bind group here has to
    /// write itself because nothing else will: `(set, binding, sampler)`.
    pub immutable: Vec<(u32, u32, SamplerId)>,
}

pub(super) struct ComputePipe {
    pub state: Shared<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub shader: MtlShader,
    pub threads: MTLSize,
    pub immutable: Vec<(u32, u32, SamplerId)>,
}

/// The built-in shaders' compiled functions. A loaded shader keeps its own.
pub(super) type Functions = HashMap<ShaderId, Shared<Retained<ProtocolObject<dyn MTLFunction>>>>;

impl MtlDriver {
    pub fn load_shader(&self, shader: &CompiledShader) -> ShaderModuleId {
        ShaderModuleId(
            self.shaders
                .lock()
                .unwrap()
                .insert(Arc::new(Loaded::new(shader))),
        )
    }

    pub fn destroy_shader(&self, shader: ShaderModuleId) {
        self.shaders
            .lock()
            .unwrap()
            .remove(shader.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn destroy_pipeline(&self, pipeline: PipelineId) {
        self.pipelines
            .lock()
            .unwrap()
            .remove(pipeline.0, self.frame.load(Ordering::Relaxed));
    }

    fn shader(&self, shader: ShaderRef) -> Result<MtlShader, String> {
        match shader {
            ShaderRef::BuiltIn(id) => Ok(MtlShader::BuiltIn(id)),
            ShaderRef::Loaded(id) => self
                .shaders
                .lock()
                .unwrap()
                .get(id.0)
                .cloned()
                .map(MtlShader::Loaded)
                .ok_or_else(|| format!("loaded shader {} was destroyed", id.0)),
        }
    }

    /// The shader's entry function, compiling its library the first time.
    pub(super) fn function(
        &self,
        shader: &MtlShader,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
        match shader {
            MtlShader::BuiltIn(id) => {
                let mut functions = self.functions.lock().unwrap();
                if let Some(function) = functions.get(id) {
                    return Ok(function.0.clone());
                }
                let function = self.compile(shader.view())?;
                functions.insert(*id, Shared(function.clone()));
                Ok(function)
            }
            MtlShader::Loaded(loaded) => {
                let mut kept = loaded.function.lock().unwrap();
                if let Some(function) = &*kept {
                    return Ok(function.0.clone());
                }
                let function = self.compile(shader.view())?;
                *kept = Some(Shared(function.clone()));
                Ok(function)
            }
        }
    }

    fn compile(
        &self,
        shader: ShaderView,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, String> {
        let source = NSString::from_str(shader.msl);
        // `invariant gl_Position` becomes `[[position, invariant]]`, which Metal
        // honours only when told to: without it the prepass and the forward pass
        // can disagree on a depth by a bit, and the forward pass's `Equal` test
        // then rejects every fragment of the opaque queue.
        let options = MTLCompileOptions::new();
        options.setPreserveInvariance(true);
        let library = self
            .device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|error| format!("{} does not compile as MSL: {error}", shader.name))?;
        // The translator's name for a GLSL `main`.
        library
            .newFunctionWithName(ns_string!("main0"))
            .ok_or_else(|| format!("{} has no entry point", shader.name))
    }

    pub fn try_create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<PipelineId, String> {
        let vertex_shader = self.shader(desc.vertex)?;
        let fragment_shader = self.shader(desc.fragment)?;
        let vertex = self.function(&vertex_shader)?;
        let fragment = self.function(&fragment_shader)?;
        let named = |e: &dyn std::fmt::Display| {
            format!(
                "{} with {}: {e}",
                vertex_shader.view().name,
                fragment_shader.view().name
            )
        };

        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(&vertex));
        descriptor.setFragmentFunction(Some(&fragment));
        if !desc.vertex_streams.is_empty() {
            descriptor.setVertexDescriptor(Some(&vertex_descriptor(
                &vertex_shader,
                desc.vertex_streams,
            )));
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
            .map_err(|error| named(&error))?;

        let depth = desc.depth_test.map(|test| {
            let stencil = MTLDepthStencilDescriptor::new();
            stencil.setDepthCompareFunction(match test.compare {
                DepthCompare::Less => MTLCompareFunction::Less,
                DepthCompare::Equal => MTLCompareFunction::Equal,
                DepthCompare::LessOrEqual => MTLCompareFunction::LessEqual,
                DepthCompare::Greater => MTLCompareFunction::Greater,
                DepthCompare::GreaterOrEqual => MTLCompareFunction::GreaterEqual,
            });
            stencil.setDepthWriteEnabled(test.write);
            self.device
                .newDepthStencilStateWithDescriptor(&stencil)
                .map(Shared)
                .ok_or_else(|| named(&"the device made no depth state"))
        });
        let depth = depth.transpose()?;

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
            vertex: vertex_shader,
            fragment: fragment_shader,
            immutable,
        });
        Ok(PipelineId(self.pipelines.lock().unwrap().insert(pipe)))
    }

    fn try_create_compute_pipeline(
        &self,
        desc: &ComputePipelineDesc,
    ) -> Result<PipelineId, String> {
        let shader = self.shader(desc.shader)?;
        let function = self.function(&shader)?;
        let name = shader.view().name;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| format!("{name}: {error}"))?;
        let [x, y, z] = shader
            .view()
            .workgroup_size
            .ok_or_else(|| format!("{name} declares no workgroup size"))?;
        let pipe = Pipe::Compute(ComputePipe {
            state: Shared(state),
            shader,
            threads: MTLSize {
                width: x as usize,
                height: y as usize,
                depth: z as usize,
            },
            immutable: Vec::new(),
        });
        Ok(PipelineId(self.pipelines.lock().unwrap().insert(pipe)))
    }

    /// A comparison or fixed sampler is an ordinary argument-buffer sampler
    /// here, so the layouts Vulkan needs baked in need nothing.
    pub fn try_create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> Result<PipelineId, String> {
        let id = self.try_create_compute_pipeline(desc)?;
        if let Pipe::Compute(compute) = self
            .pipelines
            .lock()
            .unwrap()
            .get_mut(id.0)
            .expect("just created")
        {
            compute.immutable = immutable_samplers.to_vec();
        }
        Ok(id)
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

fn vertex_descriptor(
    shader: &MtlShader,
    streams: &[VertexStream],
) -> Retained<MTLVertexDescriptor> {
    let attributes_of = stage_in_attributes(shader.view().msl);
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
                VertexFormat::Snorm16x2 => MTLVertexFormat::Short2Normalized,
                VertexFormat::Snorm16x4 => MTLVertexFormat::Short4Normalized,
            });
            unsafe { described.setOffset(attribute.offset as usize) };
            unsafe { described.setBufferIndex(buffer) };
        }
    }
    descriptor
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::gfx::rhi::{Device, TextureViewId};
    use crate::gfx::vulkan::driver::tests as shared;

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

    pub(in crate::gfx::metal) fn loaded(id: ShaderId) -> MtlShader {
        MtlShader::Loaded(Arc::new(Loaded::new(&CompiledShader::from(id.blob()))))
    }

    /// A loaded shader is read through the same accessor as a built-in one, so
    /// the same bytes have to give the same answers.
    #[test]
    fn a_loaded_shader_reads_as_the_built_in_one_it_was_made_from() {
        let built_in = MtlShader::BuiltIn(ShaderId::PrepassVert);
        let loaded = loaded(ShaderId::PrepassVert);
        assert_eq!(
            stage_in_attributes(loaded.view().msl),
            stage_in_attributes(built_in.view().msl)
        );
        assert_eq!(loaded.view().name, built_in.view().name);
        assert_eq!(loaded.view().resources, built_in.view().resources);
        assert_eq!(
            loaded.view().push_constant_bytes,
            built_in.view().push_constant_bytes
        );
        assert!(
            loaded.view().spirv.is_empty(),
            "Metal keeps the MSL, not the SPIR-V"
        );
    }

    // On a device: the Vulkan driver's test bodies, against this driver.

    fn metal() -> Device {
        Device::Metal(MtlDriver::new().expect("this machine runs the Metal backend"))
    }

    fn read_metal(device: &Device, view: TextureViewId) -> Vec<u8> {
        device.metal().read_texture(view)
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_loaded_shader_draws_what_its_built_in_twin_draws_on_metal() {
        shared::a_loaded_shader_draws_what_its_built_in_twin_draws(&metal(), &read_metal);
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn msl_that_does_not_compile_is_an_error_on_metal() {
        let broken = CompiledShader {
            msl: "this is not Metal".to_string(),
            ..CompiledShader::from(ShaderId::OutlineMaskFrag.blob())
        };
        shared::a_broken_loaded_shader_is_an_error(&metal(), &broken);
    }

    /// The library compiles and has its `main0`; it is the pipeline state that
    /// Metal refuses, because an integer output cannot be written to a UNORM
    /// attachment.
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_pipeline_metal_cannot_build_is_an_error_on_metal() {
        let broken = CompiledShader {
            name: "integer_output".to_string(),
            msl: "#include <metal_stdlib>\nusing namespace metal;\n\
                  struct main0_out { int4 f_color [[color(0)]]; };\n\
                  fragment main0_out main0() { main0_out out = {}; out.f_color = int4(1); return out; }\n"
                .to_string(),
            ..CompiledShader::from(ShaderId::OutlineMaskFrag.blob())
        };
        shared::a_broken_loaded_shader_is_an_error(&metal(), &broken);
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn retired_pipelines_and_shaders_are_recycled_safely_on_metal() {
        shared::retired_pipelines_and_shaders_are_recycled_safely(&metal(), &read_metal);
    }
}
