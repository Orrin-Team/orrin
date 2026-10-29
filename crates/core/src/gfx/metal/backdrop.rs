//! The overlay's backdrop capture on Metal.
//!
//! Vulkan scales with a blit; a Metal blit copies texel for texel and cannot, so
//! each halving is a fullscreen triangle that reads the larger level through a
//! bilinear sampler. The shader is a few lines of MSL kept here rather than in
//! the shader manifest: it belongs to this backend alone, since the other one
//! has no use for it.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, ns_string};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLLibrary, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction, MTLTexture,
};

use super::{MtlDriver, Shared};

const SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;
struct V { float4 position [[position]]; float2 uv; };
vertex V downsample_vs(uint i [[vertex_id]]) {
    float2 c = float2((i << 1) & 2, i & 2);
    V o;
    o.position = float4(c * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(c.x, 1.0 - c.y);
    return o;
}
fragment float4 downsample_fs(V in [[stage_in]], texture2d<float> source [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return source.sample(s, in.uv);
}
";

/// Pipelines for the downsample, one per colour format.
pub(super) type Downsamples = Vec<(
    MTLPixelFormat,
    Shared<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
)>;

impl MtlDriver {
    fn downsample_pipeline(
        &self,
        format: MTLPixelFormat,
    ) -> Retained<ProtocolObject<dyn MTLRenderPipelineState>> {
        let mut cache = self.downsamples.lock().unwrap();
        if let Some((_, pipeline)) = cache.iter().find(|(known, _)| *known == format) {
            return pipeline.0.clone();
        }
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .unwrap_or_else(|error| panic!("the backdrop downsample does not compile: {error}"));
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(
            library
                .newFunctionWithName(ns_string!("downsample_vs"))
                .as_deref(),
        );
        descriptor.setFragmentFunction(
            library
                .newFunctionWithName(ns_string!("downsample_fs"))
                .as_deref(),
        );
        // SAFETY: attachment 0 of a descriptor is always valid to address.
        unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) }.setPixelFormat(format);
        let pipeline = self
            .device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .unwrap_or_else(|error| panic!("the backdrop downsample has no pipeline: {error}"));
        cache.push((format, Shared(pipeline.clone())));
        pipeline
    }

    /// Fill `into` with a bilinear reduction of `from`.
    pub(super) fn downsample(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        from: &ProtocolObject<dyn MTLTexture>,
        into: &ProtocolObject<dyn MTLTexture>,
    ) {
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: attachment 0 of a descriptor is always valid to address.
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(into));
        attachment.setLoadAction(MTLLoadAction::DontCare);
        attachment.setStoreAction(MTLStoreAction::Store);
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("failed to open a render encoder");
        encoder.setRenderPipelineState(&self.downsample_pipeline(into.pixelFormat()));
        // SAFETY: slot 0 is the one texture the fragment function declares.
        unsafe { encoder.setFragmentTexture_atIndex(Some(from), 0) };
        // SAFETY: three vertices make the one triangle the vertex function emits.
        unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.endEncoding();
    }
}
