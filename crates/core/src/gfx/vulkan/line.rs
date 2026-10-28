//! Debug-line overlay pass.
//!
//! Draws the engine's per-frame [`DebugLine`] buffer as GPU line primitives into
//! the forward (HDR) render pass, right after the scene geometry. Sharing that
//! subpass is deliberate: the pass reuses the existing depth buffer, so debug
//! lines are correctly *occluded* by scene geometry, and it reuses the camera
//! view-projection. The one accepted cost is colour: the forward target is HDR
//! and tonemapped downstream, so a line's on-screen colour drifts from the exact
//! RGBA the script requested. Drawing post-tonemap would fix the colour but lose
//! depth occlusion — see the design notes on the issue.

use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;

use crate::gfx::format::SampleCount;
use crate::gfx::rhi::{
    CullMode, DepthCompare, DepthTest, Device, Encoder, GraphicsPipelineDesc, PipelineId, Topology,
    VertexAttribute, VertexFormat, VertexStream, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::DebugLine;

use super::MSAA_SAMPLES;
use super::forward::ForwardTargets;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;

/// One endpoint of a debug line: world-space position + RGBA colour. Two of
/// these make a segment, drawn with `PrimitiveTopology::LineList`.
#[derive(BufferContents, VertexTrait, Clone, Copy)]
#[repr(C)]
pub struct LineVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl LineVertex {
    /// This stream as the RHI describes it; a test holds it against the layout
    /// the `Vertex` derive produces.
    pub const STREAM: VertexStream = VertexStream {
        stride: 28,
        per_instance: false,
        attributes: &[
            VertexAttribute {
                name: "position",
                format: VertexFormat::Float3,
                offset: 0,
            },
            VertexAttribute {
                name: "color",
                format: VertexFormat::Float4,
                offset: 12,
            },
        ],
    };
}

pub struct LinePass {
    /// Indexed `[msaa][subsurface]`, matching the four shapes of
    /// [`ForwardTargets`]: two sample counts, each with and without the
    /// diffusible colour target.
    pipelines: [[PipelineId; 2]; 2],
}

impl LinePass {
    /// Build one line pipeline per forward target shape.
    pub fn new(device: &Device, targets: &ForwardTargets) -> Self {
        let pipeline = |msaa: bool, subsurface: bool| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[LineVertex::STREAM],
                cull: CullMode::None,
                // Tested against the scene, never written.
                depth_test: Some(DepthTest {
                    write: false,
                    compare: DepthCompare::Less,
                }),
                depth_format: Some(DEPTH_FORMAT),
                // A description carries formats, not a sample count, and a
                // pipeline whose count disagrees with the attachments it is used
                // with is invalid. The two travel together from `ForwardTargets`.
                samples: if msaa {
                    MSAA_SAMPLES
                } else {
                    SampleCount::Sample1
                },
                topology: Topology::Lines,
                // Masked past the first attachment: this shader declares one
                // output, and the subsurface target shape has two. See
                // `GraphicsPipelineDesc::mask_extra_attachments`.
                mask_extra_attachments: true,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::LineVert,
                    ShaderId::LineFrag,
                    &targets.colors(msaa, subsurface),
                )
            })
        };
        Self {
            pipelines: [
                [pipeline(false, false), pipeline(false, true)],
                [pipeline(true, false), pipeline(true, true)],
            ],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &Device,
        builder: &mut Encoder,
        lines: &[DebugLine],
        view: &FrameView,
        extent: [u32; 2],
        subsurface: bool,
        msaa: bool,
    ) {
        if lines.is_empty() {
            return;
        }
        let Device::Vulkan(driver) = device;

        // Whichever shape the executor opened the rendering instance with.
        let pipeline = self.pipelines[msaa as usize][subsurface as usize];

        // Flatten each segment into its two endpoints for `LineList` topology.
        let mut bytes = Vec::with_capacity(lines.len() * 2 * LineVertex::STREAM.stride as usize);
        for line in lines {
            for end in [line.from, line.to] {
                for float in end.to_array().iter().chain(&line.color) {
                    bytes.extend_from_slice(&float.to_ne_bytes());
                }
            }
        }
        let buffer = driver.upload_vertices(&bytes);

        // The frame's jittered view-projection, so a line lands on the same
        // subpixel as the geometry it is drawn against.
        let mut words = [0u32; 16];
        for (word, float) in words
            .iter_mut()
            .zip(view.view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = float.to_bits();
        }

        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        builder.bind_pipeline(pipeline);
        builder.push_constants(&words);
        builder.bind_vertex_buffer(buffer);
        builder.draw(lines.len() as u32 * 2, 1, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_stream_matches_its_derived_layout() {
        let derived = LineVertex::per_vertex();
        assert_eq!(LineVertex::STREAM.stride, derived.stride);
        for attribute in LineVertex::STREAM.attributes {
            assert_eq!(
                derived.members[attribute.name].offset, attribute.offset,
                "{}",
                attribute.name
            );
        }
        assert_eq!(LineVertex::STREAM.attributes.len(), derived.members.len());
    }
}
