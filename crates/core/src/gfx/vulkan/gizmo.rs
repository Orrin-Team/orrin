//! The transform gizmo's pass: the editor's strokes, over the finished frame.
//!
//! A node of its own after the tonemap, writing the swapchain image, rather
//! than lines in the forward pass the way debug lines are. Drawn there, a
//! handle would be depth-tested against the scene and lose to whatever stands
//! in front of the selection, and its colour would go through exposure and the
//! tonemap curve. Here it is neither, and it sits under the editor's own UI
//! because the overlay is the node after it.
//!
//! The strokes arrive already in screen space (see `editor::gizmo`), which is
//! what keeps the gizmo one size on screen at any distance, and is also what
//! the editor hit-tests the pointer against — the shape grabbed is the shape
//! drawn.

use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Blend, Device, Encoder, GraphicsPipelineDesc, PipelineId, VertexAttribute, VertexFormat,
    VertexStream, Viewport,
};
use crate::gfx::shaders::ShaderId;

/// One corner of a stroke's quad, in physical pixels from the top left.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    test,
    derive(
        vulkano::buffer::BufferContents,
        vulkano::pipeline::graphics::vertex_input::Vertex
    )
)]
#[repr(C)]
pub struct GizmoVertex {
    #[cfg_attr(test, format(R32G32_SFLOAT))]
    pub position: [f32; 2],
    /// Distance across the stroke from its centre line, and its half width,
    /// in pixels. What the fragment shader antialiases the edge from.
    #[cfg_attr(test, format(R32G32_SFLOAT))]
    pub edge: [f32; 2],
    /// sRGB, unpremultiplied.
    #[cfg_attr(test, format(R8G8B8A8_UNORM))]
    pub color: [u8; 4],
}

impl GizmoVertex {
    /// This stream as the RHI describes it; a test holds it against the layout
    /// the `Vertex` derive produces.
    pub const STREAM: VertexStream = VertexStream {
        stride: 20,
        per_instance: false,
        attributes: &[
            VertexAttribute {
                name: "position",
                format: VertexFormat::Float2,
                offset: 0,
            },
            VertexAttribute {
                name: "edge",
                format: VertexFormat::Float2,
                offset: 8,
            },
            VertexAttribute {
                name: "color",
                format: VertexFormat::Unorm8x4,
                offset: 16,
            },
        ],
    };
}

pub struct GizmoPass {
    pipeline: PipelineId,
    linear_output: bool,
}

impl GizmoPass {
    pub fn new(device: &Device, format: Format) -> Self {
        Self {
            pipeline: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[GizmoVertex::STREAM],
                blend: Blend::Alpha,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::GizmoVert,
                    ShaderId::GizmoFrag,
                    &[format],
                )
            }),
            linear_output: format.is_srgb(),
        }
    }

    /// Draw `vertices`, a triangle list, into the swapchain image the executor
    /// opened. Nothing is recorded for an empty list — no selection, or a tool
    /// with no handles.
    pub fn record(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        vertices: &[GizmoVertex],
        extent: [u32; 2],
    ) {
        if vertices.is_empty() {
            return;
        }
        // Packed field by field to the stream's declared layout (20 bytes a
        // vertex), which is what `the_gizmo_stream_matches_its_derived_layout`
        // holds against the struct's own.
        let mut bytes = Vec::with_capacity(vertices.len() * GizmoVertex::STREAM.stride as usize);
        for vertex in vertices {
            for float in vertex.position.iter().chain(&vertex.edge) {
                bytes.extend_from_slice(&float.to_ne_bytes());
            }
            bytes.extend_from_slice(&vertex.color);
        }
        let buffer = device.upload_vertices(&bytes);

        let size = [extent[0] as f32, extent[1] as f32];
        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: size,
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.pipeline);
        encoder.push_constants(&[
            size[0].to_bits(),
            size[1].to_bits(),
            self.linear_output as u32,
        ]);
        encoder.bind_vertex_buffer(buffer);
        encoder.draw(vertices.len() as u32, 1, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vulkano::pipeline::graphics::vertex_input::Vertex as _;

    #[test]
    fn the_gizmo_stream_matches_its_derived_layout() {
        let derived = GizmoVertex::per_vertex();
        assert_eq!(GizmoVertex::STREAM.stride, derived.stride);
        for attribute in GizmoVertex::STREAM.attributes {
            let member = &derived.members[attribute.name];
            assert_eq!(member.offset, attribute.offset, "{}", attribute.name);
        }
        assert_eq!(GizmoVertex::STREAM.attributes.len(), derived.members.len());
    }
}
