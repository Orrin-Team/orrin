//! The selection outline: two graph nodes, after the tonemap and before the
//! gizmo.
//!
//! `outline_mask` draws every selected or hovered mesh, flat and with no depth
//! test, into a two-channel mask: red where a selected one covers, green where
//! the hovered one does. `outline` then reads the mask and paints a band just
//! outside each shape onto the finished frame. With no depth test the outline
//! shows through whatever stands in front of the selection, which is what
//! keeps a selected thing findable behind a wall.
//!
//! A mask of its own rather than the scene's stencil or depth: the outline is
//! the editor's and is drawn after the frame is finished, by which point the
//! scene's targets are resolved, aliased or gone, and a frame without the
//! editor declares neither node.

use glam::Mat4;

use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Binding, BindingResource, Blend, CullMode, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;

use super::forward::GpuMesh;
use crate::gfx::PositionVertex;
use crate::scene::MeshHandle;

/// Two channels: selected and hovered.
pub const OUTLINE_MASK_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rg8Unorm;

/// The selection's colour, and the hover's, in sRGB.
const SELECTED: [f32; 4] = [1.0, 0.62, 0.16, 1.0];
const HOVERED: [f32; 4] = [0.55, 0.78, 1.0, 0.85];

/// One mesh to outline, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineItem {
    pub mesh: MeshHandle,
    pub model: Mat4,
    /// Selected rather than only under the pointer.
    pub selected: bool,
}

pub struct OutlinePass {
    mask: PipelineId,
    outline: PipelineId,
    sampler: SamplerId,
    linear_output: bool,
}

impl OutlinePass {
    pub fn new(device: &Device, swapchain_format: Format) -> Self {
        Self {
            // Both faces: a mask of coverage, and a mesh seen from inside still
            // covers what it covers.
            mask: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[PositionVertex::STREAM],
                cull: CullMode::None,
                blend: Blend::Max,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::OutlineMaskVert,
                    ShaderId::OutlineMaskFrag,
                    &[OUTLINE_MASK_FORMAT],
                )
            }),
            outline: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                blend: Blend::Alpha,
                ..GraphicsPipelineDesc::fullscreen(
                    ShaderId::FullscreenVert,
                    ShaderId::OutlineFrag,
                    &[swapchain_format],
                )
            }),
            sampler: device.create_sampler(SamplerDesc {
                linear: false,
                mipmap_linear: false,
                repeat: true,
                compare: false,
                anisotropy: None,
                max_lod: Some(0.0),
            }),
            linear_output: swapchain_format.is_srgb(),
        }
    }

    pub fn record_mask(
        &self,
        builder: &mut Encoder,
        items: &[OutlineItem],
        meshes: &[GpuMesh],
        view_proj: Mat4,
        extent: [u32; 2],
    ) {
        if items.is_empty() {
            return;
        }
        builder.set_viewport(viewport(extent));
        builder.bind_pipeline(self.mask);
        builder.bind_geometry(true);
        for item in items {
            let Some(mesh) = meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let value = if item.selected {
                [1.0, 0.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0, 0.0]
            };
            let mvp = (view_proj * item.model).to_cols_array_2d();
            let mut words = [0u32; 20];
            for (word, float) in words.iter_mut().zip(mvp.iter().flatten().chain(&value)) {
                *word = float.to_bits();
            }
            builder.push_constants(&words);
            builder.draw_indexed(
                mesh.span.index_count,
                1,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                0,
            );
        }
    }

    pub fn record_outline(
        &self,
        device: &Device,
        builder: &mut Encoder,
        mask: TextureViewId,
        extent: [u32; 2],
        any: bool,
    ) {
        if !any {
            return;
        }
        let set = device.create_bind_group(
            self.outline,
            0,
            &[Binding {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    view: mask,
                    sampler: self.sampler,
                },
            }],
        );
        builder.set_viewport(viewport(extent));
        builder.bind_pipeline(self.outline);
        builder.bind_group(0, set);
        let mut words = [0u32; 9];
        for (word, float) in words.iter_mut().zip(SELECTED.iter().chain(&HOVERED)) {
            *word = float.to_bits();
        }
        words[8] = self.linear_output as u32;
        builder.push_constants(&words);
        builder.draw(3, 1, 0, 0);
    }
}

fn viewport(extent: [u32; 2]) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: [0.0, 1.0],
    }
}
