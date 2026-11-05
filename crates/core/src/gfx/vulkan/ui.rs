//! The in-game UI's pass: `gfx::ui`'s quads, over the tonemapped frame.
//!
//! A node of its own straight after the tonemap, for the gizmo's reasons and
//! one more: the game's colours are the ones it asked for, not the scene's
//! exposure of them, and every mark the editor makes — outline, handles, its
//! own UI — draws after it and so stays on top.
//!
//! One instanced draw per [`UiBatch`]: six vertices, one instance per quad.
//! A batch's source is a push constant, so the material texture array is
//! indexed uniformly and the atlas is a second set rather than a slot in it,
//! which keeps the table's [`MAX_TEXTURES`](crate::gfx::MAX_TEXTURES) for
//! materials.
//!
//! The atlas is copied at the top of the frame's command buffer, ahead of every
//! pass, because the draw itself is inside a render pass where nothing may be
//! copied. Only rows the atlas has ever written are sent, and only in a frame
//! whose atlas changed — after the first few frames of a HUD, none.

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, Device, Encoder, GraphicsPipelineDesc,
    PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension, TextureViewId,
    VertexAttribute, VertexFormat, VertexStream, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::ui::{UiFrame, UiSource};

/// The texture index that names the atlas rather than a material texture.
/// `ATLAS` in `ui.frag`.
const ATLAS: u32 = u32::MAX;

const ATLAS_FORMAT: Format = Format::R8Unorm;

/// [`UiQuad`] as the pipeline reads it, one per instance.
const STREAM: VertexStream = VertexStream {
    stride: 36,
    per_instance: true,
    attributes: &[
        VertexAttribute {
            name: "rect",
            format: VertexFormat::Float4,
            offset: 0,
        },
        VertexAttribute {
            name: "uv",
            format: VertexFormat::Float4,
            offset: 16,
        },
        VertexAttribute {
            name: "color",
            format: VertexFormat::Unorm8x4,
            offset: 32,
        },
    ],
};

/// The atlas as last uploaded.
struct Atlas {
    view: TextureViewId,
    set: BindGroupId,
    extent: [u32; 2],
    generation: u64,
}

pub(super) struct UiPass {
    pipeline: PipelineId,
    linear_output: bool,
    sampler: SamplerId,
    atlas: Option<Atlas>,
    /// The material table as set 0 of this pipeline, until a texture changes.
    textures: Option<BindGroupId>,
}

impl UiPass {
    pub(super) fn new(device: &Device, format: Format) -> Self {
        Self {
            pipeline: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex_streams: &[STREAM],
                blend: Blend::Alpha,
                partial_texture_set: Some(0),
                ..GraphicsPipelineDesc::fullscreen(ShaderId::UiVert, ShaderId::UiFrag, &[format])
            }),
            linear_output: format.is_srgb(),
            // Mipmapped for an icon drawn smaller than its file; clamped so a
            // filtered edge does not wrap round to the opposite one.
            sampler: device.create_sampler(SamplerDesc {
                linear: true,
                mipmap_linear: true,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: None,
            }),
            atlas: None,
            textures: None,
        }
    }

    /// Forget the material table's set: a texture was loaded or replaced.
    pub(super) fn invalidate_textures(&mut self, device: &Device) {
        if let Some(set) = self.textures.take() {
            device.destroy_bind_group(set);
        }
    }

    /// Bring the atlas and the texture table up to date for `frame`. Recorded
    /// at the top of the frame, outside any render pass.
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        encoder: &mut Encoder,
        frame: UiFrame<'_>,
        textures: &[TextureViewId],
    ) {
        if frame.is_empty() {
            return;
        }
        if self.textures.is_none() {
            self.textures =
                Some(device.create_texture_array_group(self.pipeline, 0, textures, self.sampler));
        }

        let source = frame.atlas;
        if self
            .atlas
            .as_ref()
            .is_some_and(|atlas| atlas.generation == source.generation())
        {
            return;
        }
        let extent = source.extent();
        let fresh = self
            .atlas
            .as_ref()
            .is_none_or(|atlas| atlas.extent != extent);
        if fresh {
            if let Some(old) = self.atlas.take() {
                device.destroy_bind_group(old.set);
                device.destroy_texture(old.view);
            }
            let view = device.create_texture(&TextureDesc {
                format: ATLAS_FORMAT,
                dimension: TextureDimension::D2,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                memoryless: false,
            });
            let set = device.create_persistent_bind_group(
                self.pipeline,
                1,
                &[Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view,
                        sampler: self.sampler,
                    },
                }],
            );
            self.atlas = Some(Atlas {
                view,
                set,
                extent,
                generation: 0,
            });
        }
        let atlas = self.atlas.as_mut().expect("created above");
        let rows = source.used_rows();
        let texels = (extent[0] * rows) as usize;
        let staging = device.upload_block(BlockKind::Staging, &source.pixels()[..texels]);
        encoder.write_texture(atlas.view, [0, 0], [extent[0], rows], staging, fresh);
        atlas.generation = source.generation();
    }

    /// Draw `frame` into the swapchain image the executor opened.
    pub(super) fn record(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        frame: UiFrame<'_>,
        extent: [u32; 2],
    ) {
        let (Some(atlas), Some(textures)) = (&self.atlas, self.textures) else {
            return;
        };
        if frame.is_empty() {
            return;
        }
        let quads = device.upload_vertices(bytemuck::cast_slice(frame.quads));
        let size = [extent[0] as f32, extent[1] as f32];
        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: size,
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.pipeline);
        encoder.bind_group(0, textures);
        encoder.bind_group(1, atlas.set);
        encoder.bind_vertex_buffer(quads);
        for batch in frame.batches {
            let index = match batch.source {
                UiSource::Atlas => ATLAS,
                UiSource::Texture(handle) => handle.0,
            };
            encoder.push_constants(&[
                size[0].to_bits(),
                size[1].to_bits(),
                self.linear_output as u32,
                index,
            ]);
            encoder.draw(6, batch.quads.len() as u32, 0, batch.quads.start);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;
    use crate::gfx::ui::UiQuad;

    #[test]
    fn the_instance_stream_matches_the_quad() {
        assert_eq!(STREAM.stride as usize, size_of::<UiQuad>());
        let offsets = [
            ("rect", offset_of!(UiQuad, rect)),
            ("uv", offset_of!(UiQuad, uv)),
            ("color", offset_of!(UiQuad, color)),
        ];
        assert_eq!(STREAM.attributes.len(), offsets.len());
        for (attribute, (name, offset)) in STREAM.attributes.iter().zip(offsets) {
            assert_eq!((attribute.name, attribute.offset as usize), (name, offset));
        }
    }
}
