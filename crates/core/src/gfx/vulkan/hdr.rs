use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;

/// Offscreen colour format the frame's radiance is carried in. Float, so values
/// can exceed 1.0 before tonemapping clamps them back to displayable range.
///
/// Packed into 32 bits rather than 64. Every full-resolution pass in the optical
/// chain reads one image of this and writes another, so the width of this
/// constant is most of the frame's bandwidth — halving it is worth more than the
/// arithmetic in any of those passes. At 1080p it also brings a read-one-write-one
/// pass to a 16 MB working set, which fits inside a 9070 XT's 64 MB Infinity
/// Cache; that is worth more again than the raw halving.
///
/// Three things are given up, and none of them is used by anything this carries:
/// there is no alpha channel, no negative values, and about five bits of
/// mantissa. Radiance is non-negative by definition, and in physical units with
/// EV100 metering five bits of mantissa is far below what the tonemap resolves —
/// it is what Unreal has used for scene colour for a decade. The targets that
/// *do* need one of the three use [`HDR_WIDE_FORMAT`].
///
/// Not in the Vulkan mandatory `STORAGE_IMAGE` list, though every desktop driver
/// supports it; [`VkContext`](super::context::VkContext) checks at startup rather
/// than letting a compute pass fail to bind much later.
pub const HDR_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rg11B10Float;

/// The wider colour format, for the targets [`HDR_FORMAT`] cannot carry.
///
/// Three kinds of target need it, for three different reasons:
///
/// - `msaa_hdr`, because alpha to coverage reads the first attachment's alpha,
///   and an attachment with no alpha component behaves as though it were 1.0 —
///   which would silently stop every cutout being cut. A render-pass resolve
///   also requires both images to have the same format, so `hdr_color` follows
///   it whenever the frame is multisampled.
/// - `dof_prefiltered`, `dof_near` and `dof_far`, because the depth-of-field
///   chain packs a *signed* circle of confusion and then a coverage weight into
///   alpha, and the composite divides by it.
/// - The depth-of-field and motion-blur tile images, because they carry two
///   channels of signed maxima — velocities, and near/far CoC radii.
pub const HDR_WIDE_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rgba16Float;

pub struct HdrPass {
    tonemap_pipeline: PipelineId,
    sampler: SamplerId,
    /// The exposure applied when metering is off, already carrying the
    /// compensation dial. With metering on the shader ignores it and reads the
    /// value the averaging pass wrote instead.
    pub manual_exposure: f32,
    pub auto_exposure: bool,
    /// Zero when bloom is off, which turns the shader's blend into a no-op
    /// without a second pipeline or a branch that costs anything.
    pub bloom_strength: f32,
}

impl HdrPass {
    pub fn new(device: &Device, swapchain_format: Format) -> Self {
        let tonemap_pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            vertex: ShaderId::FullscreenVert,
            fragment: ShaderId::TonemapFrag,
            color_formats: &[swapchain_format],
            depth_format: None,
        });
        let sampler = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(1.0),
        });

        Self {
            tonemap_pipeline,
            sampler,
            manual_exposure: 1.0,
            auto_exposure: true,
            bloom_strength: 0.0,
        }
    }

    /// `hdr_view` is the graph's resolved HDR color target, declared as this
    /// pass's `Sampled` input; `exposure` is the buffer the metering passes
    /// wrote, declared as its `StorageRead`.
    pub fn record_tonemap(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        hdr_view: TextureViewId,
        exposure: BufferId,
        bloom_view: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.tonemap_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: hdr_view,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(exposure),
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::SampledTexture {
                        view: bloom_view,
                        sampler: self.sampler,
                    },
                },
            ],
        );

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.tonemap_pipeline);
        encoder.bind_group(0, set);
        encoder.push_constants(&[
            self.manual_exposure.to_bits(),
            self.auto_exposure as u32,
            self.bloom_strength.to_bits(),
        ]);
        encoder.draw(3, 1, 0, 0);
    }
}
