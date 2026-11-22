use crate::gfx::PositionVertex;
use crate::gfx::format::SampleCount;
use crate::gfx::graph::ImageUsage;
use crate::gfx::images::ImageEntry;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferId, CullMode, DepthCompare,
    DepthTest, Device, Encoder, GraphicsPipelineDesc, PipelineId, SamplerId, TextureDesc,
    TextureDimension, TextureViewId, Topology, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::shadow_views::{AtlasRect, Layer, ShadowPlan};

use std::ops::Range;

use super::cull::{CullFrame, Draws};
use glam::Mat4;

/// The depth format of the shadow atlas, and deliberately not the camera's.
///
/// Sixteen bits suffice here and not for the camera: a sun view is orthographic,
/// so its depth is linear over `2 * radius + pullback` — about 200 m at the
/// default distance, a 3 mm step — and a punctual face spends its precision near
/// the light, where its contacts are. The saving is half of every clear, write
/// and filtered read the shadow passes make, and half their memory.
///
/// The bias settings inherit one consequence: a constant depth bias is measured
/// in the format's smallest step, which for UNORM is a fixed 2^-16 of the map's
/// range, where for D32 it was relative to the biased depth and nearly nothing.
pub(super) const SHADOW_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::D16Unorm;

/// Where `shadow.frag` declares the material texture array, in the masked
/// variant that alpha-tests; the plain one declares no such set. Binding 0 of
/// it, which is what [`VkContext::mark_texture_array_partial`] assumes.
const TEXTURE_SET: usize = 2;

pub struct ShadowPass {
    pipeline: PipelineId,
    /// The caster pipeline for a `Masked` material: two-sided, and running the
    /// fragment shader's alpha test.
    ///
    /// Without it a cutout casts the shadow of the quad its texels are painted
    /// on, which is the single most visible way foliage goes wrong — the
    /// silhouette is right in the frame and a rectangle on the ground. It needs
    /// the material table and the texture array that the depth-only pipeline
    /// has no use for, so its layout has three sets where the plain one has one;
    /// set 0 is identical in both, which is what lets the object set stay bound
    /// across a switch.
    masked_pipeline: PipelineId,
    /// Writes the far plane over rectangles of an atlas: what clears a tile
    /// before it is redrawn. See `shadow_rect.vert` for why it is a draw.
    fill_pipeline: PipelineId,
    /// Writes the cached layer's depth over rectangles of the live atlas.
    restore_pipeline: PipelineId,
    /// A 1x1 depth texture of 1.0, bound as the atlas when there is none, so
    /// the forward shader samples "fully lit" with no second code path. Not a
    /// graph resource: with nothing casting the graph has no atlas at all, so
    /// there is nothing for the forward pass to declare a read of.
    lit_atlas_view: TextureViewId,
    /// The two atlases, kept here rather than by the graph because a view that
    /// is not redrawn keeps its tile from an earlier frame and a graph
    /// transient starts each frame `Undefined`. Allocated to the description
    /// `frame::shadow_atlas_desc` declares, and replaced only when that
    /// changes.
    atlases: Option<Atlases>,
}

impl ShadowPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |vertex: ShaderId, fragment: ShaderId, cull, rects: bool| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: vertex.into(),
                fragment: fragment.into(),
                // No color attachments to blend into.
                color_formats: &[],
                depth_format: Some(SHADOW_FORMAT),
                vertex_streams: if rects {
                    &[]
                } else {
                    &[PositionVertex::STREAM]
                },
                cull,
                depth_test: Some(DepthTest {
                    write: true,
                    // A rectangle draw replaces what is there, whatever it is.
                    compare: if rects {
                        DepthCompare::Always
                    } else {
                        DepthCompare::Less
                    },
                }),
                // Only the masked variant samples anything — the plain one
                // writes depth and nothing else — so on that pipeline this finds
                // no set to mark and does nothing.
                partial_texture_set: (!rects).then_some(TEXTURE_SET as u32),
                dynamic_depth_bias: !rects,
                samples: SampleCount::Sample1,
                alpha_to_coverage: false,
                immutable_samplers: &[],
                blend: Blend::None,
                topology: Topology::Triangles,
                mask_extra_attachments: false,
                attachment_blends: &[],
                layout_from: None,
                layout_prefix_from: None,
            })
        };
        // Back-face culling matches the forward pass, which is only correct
        // because the sun's projections apply the same Y flip the camera's
        // does — otherwise the winding is mirrored here and this would cull
        // front faces instead.
        let plain = pipeline(
            ShaderId::ShadowVert,
            ShaderId::ShadowFrag,
            CullMode::Back,
            false,
        );
        let masked_pipeline = pipeline(
            ShaderId::ShadowVertMasked,
            ShaderId::ShadowFragMasked,
            CullMode::None,
            false,
        );
        let fill_pipeline = pipeline(
            ShaderId::ShadowRectVert,
            ShaderId::ShadowFrag,
            CullMode::None,
            true,
        );
        let restore_pipeline = pipeline(
            ShaderId::ShadowRectVertRestore,
            ShaderId::ShadowRestoreFrag,
            CullMode::None,
            true,
        );
        Self {
            pipeline: plain,
            masked_pipeline,
            fill_pipeline,
            restore_pipeline,
            lit_atlas_view: build_lit_depth(device),
            atlases: None,
        }
    }

    /// The live and cached atlases at `resolution`, and whether they were just
    /// (re)allocated — in which case every view they held is gone and the
    /// caller's planner has to forget them.
    ///
    /// New atlases are cleared to the far plane and left
    /// `ShaderReadOnlyOptimal`, so the layout the graph declares them entering
    /// in is true from their first frame rather than only from their second.
    pub(super) fn atlases(
        &mut self,
        device: &Device,
        resolution: u32,
    ) -> (ImageEntry, ImageEntry, bool) {
        if let Some(atlases) = &self.atlases
            && atlases.resolution == resolution
        {
            return (atlases.live.clone(), atlases.cached.clone(), false);
        }
        if let Some(old) = self.atlases.take() {
            device.destroy_texture(old.live.whole);
            device.destroy_texture(old.cached.whole);
        }
        let allocate = || {
            let desc = super::frame::shadow_atlas_desc(resolution);
            let whole = device.create_texture(&TextureDesc {
                format: desc.format,
                dimension: TextureDimension::D2,
                extent: [resolution, resolution, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_DST,
                memoryless: false,
            });
            device.fill_depth(whole, crate::gfx::depth::SHADOW_CLEAR);
            ImageEntry {
                whole,
                levels: Vec::new(),
                layers: Vec::new(),
                extent: [resolution, resolution],
                mip_levels: 1,
            }
        };
        let atlases = Atlases {
            live: allocate(),
            cached: allocate(),
            resolution,
        };
        let entries = (atlases.live.clone(), atlases.cached.clone(), true);
        self.atlases = Some(atlases);
        entries
    }

    /// Upload the rectangles a frame clears and restores, and build the sets
    /// their two pipelines draw them through. `None` when it has none.
    pub(super) fn upload_rects(&self, device: &Device, plan: &ShadowPlan) -> Option<ShadowRects> {
        let atlases = self.atlases.as_ref()?;
        let count = plan.static_fills.len() + plan.live_fills.len() + plan.restores.len();
        if count == 0 {
            return None;
        }
        let fill = |rect: &AtlasRect| GpuRect {
            rect: corners(rect),
            tile: [0; 4],
            wrap: [0; 4],
        };
        let mut rects: Vec<GpuRect> = Vec::with_capacity(count);
        rects.extend(plan.static_fills.iter().map(fill));
        rects.extend(plan.live_fills.iter().map(fill));
        rects.extend(plan.restores.iter().map(|restore| GpuRect {
            rect: corners(&restore.rect),
            tile: [
                restore.tile.offset[0] as i32,
                restore.tile.offset[1] as i32,
                restore.tile.size as i32,
                0,
            ],
            wrap: [restore.wrap[0] as i32, restore.wrap[1] as i32, 0, 0],
        }));
        let buffer = device.upload_block(BlockKind::Storage, bytemuck::cast_slice(&rects));
        let rects = Binding {
            binding: 0,
            resource: BindingResource::StorageBuffer(buffer),
        };
        let static_fills = plan.static_fills.len() as u32;
        let live_fills = static_fills + plan.live_fills.len() as u32;
        Some(ShadowRects {
            fill_set: device.create_transient_bind_group(self.fill_pipeline, 0, &[rects]),
            restore_set: device.create_transient_bind_group(
                self.restore_pipeline,
                0,
                &[
                    rects,
                    Binding {
                        binding: 1,
                        resource: BindingResource::Texture(atlases.cached.whole),
                    },
                ],
            ),
            static_fills: 0..static_fills,
            live_fills: static_fills..live_fills,
            restores: live_fills..count as u32,
            atlas: atlases.resolution,
        })
    }

    /// The "everything is lit" atlas, bound when nothing casts.
    pub(super) fn lit_atlas_view(&self) -> TextureViewId {
        self.lit_atlas_view
    }

    /// The per-object descriptor set every view binds.
    ///
    /// Built once per frame and handed to each view, rather than once per
    /// view: all of them read the same buffer through the same layout.
    pub(super) fn build_object_set(
        &self,
        device: &Device,
        rows: BufferId,
        indices: BufferId,
    ) -> BindGroupId {
        device.create_transient_bind_group(
            self.pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(rows),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(indices),
                },
            ],
        )
    }

    pub(super) fn build_material_set(&self, device: &Device, materials: BufferId) -> BindGroupId {
        device.create_persistent_bind_group(
            self.masked_pipeline,
            1,
            &[Binding {
                binding: 0,
                resource: BindingResource::StorageBuffer(materials),
            }],
        )
    }

    pub(super) fn build_texture_set(
        &self,
        device: &Device,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> BindGroupId {
        device.create_texture_array_group(
            self.masked_pipeline,
            TEXTURE_SET as u32,
            textures,
            sampler,
        )
    }

    /// Record one layer's share of the frame's shadow work into its atlas.
    ///
    /// One render pass for every view in the layer, which is the whole reason
    /// the atlas exists: a tile is a viewport, so four sun views and the faces
    /// of eight lights are one barrier rather than fifty-two. The attachment
    /// loads what it held (see `rendering::rendering_info`): a view the plan
    /// does not touch is sampled from the texels it already has.
    ///
    /// First the rectangles — tiles cleared before they are redrawn, and in
    /// the live layer the texels put back to the cached layer's — and then the
    /// caster draws, each one of the cull dispatch's own views, numbered from
    /// `first_view` in the plan's order.
    pub(super) fn record(
        &self,
        builder: &mut Encoder,
        plan: &ShadowPlan,
        layer: Layer,
        rects: Option<&ShadowRects>,
        (frame, commands, first_view): (&CullFrame, BufferId, u32),
        sets: &CasterSets,
    ) {
        if let Some(rects) = rects {
            let whole = AtlasRect {
                min: [0, 0],
                max: [rects.atlas; 2],
            };
            let mut aimed = false;
            let mut draw = |builder: &mut Encoder, pipeline, set, range: &Range<u32>| {
                if range.is_empty() {
                    return;
                }
                if !std::mem::replace(&mut aimed, true) {
                    self.set_target(builder, [0, 0], rects.atlas, whole);
                }
                builder.bind_pipeline(pipeline);
                builder.bind_group(0, set);
                builder.push_constants(&[(rects.atlas as f32).to_bits()]);
                builder.draw(6, range.len() as u32, 0, range.start);
            };
            match layer {
                Layer::Static => {
                    draw(
                        builder,
                        self.fill_pipeline,
                        rects.fill_set,
                        &rects.static_fills,
                    );
                }
                Layer::Live => {
                    draw(
                        builder,
                        self.fill_pipeline,
                        rects.fill_set,
                        &rects.live_fills,
                    );
                    draw(
                        builder,
                        self.restore_pipeline,
                        rects.restore_set,
                        &rects.restores,
                    );
                }
            }
        }

        // Position alone: a depth-only pass reads no surface stream.
        builder.bind_geometry(true);

        // One guard for every view rather than one per call into `draw`. The
        // two pipelines and the three descriptor sets are the same for the
        // whole atlas, so past the first view nothing below rebinds either.
        let mut bound = Bound::default();
        let mut biased = None;
        for (index, draw) in plan.draws.iter().enumerate() {
            if draw.layer != layer {
                continue;
            }
            // Dynamic so the editor's bias sliders tune acne live instead of
            // rebuilding the pipeline on every drag, and so views with
            // different biases can share the pass: the sun's and the lamps'
            // are tuned apart, because an orthographic depth unit is the same
            // distance everywhere and a perspective one is millimetres at the
            // light and metres at its range, and a light may carry its own.
            // `clamp` stays 0.0: a nonzero one needs the `depth_bias_clamp`
            // device feature. Set after the rectangles, whose pipelines have
            // no bias of their own and leave the dynamic one undefined.
            let bias = plan.views.views[draw.view].bias;
            if biased != Some(bias) {
                builder.set_depth_bias(bias[0], 0.0, bias[1]);
                biased = Some(bias);
            }
            let draws = Draws {
                frame,
                view: first_view + index as u32,
                commands,
            };
            self.set_target(builder, draw.tile.offset, draw.tile.size, draw.scissor);
            self.draw(builder, draws, draw.view_proj, sets, &mut bound);
        }
    }

    /// Aim the rasteriser at one tile, and let it write `scissor` of it.
    ///
    /// The scissor is not redundant with the viewport. A viewport confines where
    /// NDC *lands*, and an implementation is allowed a guard band around the
    /// clip volume — so without a scissor a triangle straddling a tile's edge
    /// may write a fragment into the neighbour, which is another light's depth.
    /// And a strip of a sun level is drawn through its whole tile's viewport
    /// with the scissor cut down to the strip.
    fn set_target(&self, builder: &mut Encoder, offset: [u32; 2], size: u32, scissor: AtlasRect) {
        builder.set_viewport(Viewport {
            origin: [offset[0] as f32, offset[1] as f32],
            extent: [size as f32, size as f32],
            depth_range: [0.0, 1.0],
        });
        builder.set_scissor(
            scissor.min,
            [
                scissor.max[0] - scissor.min[0],
                scissor.max[1] - scissor.min[1],
            ],
        );
    }

    /// The dispatch's two multi-draws over this view's commands, one per
    /// pipeline variant. A view culled to nothing still records them, with every
    /// instance count zero.
    fn draw(
        &self,
        builder: &mut Encoder,
        casters: Draws<'_>,
        view_proj: Mat4,
        sets: &CasterSets,
        bound: &mut Bound,
    ) {
        for region in casters.regions_by_mask() {
            self.bind(builder, sets, bound, region.variant.masked, view_proj);
            builder.draw_indexed_indirect(region.commands, region.range);
        }
    }

    /// Put the pipeline this draw needs, and the light's matrix, in front of it.
    ///
    /// `bound` is the caller's, so a run that wants what the last one bound
    /// binds nothing — which is what keeps the atlas from rebinding a pipeline
    /// and three descriptor sets on each of its nineteen tiles.
    fn bind(
        &self,
        builder: &mut Encoder,
        sets: &CasterSets,
        bound: &mut Bound,
        wants_masked: bool,
        view_proj: Mat4,
    ) {
        let pipeline = if wants_masked {
            self.masked_pipeline
        } else {
            self.pipeline
        };
        if bound.needs_pipeline(wants_masked) {
            builder.bind_pipeline(pipeline);
            for (set, group) in sets.for_pipeline(wants_masked).into_iter().enumerate() {
                builder.bind_group(set as u32, group);
            }
        }
        // Per-pass push constants: 64 bytes, comfortably inside the 128-byte
        // guaranteed `maxPushConstantsSize`. The light's matrix rides here
        // rather than in a uniform buffer because it changes once per pass, not
        // once per draw.
        //
        // One block for both pipeline variants. It was two while the cutout
        // variant also named a material row to alpha-test against — a member the
        // plain variant's reflected range did not have, so writing it there
        // would have run past the end. The material now travels with the
        // instance, which is what leaves the two ranges the same shape. See
        // [`instances`](super::instances).
        let mut words = [0u32; 16];
        for (word, value) in words
            .iter_mut()
            .zip(view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = value.to_bits();
        }
        builder.push_constants(&words);
    }
}

/// Which pipeline the rasteriser already has, so the draw loop binds one only
/// when the run's material actually changes it.
///
/// Held by the caller rather than by [`ShadowPass::draw`], and that is the whole
/// point: the atlas calls into that loop once per view, and a guard scoped to
/// the call rebound a pipeline and three descriptor sets on every tile. Spans
/// one render pass instance and never more — each `record` starts a fresh one,
/// so nothing here assumes what a render pass boundary preserves.
#[derive(Default)]
struct Bound(Option<bool>);

impl Bound {
    /// Whether the pipeline `masked` selects has to be bound, recording that it
    /// now is.
    fn needs_pipeline(&mut self, masked: bool) -> bool {
        self.0.replace(masked) != Some(masked)
    }
}

/// What a caster draw binds, in the two shapes the two pipelines want.
///
/// One struct rather than three parameters threaded through `record` and
/// `draw`, and it holds the *cutout* sets as an `Option`
/// because a frame is entitled to have no material table cached yet — nothing
/// here forces the alpha-testing pipeline into existence for a scene with no
/// foliage in it.
pub(super) struct CasterSets {
    pub objects: BindGroupId,
    pub materials: BindGroupId,
    pub textures: BindGroupId,
}

impl CasterSets {
    fn for_pipeline(&self, masked: bool) -> Vec<BindGroupId> {
        if masked {
            vec![self.objects, self.materials, self.textures]
        } else {
            vec![self.objects]
        }
    }
}

struct Atlases {
    live: ImageEntry,
    cached: ImageEntry,
    resolution: u32,
}

/// One rectangle as `shadow_rect.vert` reads it. Layout-frozen against `Rect`
/// there.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct GpuRect {
    rect: [f32; 4],
    tile: [i32; 4],
    wrap: [i32; 4],
}

fn corners(rect: &AtlasRect) -> [f32; 4] {
    [
        rect.min[0] as f32,
        rect.min[1] as f32,
        rect.max[0] as f32,
        rect.max[1] as f32,
    ]
}

/// A frame's rectangles on the device: which instances of the one buffer are
/// the cached layer's clears, the live layer's, and the live layer's restores.
pub(super) struct ShadowRects {
    fill_set: BindGroupId,
    restore_set: BindGroupId,
    static_fills: Range<u32>,
    live_fills: Range<u32>,
    restores: Range<u32>,
    /// Edge of the atlas, which is what turns a texel into clip space.
    atlas: u32,
}

/// A 1x1 depth texture cleared to 1.0 — "nothing is nearer than the far
/// plane", which every comparison reads as lit.
fn build_lit_depth(device: &Device) -> TextureViewId {
    let texture = device.create_texture(&TextureDesc {
        format: SHADOW_FORMAT,
        dimension: TextureDimension::D2,
        extent: [1, 1, 1],
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::Sample1,
        usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
        memoryless: false,
    });
    device.fill_depth(texture, crate::gfx::depth::SHADOW_CLEAR);
    texture
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The saving this guard exists for. The atlas enters the draw loop once per
    /// face — nineteen times on a demo frame — and every one of those used to
    /// rebind a pipeline and three descriptor sets that were already bound,
    /// because the guard was scoped to the call rather than to the pass.
    #[test]
    fn a_pipeline_already_bound_is_not_bound_again() {
        let mut bound = Bound::default();
        assert!(bound.needs_pipeline(false));
        assert!(!bound.needs_pipeline(false));
        assert!(!bound.needs_pipeline(false));
    }

    /// And what it must not cost: a caster list that mixes foliage with
    /// everything else still switches pipelines mid-tile, so the guard has to
    /// notice a change in both directions rather than latch on the first bind.
    #[test]
    fn a_changed_pipeline_is_bound_again() {
        let mut bound = Bound::default();
        assert!(bound.needs_pipeline(false));
        assert!(bound.needs_pipeline(true));
        assert!(bound.needs_pipeline(false));
        assert!(!bound.needs_pipeline(false));
    }

    /// A fresh pass binds its first pipeline, whichever it is. Each render pass
    /// instance starts one of these rather than inheriting it, which is what
    /// makes the unset default the correct one.
    #[test]
    fn a_fresh_pass_binds_its_first_pipeline() {
        assert!(Bound::default().needs_pipeline(false));
        assert!(Bound::default().needs_pipeline(true));
    }
}
