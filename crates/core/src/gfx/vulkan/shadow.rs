use std::ops::Range;
use std::sync::Arc;

use glam::Mat4;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::ClearDepthStencilImageInfo;
use vulkano::command_buffer::DrawIndexedIndirectCommand;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::ClearDepthStencilValue;
use vulkano::image::sampler::Sampler;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;

use crate::geom::Frustum;
use crate::gfx::format::SampleCount;
use crate::gfx::punctual::ShadowAtlas;
use crate::gfx::rhi::{
    Blend, CullMode, DepthCompare, DepthRect, DepthTest, Device, Encoder, GraphicsPipelineDesc,
    PipelineId, Topology, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DrawList, PositionVertex};

use super::PassCtx;
use super::context::VkContext;
use super::convert::vk_format;
use super::cull::{CullFrame, Draws};
use super::instances::{GpuObject, InstanceEntry};
use super::record::{self, Recorder};
use super::swapchain::DEPTH_FORMAT;

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
    /// 1x1 array depth image cleared to 1.0, bound when shadows are off so the
    /// forward shader samples "fully lit" with no second code path. Not a graph
    /// resource: with shadows off the graph has no cascade image at all, so
    /// there is nothing for the forward pass to declare a read of.
    lit_view: Arc<ImageView>,
    /// A 1x1 non-array depth texture of 1.0, bound as the punctual atlas when
    /// there is none. The cascades' `lit_view` cannot serve: the forward shader
    /// declares the atlas as a plain `texture2D` and the cascades as a
    /// `texture2DArray`, and a view satisfies one or the other.
    lit_atlas_view: Arc<ImageView>,
    pub constant_bias: f32,
    pub slope_bias: f32,
    /// The punctual maps get their own pair. A cascade is orthographic, so one
    /// depth unit is the same world distance everywhere in it and a constant
    /// bias means one thing; a punctual face is perspective, where the same
    /// offset is millimetres at the light and metres at its range. They are
    /// tuned apart because they cannot be tuned together.
    pub punctual_constant_bias: f32,
    pub punctual_slope_bias: f32,
}

fn layout(
    device: &Device,
    pipeline: PipelineId,
    set: usize,
) -> Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> {
    let Device::Vulkan(driver) = device;
    driver.set_layout(pipeline, set)
}

impl ShadowPass {
    pub fn new(ctx: &VkContext, device: &Device) -> Self {
        let pipeline = |vertex, fragment, cull| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex,
                fragment,
                // No color attachments to blend into.
                color_formats: &[],
                depth_format: Some(DEPTH_FORMAT),
                vertex_streams: &[PositionVertex::STREAM],
                cull,
                depth_test: Some(DepthTest {
                    write: true,
                    compare: DepthCompare::Less,
                }),
                // Only the masked variant samples anything — the plain one
                // writes depth and nothing else — so on that pipeline this finds
                // no set to mark and does nothing.
                partial_texture_set: Some(TEXTURE_SET as u32),
                dynamic_depth_bias: true,
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
        // because `fit_cascade` applies the same Y flip the camera projection
        // does — otherwise the winding is mirrored here and this would cull front
        // faces instead.
        let plain = pipeline(ShaderId::ShadowVert, ShaderId::ShadowFrag, CullMode::Back);
        let masked_pipeline = pipeline(
            ShaderId::ShadowVertMasked,
            ShaderId::ShadowFragMasked,
            CullMode::None,
        );
        Self {
            pipeline: plain,
            masked_pipeline,
            lit_view: build_lit_view(ctx),
            lit_atlas_view: build_lit_atlas_view(ctx),
            constant_bias: 1.25,
            slope_bias: 2.5,
            punctual_constant_bias: 2.0,
            punctual_slope_bias: 3.0,
        }
    }

    /// The "everything is lit" atlas, bound when nothing punctual casts.
    pub(super) fn lit_atlas_view(&self) -> Arc<ImageView> {
        self.lit_atlas_view.clone()
    }

    /// The "everything is lit" view, bound when shadows are disabled.
    #[allow(dead_code)]
    pub(super) fn lit_view(&self) -> Arc<ImageView> {
        self.lit_view.clone()
    }

    /// The per-object descriptor set every cascade binds.
    ///
    /// Built once per frame and handed to each cascade, rather than once per
    /// cascade: all of them read the same buffer through the same layout.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        rows: &Subbuffer<[GpuObject]>,
        indices: &Subbuffer<[InstanceEntry]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.pipeline, 0),
            [
                WriteDescriptorSet::buffer(0, rows.clone()),
                WriteDescriptorSet::buffer(1, indices.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// The set-1 material table and set-2 texture array the cutout pipeline
    /// alpha-tests against, over the same buffer and the same views every other
    /// pass reads. Cached by the renderer beside the prepass's, and rebuilt on
    /// the same invalidations.
    pub(super) fn build_material_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        materials: &Subbuffer<[super::forward::GpuMaterial]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.masked_pipeline, 1),
            [WriteDescriptorSet::buffer(0, materials.clone())],
            [],
        )
        .unwrap()
    }

    pub(super) fn build_texture_set(
        &self,
        ctx: &VkContext,
        device: &Device,
        textures: &[Arc<ImageView>],
        sampler: &Arc<Sampler>,
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array = (0..ctx.texture_array_len(textures.len())).map(|index| {
            textures
                .get(index)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            layout(device, self.masked_pipeline, TEXTURE_SET),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    /// Record one cascade's depth pass.
    ///
    /// `resolution` is the shadow map's, not the frame's — every other pass in
    /// the executor takes the swapchain extent, and using it here would render
    /// the cascade into a corner of its own map with nothing reporting an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &self,
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
        casters: Draws<'_>,
        view_proj: Mat4,
        resolution: u32,
        sets: &CasterSets,
    ) {
        self.set_tile(builder, [0, 0], resolution);
        // Position alone: a depth-only pass reads no surface stream. Bound here
        // rather than in `draw` because vertex bindings are command-buffer
        // state that outlives a draw, and a secondary starts with none.
        builder.bind_geometry(renderer.arena, true);
        // Dynamic so the editor's bias sliders tune acne live instead of
        // rebuilding the pipeline on every drag. `clamp` stays 0.0: a nonzero
        // one needs the `depth_bias_clamp` device feature. Set before any
        // pipeline is bound, which is fine and is the point of dynamic state: it
        // survives the pipeline switches `draw` makes between cutout runs and
        // the rest.
        builder.set_depth_bias(self.constant_bias, 0.0, self.slope_bias);

        // A cascade is one view into a map it owns outright, so its guard starts
        // and ends here.
        self.draw(
            device,
            builder,
            renderer,
            casters,
            view_proj,
            sets,
            // Already culled, and against a wider test than this one: a caster
            // outside the cascade's box can still shadow into it, which is what
            // `casts_into` extends the box along the sun to catch. Re-testing
            // here with the plain frustum would undo that and drop shadows.
            None,
            &mut Bound::default(),
        );
    }

    /// Record every face of every punctual caster into one atlas.
    ///
    /// One render pass for all of them, which is the whole reason the atlas
    /// exists: a tile is a viewport, so the sixteen-light worst case is one
    /// barrier rather than ninety-six. `casters` and `bases` are indexed like
    /// `atlas.casters` — one culled list per *light*, drawn into each of its
    /// faces, because a point light's reach is small enough that culling its six
    /// frustums apart would cost more test than it saved draw.
    #[allow(
        clippy::too_many_arguments,
        reason = "a pass records from the frame's bindings; a params struct only renames the list"
    )]
    pub(super) fn record_atlas(
        &self,
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
        atlas: &ShadowAtlas,
        casters: &[DrawList<'_>],
        bases: &[u32],
        gpu: Option<(&CullFrame, &Subbuffer<[DrawIndexedIndirectCommand]>, u32)>,
        sets: &CasterSets,
    ) {
        // Every tile this frame assigned, cleared before anything draws — and
        // *only* those, which is the point: the atlas attachment loads
        // `DontCare` (see `rendering::rendering_info`), so the 45-odd tiles a
        // typical scene leaves unassigned cost nothing rather than 67 MB of
        // blanking.
        //
        // Safe only because `atlas_shadow` in `shading.glsl` clamps every one of
        // its nine taps to half a texel inside the tile it belongs to: nothing
        // can read a texel outside an assigned tile, so whatever is left there
        // is unobservable. The cascades cannot make that promise — they
        // deliberately let a tap fall past the edge onto the sampler's white
        // border — which is why they clear and this does not.
        //
        // Over every assigned face rather than only the ones with something to
        // draw: a light whose caster list came back empty still has its tile
        // sampled, and it has to read as far, or the light would be shadowed by
        // whatever the last frame left in that tile.
        let tiles: Vec<DepthRect> = atlas
            .casters
            .iter()
            .flat_map(|caster| {
                &atlas.faces[caster.first_face..caster.first_face + caster.face_count]
            })
            .map(|face| DepthRect {
                offset: face.tile.offset,
                extent: [face.tile.size; 2],
            })
            .collect();
        if tiles.is_empty() {
            return;
        }
        builder.clear_depth(&tiles);
        builder.bind_geometry(renderer.arena, true);

        builder.set_depth_bias(self.punctual_constant_bias, 0.0, self.punctual_slope_bias);

        // One guard for every light and every face rather than one per call into
        // `draw`. The two pipelines and the three descriptor sets are the same
        // for the whole atlas, so past the first run nothing below rebinds
        // either — where a guard scoped to one face rebound both on every one of
        // the nineteen tiles a demo frame assigns.
        let mut bound = Bound::default();
        for (index, caster) in atlas.casters.iter().enumerate() {
            // A light with nothing to draw is skipped on the CPU path only. The
            // compute path has no per-light list to be empty: its faces are
            // views, and a view whose every batch culled to zero records the
            // same draws with a count of zero.
            let list = casters.get(index).copied();
            let base = bases.get(index).copied();
            if gpu.is_none() && list.is_none_or(|list| list.is_empty()) {
                continue;
            }
            let faces = caster.first_face..caster.first_face + caster.face_count;
            for (offset, face) in atlas.faces[faces].iter().enumerate() {
                let face_index = (caster.first_face + offset) as u32;
                let draws = match gpu {
                    Some((frame, commands, first_view)) => Draws::Gpu {
                        frame,
                        view: first_view + face_index,
                        commands,
                    },
                    None => Draws::Cpu {
                        list: list.expect("a CPU-culled light with no list is skipped above"),
                        base: base.expect("a caster list and its base are the same length"),
                    },
                };
                self.set_tile(builder, face.tile.offset, face.tile.size);
                // Culled per face, where the caster lists were culled per
                // *light*. A point light's six frustums partition the space its
                // sphere covers, so most of what the light reaches is behind
                // five of them: on the demo frame this is what takes the atlas
                // from 254 draws to 122.
                //
                // Sound here in a way it is not for a cascade. These frustums
                // have their apex at the light, so anything that can occlude
                // light travelling into a face is inside that face's frustum —
                // there is no equivalent of the caster standing outside a
                // cascade's box and shadowing into it.
                let frustum = Frustum::from_view_projection(face.view_proj);
                self.draw(
                    device,
                    builder,
                    renderer,
                    draws,
                    face.view_proj,
                    sets,
                    Some(&frustum),
                    &mut bound,
                );
            }
        }
    }

    /// Aim the rasteriser at one tile.
    ///
    /// The scissor is not redundant with the viewport. A viewport confines where
    /// NDC *lands*, and an implementation is allowed a guard band around the
    /// clip volume — so without a scissor a triangle straddling a tile's edge
    /// may write a fragment into the neighbour, which is another light's depth.
    fn set_tile(&self, builder: &mut Encoder, offset: [u32; 2], size: u32) {
        builder.set_viewport(Viewport {
            origin: [offset[0] as f32, offset[1] as f32],
            extent: [size as f32, size as f32],
            depth_range: [0.0, 1.0],
        });
        builder.set_scissor(offset, [size, size]);
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
        casters: Draws<'_>,
        view_proj: Mat4,
        sets: &CasterSets,
        cull: Option<&Frustum>,
        bound: &mut Bound,
    ) {
        for unit in casters.units() {
            // Only ever `Some` on the CPU path: the compute path culls each
            // face as a view of its own, so what reaches here has already been
            // tested against this very frustum.
            if let Some(frustum) = cull
                && let Some((list, run)) = &unit.run
                && !run_intersects(list, run, frustum)
            {
                continue;
            }
            let Some(mesh) = renderer.meshes.get(unit.mesh as usize) else {
                continue;
            };
            // The same flag word the forward pass and the prepass read, so all
            // three cut the cutout out along the same line. A caster list with
            // no cutout in it never touches `masked_pipeline` and records
            // exactly what it recorded before this existed.
            let wants_masked = renderer
                .materials
                .get(unit.material as usize)
                .is_some_and(super::forward::GpuMaterial::is_masked);
            self.bind(device, builder, sets, bound, wants_masked, view_proj);
            builder.draw_indexed(
                mesh.span.index_count,
                unit.instances,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                unit.object_base,
            );
        }

        // The compute path's two multi-draws over this view's commands. A face
        // culled to nothing still records them, with every instance count zero.
        for region in casters.regions() {
            self.bind(device, builder, sets, bound, region.masked, view_proj);
            let Device::Vulkan(driver) = device;
            builder.draw_indexed_indirect(driver.import_buffer(region.commands.into_bytes()));
        }
    }

    /// Put the pipeline this draw needs, and the light's matrix, in front of it.
    ///
    /// `bound` is the caller's, so a run that wants what the last one bound
    /// binds nothing — which is what keeps the atlas from rebinding a pipeline
    /// and three descriptor sets on each of its nineteen tiles.
    fn bind(
        &self,
        device: &Device,
        builder: &mut Encoder,
        sets: &CasterSets,
        bound: &mut Bound,
        wants_masked: bool,
        view_proj: Mat4,
    ) {
        let Device::Vulkan(driver) = device;
        let pipeline = if wants_masked {
            self.masked_pipeline
        } else {
            self.pipeline
        };
        if bound.needs_pipeline(wants_masked) {
            builder.bind_pipeline(pipeline);
            for (set, group) in sets.for_pipeline(wants_masked).into_iter().enumerate() {
                builder.bind_group(set as u32, driver.import_set(group));
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

fn run_intersects(casters: &DrawList<'_>, run: &Range<usize>, frustum: &Frustum) -> bool {
    run.clone()
        .any(|index| frustum.intersects(&casters.item(index).bounds))
}

/// Which pipeline the rasteriser already has, so the draw loop binds one only
/// when the run's material actually changes it.
///
/// Held by the caller rather than by [`ShadowPass::draw`], and that is the whole
/// point: the atlas calls into that loop once per face, and a guard scoped to
/// the call rebound a pipeline and three descriptor sets on every tile. Spans
/// one render pass instance and never more — each `record` and `record_atlas`
/// starts a fresh one, so nothing here assumes what a render pass boundary
/// preserves.
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
/// One struct rather than three parameters threaded through `record`,
/// `record_atlas` and `draw`, and it holds the *cutout* sets as an `Option`
/// because a frame is entitled to have no material table cached yet — nothing
/// here forces the alpha-testing pipeline into existence for a scene with no
/// foliage in it.
pub(super) struct CasterSets {
    pub objects: Arc<DescriptorSet>,
    pub materials: Arc<DescriptorSet>,
    pub textures: Arc<DescriptorSet>,
}

impl CasterSets {
    fn for_pipeline(&self, masked: bool) -> Vec<Arc<DescriptorSet>> {
        if masked {
            vec![
                self.objects.clone(),
                self.materials.clone(),
                self.textures.clone(),
            ]
        } else {
            vec![self.objects.clone()]
        }
    }
}

fn build_lit_view(ctx: &VkContext) -> Arc<ImageView> {
    build_lit_depth(ctx, ImageViewType::Dim2dArray)
}

/// The same, viewed as a plain 2D image, for the punctual atlas.
///
/// Two views over two allocations rather than one image viewed both ways,
/// because a view's type is fixed at creation and the two descriptors want
/// different ones. A pair of one-texel images is not worth avoiding.
fn build_lit_atlas_view(ctx: &VkContext) -> Arc<ImageView> {
    build_lit_depth(ctx, ImageViewType::Dim2d)
}

/// A 1x1 single-layer depth image cleared to 1.0 — "nothing is nearer than the
/// far plane", which every comparison reads as lit.
fn build_lit_depth(ctx: &VkContext, view_type: ImageViewType) -> Arc<ImageView> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vk_format(DEPTH_FORMAT),
            extent: [1, 1, 1],
            array_layers: 1,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("failed to allocate the shadows-off depth texture");

    let mut builder = Recorder::new(ctx);
    // Fresh from `Image::new`, so there is nothing to preserve; cleared once and
    // then only ever sampled, so this is the whole of its synchronisation.
    builder
        .image_barrier(record::to_transfer_dst(
            image.clone(),
            record::whole_image(&image),
            ImageLayout::Undefined,
        ))
        .clear_depth_stencil_image(ClearDepthStencilImageInfo {
            clear_value: ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
            ..ClearDepthStencilImageInfo::image(image.clone())
        })
        .image_barrier(record::to_shader_read(
            image.clone(),
            record::whole_image(&image),
            ImageLayout::TransferDstOptimal,
        ));
    builder.submit_and_wait(ctx);

    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type,
            ..ImageViewCreateInfo::from_image(&image)
        },
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Aabb;
    use crate::gfx::RenderItem;
    use crate::scene::{MaterialHandle, MeshHandle};
    use glam::Vec3A;

    fn item(x: f32) -> RenderItem {
        RenderItem {
            model: Mat4::IDENTITY,
            prev_model: Mat4::IDENTITY,
            bounds: Aabb {
                min: Vec3A::new(x - 0.1, -0.1, 0.4),
                max: Vec3A::new(x + 0.1, 0.1, 0.6),
            },
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            instance: 0,
        }
    }

    /// Identity clip volume: `x, y` in `[-1, 1]` and `z` in `[0, 1]`, so an item
    /// at `x = 0` is inside and one at `x = 50` is well outside.
    fn unit_frustum() -> Frustum {
        Frustum::from_view_projection(Mat4::IDENTITY)
    }

    /// A run no instance of which reaches the face is not drawn into it. This is
    /// the saving: half of a demo frame's atlas draws are a light's caster list
    /// redrawn into a face that cannot see any of it.
    #[test]
    fn a_run_entirely_outside_the_face_is_skipped() {
        let items = [item(50.0), item(60.0), item(70.0)];
        let order = [0u32, 1, 2];
        let list = DrawList::new(&items, &order);
        assert!(!run_intersects(&list, &(0..3), &unit_frustum()));
    }

    /// And one instance inside is enough to draw the run — the draw is instanced
    /// over the whole run, so the question is whether *any* of it is visible.
    #[test]
    fn a_run_with_one_instance_inside_is_drawn() {
        let items = [item(0.0)];
        let order = [0u32];
        let list = DrawList::new(&items, &order);
        assert!(run_intersects(&list, &(0..1), &unit_frustum()));
    }

    /// The instance inside need not be the first. Testing only the item the run
    /// starts at is the mistake this guards: runs are grouped by mesh and
    /// material, never by position, so a run is scattered through the scene.
    #[test]
    fn an_instance_inside_is_found_wherever_it_sits_in_the_run() {
        let items = [item(50.0), item(60.0), item(0.0)];
        let order = [0u32, 1, 2];
        let list = DrawList::new(&items, &order);
        assert!(run_intersects(&list, &(0..3), &unit_frustum()));
    }

    /// Bounds nothing could measure are drawn rather than culled, which
    /// [`Frustum::intersects`] promises and this relies on: a caster that
    /// vanished would be a missing shadow, and a caster drawn needlessly is a
    /// draw.
    #[test]
    fn unmeasurable_bounds_are_drawn_not_culled() {
        let mut only = item(50.0);
        only.bounds = Aabb::EMPTY;
        let items = [only];
        let order = [0u32];
        let list = DrawList::new(&items, &order);
        assert!(run_intersects(&list, &(0..1), &unit_frustum()));
    }

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
