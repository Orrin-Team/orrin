//! The viewport's wireframe draw modes: one graph node after the tonemap,
//! drawing every mesh the frame drew as its edges.
//!
//! **Edges from barycentrics, on any device.** Each triangle's corners are
//! numbered 0, 1, 2 and the fragment shader draws where the smallest of the
//! three coordinates reaches zero, at a constant width in pixels with an
//! antialiased last pixel. That needs a corner to know which corner it is,
//! which an indexed draw cannot tell it — two triangles share a vertex — so the
//! draw is *not* indexed: it runs `index_count` vertices and the shader fetches
//! each one's index and position itself out of the shared geometry, bound as
//! storage through [`BindingResource::Geometry`]. The rejected alternatives
//! both need something a device may not have: a line fill mode
//! (`fillModeNonSolid`, Metal's `triangleFillMode`) gives one-pixel aliased
//! lines and is a device feature on every backend; a fragment barycentric
//! extension is missing on older desktop GPUs. De-indexing the geometry into a
//! second copy would work everywhere too, at 60 bytes a triangle of memory
//! for as long as the mode is on.
//!
//! **Over the shaded frame** the node attaches the prepass depth read-only and
//! hides the edges the scene covers. The lines are drawn with the
//! *unjittered* projection — they land on the resolved frame, where a jittered
//! line would shimmer — while that depth was rasterised jittered, so an edge's
//! depth is nudged toward the camera by a fixed fraction of itself to win
//! against its own face. **Alone** the node clears the frame and draws every
//! edge with no depth at all, which is what every engine's wireframe view shows.
//! In both, the lit frame under it is still rendered: skipping it in the alone
//! mode is a second change, to which passes the frame declares.
//!
//! **Cost.** One draw per object the frame drew, opaque and blended, each with
//! its matrix pushed — the outline mask's shape, not the multi-draw the
//! geometry passes use, because this is the editor's and only runs while asked
//! for. Off, the node is not in the graph.

use glam::Mat4;

use crate::gfx::DrawList;
use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Binding, BindingResource, Blend, DepthCompare, DepthTest, Device, Encoder, GeometryStream,
    GraphicsPipelineDesc, PipelineId, Viewport,
};
use crate::gfx::shaders::ShaderId;

use super::forward::GpuMesh;

/// How the wireframe is drawn, when it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wireframe {
    /// Edges over the shaded frame, hidden where the scene's depth covers them.
    OverShaded,
    /// Edges alone, every one, on a flat background.
    Alone,
}

/// What the alone mode clears to, linear.
pub const BACKGROUND: [f32; 4] = [0.03, 0.032, 0.036, 1.0];

/// Line colours in sRGB: dark and half-transparent over a lit frame, light on
/// the dark background.
const OVER_SHADED: [f32; 4] = [0.05, 0.05, 0.06, 0.55];
const ALONE: [f32; 4] = [0.82, 0.84, 0.88, 1.0];

/// In pixels.
const WIDTH: f32 = 1.5;

/// The fraction of its own depth an edge is moved toward the camera by. Large
/// enough to beat a face rasterised half a pixel away at a grazing angle, and
/// at a hundred metres still five centimetres.
const NUDGE: f32 = 5e-4;

pub struct WireframePass {
    over: PipelineId,
    alone: PipelineId,
    linear_output: bool,
}

impl WireframePass {
    pub fn new(device: &Device, swapchain_format: Format) -> Self {
        let formats = [swapchain_format];
        let base = GraphicsPipelineDesc {
            blend: Blend::Alpha,
            ..GraphicsPipelineDesc::fullscreen(
                ShaderId::WireframeVert,
                ShaderId::WireframeFrag,
                &formats,
            )
        };
        Self {
            over: device.create_graphics_pipeline(&GraphicsPipelineDesc {
                depth_format: Some(super::swapchain::DEPTH_FORMAT),
                depth_test: Some(DepthTest {
                    write: false,
                    compare: DepthCompare::GreaterOrEqual,
                }),
                ..base
            }),
            alone: device.create_graphics_pipeline(&base),
            linear_output: swapchain_format.is_srgb(),
        }
    }

    /// Draw every object in `lists` as its edges. `depth` is whether the node
    /// attached one, which decides the pipeline whatever the mode asked for.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &Device,
        builder: &mut Encoder,
        mode: Wireframe,
        depth: bool,
        lists: &[DrawList<'_>],
        meshes: &[GpuMesh],
        view_proj: Mat4,
        extent: [u32; 2],
    ) {
        let pipeline = if depth { self.over } else { self.alone };
        let geometry = device.create_bind_group(
            pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Geometry(GeometryStream::Positions),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Geometry(GeometryStream::Indices),
                },
            ],
        );
        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        builder.bind_pipeline(pipeline);
        builder.bind_group(0, geometry);

        let color = match mode {
            Wireframe::OverShaded => OVER_SHADED,
            Wireframe::Alone => ALONE,
        };
        let color = if self.linear_output {
            let [r, g, b, a] = color;
            [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a]
        } else {
            color
        };
        let nudge = if depth { NUDGE } else { 0.0 };
        for list in lists {
            for &index in list.order {
                let item = &list.items[index as usize];
                let Some(mesh) = meshes.get(item.mesh.0 as usize) else {
                    continue;
                };
                builder.push_constants(&push_words(view_proj * item.model, color, mesh, nudge));
                builder.draw(mesh.span.index_count, 1, 0, 0);
            }
        }
    }
}

/// The push block both stages declare: the matrix, the colour, where the mesh
/// sits in the shared geometry, the width and the nudge.
fn push_words(mvp: Mat4, color: [f32; 4], mesh: &GpuMesh, nudge: f32) -> [u32; 24] {
    let mut words = [0u32; 24];
    for (word, float) in words
        .iter_mut()
        .zip(mvp.to_cols_array().iter().chain(&color))
    {
        *word = float.to_bits();
    }
    words[20] = mesh.span.first_index;
    words[21] = mesh.span.vertex_offset as u32;
    words[22] = WIDTH.to_bits();
    words[23] = nudge.to_bits();
    words
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
