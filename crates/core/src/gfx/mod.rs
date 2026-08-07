pub mod graph;
pub mod headless;
pub mod sh;
pub mod shadows;
pub mod vulkan;

pub use headless::HeadlessBackend;

use crate::geom::Aabb;
use crate::scene::{
    BloomSettings, Camera, CpuMesh, EnvironmentSettings, HdrSettings, MaterialHandle, MeshHandle,
    SsaoSettings,
};
use glam::{Mat3, Mat4, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;

#[derive(BufferContents, VertexTrait, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    /// Object-space tangent (+U texture direction) in `xyz`; `w` is the
    /// bitangent handedness (±1) used to rebuild the TBN basis for normal maps.
    #[format(R32G32B32A32_SFLOAT)]
    pub tangent: [f32; 4],
}

/// One renderable instance, as extraction hands it to the passes. Everything a
/// pass needs is here: no pass reaches back into the world, and none recomputes
/// what extraction already knew.
///
/// Not necessarily *visible* — an object the camera culls still gets one if any
/// cascade wants it as a caster. What each pass draws is a [`DrawList`] naming a
/// subset of these, never the array itself.
#[derive(Clone, Copy, Debug)]
pub struct RenderItem {
    pub model: Mat4,
    /// Inverse-transpose of `model`'s upper 3x3, so normals stay perpendicular
    /// under non-uniform scale. Derived from the transform's rotation and scale
    /// at extraction rather than inverted out of `model` per pass per frame.
    pub normal_matrix: Mat3,
    /// World-space bounds: what the camera frustum tested, and what a shadow
    /// cascade tests against its own frustum without re-deriving anything.
    pub bounds: Aabb,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
}

/// One pass's draw order over a shared item array.
///
/// Extraction derives each renderable's matrices and bounds once, into a single
/// `items` array, and every list a frame draws — the camera's, each cascade's —
/// is an ordering of `u32` indices into it. A cascade that shares an object with
/// the camera shares the `RenderItem`, so widening a list costs four bytes
/// rather than the 144 a `RenderItem` occupies.
///
/// `order` is grouped into maximal (mesh, material) runs, which is what lets a
/// pass collapse each run into one instanced draw.
#[derive(Clone, Copy)]
pub struct DrawList<'a> {
    pub items: &'a [RenderItem],
    pub order: &'a [u32],
}

impl<'a> DrawList<'a> {
    pub fn new(items: &'a [RenderItem], order: &'a [u32]) -> Self {
        Self { items, order }
    }

    pub const EMPTY: DrawList<'static> = DrawList {
        items: &[],
        order: &[],
    };

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The item at position `i` in the draw order.
    pub fn item(&self, i: usize) -> &'a RenderItem {
        &self.items[self.order[i] as usize]
    }

    /// Split the order into maximal runs sharing a mesh and material.
    ///
    /// Correct only because extraction groups on the same key: an ungrouped
    /// order still yields valid runs, just short ones, so a missed grouping
    /// costs performance rather than producing wrong pixels.
    pub fn runs(&self) -> impl Iterator<Item = std::ops::Range<usize>> + 'a {
        let list = *self;
        let mut start = 0usize;
        std::iter::from_fn(move || {
            if start >= list.len() {
                return None;
            }
            let first = list.item(start);
            let key = (first.mesh.0, first.material.0);
            let mut end = start + 1;
            while end < list.len() && {
                let item = list.item(end);
                (item.mesh.0, item.material.0) == key
            } {
                end += 1;
            }
            let run = start..end;
            start = end;
            Some(run)
        })
    }
}

pub const MAX_POINT_LIGHTS: usize = 16;

/// Size of the shader's bound texture array (set 2). Keep at or below the
/// device's `maxPerStageDescriptorSampledImages` (≥16 guaranteed; MoltenVK
/// allows far more).
pub const MAX_TEXTURES: usize = 64;

/// The `u32` is the texture's index in the shader's array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The direction the light *travels* (e.g. roughly downward for a sun).
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Clone, Debug)]
pub struct SceneLighting {
    pub ambient_color: Vec3,
    pub ambient_intensity: f32,
    pub sun: DirectionalLight,
    /// Anything past [`MAX_POINT_LIGHTS`] is ignored.
    pub point_lights: Vec<PointLight>,
    /// Blinn-Phong specular exponent. Higher = smaller, sharper highlight.
    pub shininess: f32,
    pub specular_strength: f32,
    pub fog_color: Vec3,
    /// Fog extinction at `fog_height`. Zero disables the effect.
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub fog_height: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub emissive: Vec3,
    pub albedo_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive: Vec3::ZERO,
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
        }
    }
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::new(0.6, 0.7, 1.0),
            ambient_intensity: 0.15,
            sun: DirectionalLight {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::new(1.0, 0.97, 0.92),
                intensity: 1.0,
            },
            point_lights: Vec::new(),
            shininess: 32.0,
            specular_strength: 0.4,
            fog_color: Vec3::new(0.55, 0.62, 0.72),
            fog_density: 0.005,
            fog_height_falloff: 0.1,
            fog_height: 0.0,
        }
    }
}

// The seam between the engine and a concrete graphics API: implement for other
// backends (wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    /// Object-space bounds derived at upload; `None` for a handle this backend
    /// never issued. Mirrored into [`MeshBounds`](crate::scene::MeshBounds) at
    /// load, since culling runs before any backend type is in reach.
    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb>;
    fn load_material(&mut self, material: &Material) -> MaterialHandle;
    fn load_texture(&mut self, pixels: &[u8], width: u32, height: u32, srgb: bool)
    -> TextureHandle;
    /// Replace the environment with one baked from an equirectangular source:
    /// tightly packed RGBA f32, row-major from the top-left.
    ///
    /// Baking is synchronous — it happens once, at load, and the alternative is
    /// a half-written cubemap visible to the first frame.
    fn load_environment(&mut self, pixels: &[f32], width: u32, height: u32);
    fn resize(&mut self, extent: [u32; 2]);
    /// `dt` is the seconds elapsed since the last frame — what any temporal
    /// effect a backend runs needs, exposure adaptation being the first of them.
    /// Zero means "converge immediately", which is what a one-shot render wants.
    fn render(
        &mut self,
        draws: DrawList<'_>,
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        dt: f32,
    );
}

#[cfg(test)]
mod draw_list_tests {
    use super::{DrawList, MaterialHandle, MeshHandle, RenderItem};
    use crate::geom::Aabb;
    use glam::{Mat3, Mat4, Vec3};

    fn item(mesh: u32, material: u32) -> RenderItem {
        RenderItem {
            model: Mat4::IDENTITY,
            normal_matrix: Mat3::IDENTITY,
            bounds: Aabb {
                min: Vec3::splat(-0.5),
                max: Vec3::splat(0.5),
            },
            mesh: MeshHandle(mesh),
            material: MaterialHandle(material),
        }
    }

    /// Runs over the identity order, which is what a list nothing reordered has.
    fn ranges(items: &[RenderItem]) -> Vec<(usize, usize)> {
        let order: Vec<u32> = (0..items.len() as u32).collect();
        DrawList::new(items, &order)
            .runs()
            .map(|run| (run.start, run.end))
            .collect()
    }

    #[test]
    fn a_grouped_order_collapses_into_one_run_per_mesh_material_pair() {
        let items = [
            item(0, 0),
            item(0, 0),
            item(0, 1),
            item(3, 1),
            item(3, 1),
            item(3, 1),
        ];
        assert_eq!(ranges(&items), vec![(0, 2), (2, 3), (3, 6)]);
    }

    /// Every run must be non-empty, contiguous, and cover the order exactly —
    /// `object_base` is the run's start, so a gap or overlap would draw an
    /// instance against another object's transform.
    #[test]
    fn runs_partition_the_order_without_gaps_or_overlaps() {
        let items = [item(1, 0), item(1, 0), item(2, 7), item(2, 7), item(9, 9)];
        let ranges = ranges(&items);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, items.len());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "runs must be contiguous: {ranges:?}");
        }
        assert!(ranges.iter().all(|(start, end)| end > start));
        assert_eq!(
            ranges.iter().map(|(s, e)| e - s).sum::<usize>(),
            items.len()
        );
    }

    /// Same mesh but a different material can't share an instanced draw: the
    /// material index is a per-run push constant.
    #[test]
    fn a_material_change_breaks_a_run() {
        let items = [item(4, 0), item(4, 1)];
        assert_eq!(ranges(&items), vec![(0, 1), (1, 2)]);
    }

    /// An ungrouped order still has to partition correctly — it just yields
    /// more, shorter runs. Wrong pixels are not an acceptable cost of a missed
    /// grouping.
    #[test]
    fn an_ungrouped_order_still_partitions_correctly() {
        let items = [item(0, 0), item(5, 0), item(0, 0)];
        assert_eq!(ranges(&items), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn an_empty_order_has_no_runs() {
        assert!(ranges(&[]).is_empty());
    }

    /// The point of the indirection: runs are keyed by what the *order* points
    /// at, not by where the items happen to sit in the array.
    #[test]
    fn runs_follow_the_order_not_the_item_array() {
        let items = [item(0, 0), item(7, 7), item(0, 0)];
        let order = [0u32, 2, 1];
        let ranges: Vec<_> = DrawList::new(&items, &order)
            .runs()
            .map(|run| (run.start, run.end))
            .collect();
        assert_eq!(ranges, vec![(0, 2), (2, 3)]);
    }
}
