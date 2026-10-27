pub mod graph;
pub mod headless;
pub mod punctual;
pub mod sh;
pub mod shadows;
mod texel;
pub mod thumbnail;
pub mod vulkan;

pub use headless::HeadlessBackend;
pub use texel::{TexelFormat, TextureLevels};

use crate::geom::Aabb;
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, CpuMesh, DofSettings, EnvironmentSettings,
    FogSettings, HdrSettings, MaterialHandle, MeshHandle, MotionBlurSettings, RefractionSettings,
    SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings, TransparencySettings,
};
use glam::{Mat3, Mat4, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex as VertexTrait;

/// A vertex as meshes are *authored*: one struct with everything in it, which is
/// what mesh generation, model import and the collision builders all write.
///
/// Deliberately not what the GPU is given. Upload de-interleaves it into
/// [`PositionVertex`] and [`SurfaceVertex`], for the reason those two document —
/// but nothing on the CPU side has to know that, which is why this type still
/// exists and still has every field.
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    /// Object-space tangent (+U texture direction) in `xyz`; `w` is the
    /// bitangent handedness (±1) used to rebuild the TBN basis for normal maps.
    pub tangent: [f32; 4],
}

/// The half of a vertex a *depth-only* pass reads: where it is, and — for a
/// cutout material — where to sample the alpha from.
///
/// Split out of [`Vertex`] because the shadow passes are the frame's only
/// consumers that need this and nothing else, and a stride is paid whether or
/// not the attributes in it are fetched. AMD's RDNA guide asks for exactly this
/// ("allocate position data in a separate vertex stream to improve depth-only
/// passes"): with one interleaved 60-byte vertex, every cascade and every atlas
/// tile pulls 60 bytes per vertex to use 12 of them, four to five times over per
/// frame.
///
/// 20 bytes rather than 12 because the cutout shadow variant alpha-tests and so
/// needs the texture coordinate, and one stream that serves both shadow
/// pipelines beats a third buffer that serves one. The plain variant declares no
/// `uv` and therefore fetches none — only the stride is shared.
///
/// This costs no memory: the two halves partition [`Vertex`] rather than
/// duplicating any of it, so a mesh occupies exactly what it did before, in two
/// buffers instead of one.
#[derive(BufferContents, VertexTrait, Clone, Copy, Debug)]
#[repr(C)]
pub struct PositionVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

/// The other half: what a pass that actually *shades* needs on top of
/// [`PositionVertex`].
///
/// Bound alongside it by the four passes that rasterise a surface rather than
/// just its depth — the geometry prepass, the forward pass, and the two
/// non-opaque queues. Between them the two streams still add up to the same 60
/// bytes the single interleaved vertex was.
#[derive(BufferContents, VertexTrait, Clone, Copy, Debug)]
#[repr(C)]
pub struct SurfaceVertex {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub tangent: [f32; 4],
}

impl Vertex {
    /// The two halves this vertex is uploaded as.
    ///
    /// One function so the split is stated once: a field that moved between the
    /// two streams without its shader declaration moving with it would be a
    /// silently wrong mesh, not a compile error.
    pub fn split(&self) -> (PositionVertex, SurfaceVertex) {
        (
            PositionVertex {
                position: self.position,
                uv: self.uv,
            },
            SurfaceVertex {
                normal: self.normal,
                color: self.color,
                tangent: self.tangent,
            },
        )
    }
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
    /// What `model` was the last time this entity was extracted, and `model`
    /// itself the first time it ever was. The prepass reprojects each vertex
    /// through it to write a motion vector, so an object that appears this frame
    /// reads as stationary rather than as having flown in from wherever the
    /// slot's previous occupant stood.
    pub prev_model: Mat4,
    /// World-space bounds: what the camera frustum tested, and what a shadow
    /// cascade tests against its own frustum without re-deriving anything.
    ///
    /// [`Aabb::EMPTY`] for an opaque object under GPU culling, whose
    /// visibility the dispatch decides from its mesh's own bounds.
    pub bounds: Aabb,
    pub mesh: MeshHandle,
    pub material: MaterialHandle,
    /// Which row of the persistent instance buffer holds this object's
    /// matrices: the entity's slot, which is stable for as long as the entity
    /// lives. Every list that draws this object names the same row, so an
    /// object in the camera's list and in four cascades occupies one row rather
    /// than five, and a row survives the frames in which nothing about it
    /// changed. See `vulkan::instances::InstanceStore`.
    pub instance: u32,
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
    pub fn runs(&self) -> Runs<'a> {
        Runs {
            list: *self,
            start: 0,
        }
    }
}

/// [`DrawList::runs`], as a type a caller can hold.
///
/// Named rather than `impl Iterator` because the passes wrap it in an iterator
/// of their own — one that yields either these runs or the batches a compute
/// dispatch culled — and a wrapper cannot store what it cannot name. Collecting
/// instead would be an allocation per pass per frame.
pub struct Runs<'a> {
    list: DrawList<'a>,
    start: usize,
}

impl Iterator for Runs<'_> {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.list.len() {
            return None;
        }
        let first = self.list.item(self.start);
        let key = (first.mesh.0, first.material.0);
        let mut end = self.start + 1;
        while end < self.list.len() && {
            let item = self.list.item(end);
            (item.mesh.0, item.material.0) == key
        } {
            end += 1;
        }
        let run = self.start..end;
        self.start = end;
        Some(run)
    }
}

pub const MAX_POINT_LIGHTS: usize = 16;

/// Half the point-light cap, because a cone is a narrower thing to want a lot
/// of and every slot costs the same uniform-buffer space whether a scene uses it
/// or not. Keep in sync with `MAX_SPOT_LIGHTS` in `forward.frag`.
pub const MAX_SPOT_LIGHTS: usize = 8;

/// Size of the shader's bound texture array (set 2). Keep in sync with
/// `MAX_TEXTURES` in `shading.glsl`, `prepass.frag` and `shadow.frag` — the
/// three fragment shaders that bind the array.
///
/// The ceiling is MoltenVK's, not the desktop drivers': an M5 Pro reports
/// `maxPerStageDescriptorSampledImages` 256 and `maxPerStageResources` 287,
/// where AMD and NVIDIA report six figures. The refraction shader is the worst
/// case in the frame and samples eight images outside this array (AO, cascades,
/// environment, contact shadows, the spot atlas, the fog volume and two copies
/// of the scene), so this cap plus those has to clear both limits with room for
/// the next pass that wants a target — which is what puts it below 256 rather
/// than at it.
pub const MAX_TEXTURES: usize = 192;

/// How many decals a frame may project. Keep in sync with `MAX_DECALS` in
/// `shaders/decals.glsl`.
///
/// A flat cap with a brute-force loop behind it, and that is a deliberate first
/// version rather than an oversight. Binning decals into screen tiles is the
/// same machinery clustered lighting wants, it is a compute pass and a buffer of
/// its own, and it only starts paying at a decal count this cap does not reach —
/// sixteen boxes tested against a fragment is sixteen matrix-vector products and
/// an early-out, which is less than one of the lighting loops already costs.
/// What the cap buys in the meantime is that decals add no pass, so the render
/// graph and its golden barrier plan are untouched by the whole feature.
pub const MAX_DECALS: usize = 16;

/// One projected decal, as extraction hands it to the passes: the matrices
/// resolved, the textures already indices, the angle already a cosine.
///
/// Mirrors [`Decal`](crate::scene::Decal) the way [`PointLight`] mirrors a
/// `Light` — the component says what a decal *is* in the units it was authored
/// in, and this says what the shader needs. Nothing past extraction knows a
/// decal was ever an entity.
#[derive(Clone, Copy, Debug)]
pub struct DecalInstance {
    /// World space into the decal's unit cube, `[-0.5, 0.5]` on each axis: the
    /// inverse of the entity's world transform. Both the containment test and
    /// the texture coordinate come out of one multiply.
    pub world_to_decal: Mat4,
    /// The decal's own axes in world space, normalised — columns `x`, `y`, `z`.
    /// The projection runs along `-z`, the engine's forward.
    ///
    /// Carried rather than recovered from the inverse above, because recovering
    /// it means renormalising three rows per fragment per decal to undo a scale
    /// the CPU already knows.
    pub axes: Mat3,
    pub base_color: Vec3,
    pub opacity: f32,
    pub albedo: Option<TextureHandle>,
    pub normal: Option<TextureHandle>,
    pub metallic_roughness: Option<TextureHandle>,
    pub normal_strength: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub affects_surface: bool,
    /// Cosine of the fade angle. A cosine here rather than degrees for the
    /// reason a spot light's `outer_cos` is one: the shader has a dot product
    /// and would otherwise need an `acos` per fragment per decal to compare it.
    pub angle_cos: f32,
}

/// The `u32` is the texture's index in the shader's array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

/// Every light below carries the unit the *shader* wants rather than the one the
/// scene authored, because the conversion is per fixture and the loop over lights
/// is per pixel. `extract_lighting` is where lumens become candela; nothing past
/// it knows what kind of housing a light had.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    /// The direction the light *travels* (e.g. roughly downward for a sun).
    pub direction: Vec3,
    pub color: Vec3,
    /// Illuminance on a surface facing the light, in lux.
    ///
    /// Unconverted, unlike the punctual lights': a directional light has no
    /// distance to fall off over, so the illuminance it lays down is already what
    /// the BRDF wants to be multiplied by.
    pub illuminance: f32,
}

impl DirectionalLight {
    /// The direction *toward* the light, which is what every shader wants.
    ///
    /// One definition because two consumers march along it: the forward pass
    /// shades with it, and the contact-shadow pass traces the depth buffer
    /// toward it. A sign flipped in one of them would be a shadow cast on the
    /// lit side of everything, and nothing about either result would say which
    /// one was wrong.
    pub fn direction_to_light(&self) -> Vec3 {
        (-self.direction).normalize_or_zero()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    /// Luminous intensity in candela, already divided out of the authored lumens
    /// by [`Light::point_candela`](crate::scene::Light::point_candela).
    pub candela: f32,
    pub range: f32,
    /// Whether this light asks for atlas tiles. Asking is not getting: the atlas
    /// has a fixed number of tiles and a point light spends six, so
    /// [`fit`](punctual::fit) spends them on the lights that matter most and
    /// leaves the rest shading unshadowed.
    pub casts_shadows: bool,
}

/// A cone of light. `inner_cos`/`outer_cos` are cosines of the half angles
/// rather than the angles, because that is what the shader compares against —
/// converting once here keeps a transcendental out of the per-light loop.
#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: Vec3,
    /// The direction the cone points, normalised.
    pub direction: Vec3,
    pub color: Vec3,
    /// Luminous intensity in candela — the authored lumens through
    /// [`Light::spot_candela`](crate::scene::Light::spot_candela), so the
    /// reflector has already been accounted for and `outer_cos` below is only a
    /// falloff.
    pub candela: f32,
    pub range: f32,
    pub inner_cos: f32,
    pub outer_cos: f32,
    pub casts_shadows: bool,
}

#[derive(Clone, Debug)]
pub struct SceneLighting {
    pub ambient_color: Vec3,
    /// Luminance of the uniform fallback sky, in cd/m². Ignored once an
    /// environment is loaded — see [`AmbientLight`](crate::scene::AmbientLight).
    pub ambient_nits: f32,
    pub sun: DirectionalLight,
    /// Anything past [`MAX_POINT_LIGHTS`] is ignored.
    pub point_lights: Vec<PointLight>,
    /// Anything past [`MAX_SPOT_LIGHTS`] is ignored.
    pub spot_lights: Vec<SpotLight>,
    /// Blinn-Phong specular exponent. Higher = smaller, sharper highlight.
    pub shininess: f32,
    pub specular_strength: f32,
}

/// Which queue a material draws in.
///
/// A property of the material rather than of the entity, as it is in glTF and in
/// every engine that reads glTF: opacity is authored with the base colour and
/// the maps, so it belongs with them. Extraction reads it through
/// [`MaterialBlends`](crate::scene::MaterialBlends) to split the frame's draw
/// order in two.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Depth-tested, depth-writing, and shaded straight into the frame's colour.
    /// [`Material::alpha`] is ignored.
    #[default]
    Opaque,
    /// Opaque everywhere, but with the parts of it whose alpha falls below
    /// [`Material::alpha_cutoff`] cut away — a leaf card, a chain-link fence, a
    /// grate. Stays in the opaque queue in every sense that matters: it is in the
    /// prepass, in the caster lists, and everything screen-space treats what
    /// survives the cut as the surface it is.
    ///
    /// A mode rather than a flag on [`Opaque`](BlendMode::Opaque) because it
    /// draws through pipelines of its own. In the forward pass those enable
    /// **alpha to coverage**, which spends the MSAA samples the frame is already
    /// paying for on the cutout's edge rather than on the mesh's silhouette — so
    /// the edge is antialiased for very close to nothing, and no `discard`
    /// appears in the opaque shader to cost every other material its early
    /// depth test. The 1-sample passes — prepass, shadow — have no coverage to
    /// spend and alpha-test with a `discard` instead, in shader variants of
    /// their own so that the same early-Z argument holds there.
    ///
    /// Two-sided, because a cutout sheet is what this exists for and a leaf has
    /// no back to cull.
    Masked,
    /// Accumulated by the weighted-blended pass instead: depth-tested against
    /// the opaque scene but writing no depth, and composited afterwards. Never
    /// written into the geometry prepass, so nothing screen-space — occlusion,
    /// reflections, defocus, the shutter — treats it as a surface.
    Blend,
    /// Refracted rather than blended: drawn by its own pass, which samples the
    /// already-composited frame through the surface instead of mixing with what
    /// the framebuffer happens to hold.
    ///
    /// A queue of its own rather than a flag on [`Blend`](BlendMode::Blend)
    /// because the two disagree about ordering. Weighted-blended transparency is
    /// commutative and must not be sorted; a surface that *fetches* its own
    /// background has to be drawn back to front, since the second one to sample
    /// the same pixel would otherwise double-count what is behind it. Kept out
    /// of the prepass and the caster lists exactly as `Blend` is.
    Transmissive,
}

impl BlendMode {
    /// Whether this mode leaves the opaque queue — out of the prepass, out of
    /// every caster list, and therefore invisible to everything screen-space.
    ///
    /// One predicate rather than two comparisons at each site, so adding a third
    /// non-opaque mode cannot half-land: extraction, `MaterialBlends` and the
    /// demo scene all ask this question and none of them cares *which* non-opaque
    /// queue the material ends up in.
    ///
    /// [`Masked`](BlendMode::Masked) answers *true*, and that is the whole
    /// difference between it and the two below it: a cutout writes depth, so it
    /// is a surface the prepass and the shadow maps have every reason to know
    /// about. What it does not share with [`Opaque`](BlendMode::Opaque) is a
    /// pipeline, which is [`Self::is_masked`]'s question and not this one.
    pub fn is_opaque(self) -> bool {
        matches!(self, BlendMode::Opaque | BlendMode::Masked)
    }

    /// Whether a draw in the opaque queue needs the alpha-testing pipeline
    /// rather than the plain one.
    ///
    /// Asked per run by each of the three passes that rasterise opaque geometry,
    /// against the material table the backend already holds — which is why a
    /// cutout needs no draw list of its own. Extraction stays one opaque queue,
    /// sorted and ordered front to back exactly as before, and the passes switch
    /// pipeline where a run's answer changes.
    pub fn is_masked(self) -> bool {
        matches!(self, BlendMode::Masked)
    }
}

/// Named for the glTF extensions each block mirrors — `KHR_materials_clearcoat`,
/// `_sheen`, `_anisotropy`, `_transmission` and `_volume` — so a loader has one
/// obvious place to put what it read and no translation table to get wrong.
///
/// Every block is inert at its default: the shader gates each lobe on a flag
/// derived from whether the block was actually set, so a plain metallic-roughness
/// material costs exactly what it did before any of this existed.
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_color: Vec3,
    /// Opacity, multiplied by the albedo map's alpha. Read for
    /// [`BlendMode::Blend`] as the weight the surface accumulates with, and for
    /// [`BlendMode::Masked`] as the value tested against
    /// [`Self::alpha_cutoff`] — glTF's rule, and the reason the two modes share
    /// one field rather than each having a private one.
    pub alpha: f32,
    /// What [`Self::alpha`] must reach for a [`BlendMode::Masked`] fragment to
    /// survive. Ignored by every other mode.
    ///
    /// The cut is *softened over one pixel* rather than taken as a hard
    /// comparison: the forward pass converts the distance from the cutoff into
    /// coverage across the pixel's own footprint, which is what lets the MSAA
    /// samples already being paid for antialias the edge. A hard test would
    /// resolve to the same four-level staircase a cutout has always had.
    ///
    /// `0.5` because that is glTF's default and because a foliage atlas is
    /// authored against it.
    pub alpha_cutoff: f32,
    pub blend: BlendMode,
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    /// Radiance the surface emits on its own, in **nits** (cd/m²) — the same
    /// unit the frame is measured in, because that is what it adds to. A monitor
    /// is a few hundred, a neon tube a couple of thousand, a filament tens of
    /// thousands.
    ///
    /// A luminance rather than a power, unlike every light above: this is spread
    /// over whatever area the mesh happens to have, so scaling the mesh scales
    /// how much light it appears to put out — which is the one place emission
    /// differs from a fixture with a lumen rating.
    pub emissive: Vec3,
    pub albedo_texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,

    /// Strength of the second specular lobe, `0` for none. The coat is a thin
    /// dielectric film over everything else: it adds its own reflection and
    /// attenuates the layers beneath by what it reflected away.
    pub clearcoat: f32,
    /// Perceptual roughness of that lobe, independent of the base's — a scuffed
    /// coat over polished metal is the whole reason the two are separate.
    pub clearcoat_roughness: f32,
    /// `r` = strength, `g` = roughness, multiplying the two scalars above.
    pub clearcoat_texture: Option<TextureHandle>,
    /// Tangent-space normals for the coat alone. Absent, the coat uses the
    /// *geometric* normal rather than the base layer's normal-mapped one, which
    /// is glTF's rule and the physical one: an orange-peel coat and the grain
    /// under it are different surfaces.
    pub clearcoat_normal_texture: Option<TextureHandle>,

    /// Retroreflective rim lobe for cloth. Black for none; this is a colour
    /// rather than a scalar because velvet and satin owe their look to a sheen
    /// tinted away from the base.
    pub sheen_color: Vec3,
    /// Width of that lobe. Low is a tight satin edge, high a broad velvet bloom.
    pub sheen_roughness: f32,
    /// `rgb` = colour, `a` = roughness, multiplying the two above.
    pub sheen_texture: Option<TextureHandle>,

    /// How far the specular highlight is stretched, in `[-1, 1]`. Positive
    /// stretches along the tangent (brushed metal, hair), negative across it;
    /// `0` is the isotropic GGX everything else uses.
    pub anisotropy: f32,
    /// Rotation of the stretch within the tangent plane, in radians. What lets a
    /// brushed disc have circular grain under a UV set that does not.
    pub anisotropy_rotation: f32,
    /// `rg` = direction as a signed tangent-space vector, `b` = strength.
    pub anisotropy_texture: Option<TextureHandle>,

    /// How much light passes *through* rather than being diffusely reflected.
    /// Non-zero only means anything for [`BlendMode::Transmissive`], which is
    /// the queue that owns the pass able to fetch what is behind the surface.
    pub transmission: f32,
    /// Index of refraction, `1.5` for glass. Drives both the Fresnel term and
    /// how far the refraction pass bends its lookup.
    pub ior: f32,
    /// Thickness of the volume behind the surface, in metres. Zero makes it a
    /// *thin* surface — a window pane, refracting but with no interior to travel
    /// through — which is why it is the default.
    ///
    /// Read by the subsurface block below as well, and it has to be the same
    /// number: how far light travels inside a wax block before it leaves is the
    /// question both a refracted ray and a scattered one are asking. That is also
    /// why an opaque material with subsurface scattering wants this authored —
    /// the default zero is a surface with no interior, and nothing transmits
    /// through one.
    pub thickness: f32,
    /// What the volume absorbs, Beer-Lambert, over `attenuation_distance`. White
    /// is a clear medium.
    pub attenuation_color: Vec3,
    /// The distance at which `attenuation_color` is reached. Infinite for a
    /// medium that never absorbs.
    pub attenuation_distance: f32,
    /// `r` = transmission, `g` = thickness, multiplying the two above. The green
    /// channel is the thickness map for the subsurface block too, for the reason
    /// [`Self::thickness`] is shared.
    pub transmission_texture: Option<TextureHandle>,

    /// Tint of the light that entered the surface, bounced around inside it and
    /// came back out. Black for none.
    ///
    /// A colour rather than a scalar, and that is the whole feature: what returns
    /// has travelled through a medium that absorbed some wavelengths more than
    /// others, so an ear lit from behind is red rather than bright. A surface
    /// without this term reflects everything at the boundary it arrived at, which
    /// is precisely what makes skin, leaves, wax and marble read as plastic.
    pub subsurface_color: Vec3,
    /// How far light travels inside the medium before it has scattered away, per
    /// channel, in **metres** — one world unit, as everywhere else.
    ///
    /// Per channel because it must be: one distance scatters every wavelength the
    /// same way and produces a grey blur, and the reason skin looks like skin is
    /// that red reaches several times further than blue. Sets both the width of
    /// the screen-space diffusion and, where that pass is not running, the width
    /// of the wrapped diffuse standing in for it.
    pub subsurface_radius: Vec3,
    /// Tightness of the forward-scattering lobe — what a leaf or an ear shows
    /// when the light is behind it and the camera is nearly looking into it.
    /// Higher is a narrower halo.
    pub subsurface_forward_scatter: f32,
    /// `rgb` multiplies `subsurface_color`, so a face can scatter through thin
    /// skin and not through an eyebrow.
    pub subsurface_texture: Option<TextureHandle>,

    /// How high each texel stands above the deepest point of the surface, read
    /// from `r`, white at the top. The convention every displacement map ships
    /// in, which is why a stone or brick set's `Displacement` image can be
    /// dropped straight in.
    ///
    /// With one authored, the surface stops being a plane: the view ray is
    /// marched through the height field and every other map is read where it
    /// lands, so mortar courses sit *behind* the bricks and slide against them
    /// as the camera moves — the parallax a normal map alone cannot produce,
    /// because a normal map changes only which way a flat pixel faces.
    ///
    /// What it is not: geometry. The silhouette stays the mesh's, the depth
    /// buffer is untouched, and nothing screen-space — occlusion, reflections,
    /// contact shadows, shadow maps — sees the relief. The relief therefore
    /// fades back to a flat surface as the view turns edge-on — full strength
    /// out to about 73 degrees off the normal, gone by 84 — because displacement
    /// without the occlusion a real groove would also produce is a smear, and
    /// the missing half is precisely the silhouette a height field cannot cut.
    /// Motion vectors come
    /// from the geometry too, so the relief moves a few centimetres differently
    /// from what the temporal resolve reprojects; that is the same trade every
    /// implementation of this makes, and the error is bounded by the depth
    /// below.
    pub height_texture: Option<TextureHandle>,
    /// How deep that field goes, in **metres** — the distance between a brick's
    /// face and the back of the mortar. A centimetre or two is a masonry wall,
    /// millimetres are floor tiles.
    ///
    /// Metres rather than a fraction of the UV square, unlike most engines,
    /// because the UV answer is not a property of the material: the same brick
    /// map over a wall tiled once and a wall tiled eight times would need two
    /// different numbers for one physical groove. The shader converts using the
    /// pixel's own footprint, so retiling the wall or scaling the mesh leaves
    /// the groove the depth it was authored at.
    ///
    /// Only read once [`Self::height_texture`] is set — there is nothing to
    /// march without a field — so this is what relief looks like the moment one
    /// is dropped in, the way [`Self::clearcoat_roughness`] is.
    pub parallax_depth: f32,
    /// Samples along the ray when the surface faces the camera, and when it is
    /// edge-on. The march spends the second number where the field is stretched
    /// across many pixels and the first where it is crossed in almost none, so
    /// this is the effect's cost dial: a distant wall wants both low, a floor
    /// the camera skims wants the upper one high. Clamped to 64.
    pub parallax_min_steps: u32,
    pub parallax_max_steps: u32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            alpha: 1.0,
            alpha_cutoff: 0.5,
            blend: BlendMode::Opaque,
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive: Vec3::ZERO,
            albedo_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,

            clearcoat: 0.0,
            // Only read when `clearcoat` is non-zero, so this is what a coat
            // looks like the moment one is switched on rather than a value that
            // does nothing: a mirror-smooth film, which is what a coat is unless
            // it was authored otherwise.
            clearcoat_roughness: 0.03,
            clearcoat_texture: None,
            clearcoat_normal_texture: None,

            sheen_color: Vec3::ZERO,
            sheen_roughness: 0.3,
            sheen_texture: None,

            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            anisotropy_texture: None,

            transmission: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: Vec3::ONE,
            attenuation_distance: f32::INFINITY,
            transmission_texture: None,

            subsurface_color: Vec3::ZERO,
            // Only read once `subsurface_color` is non-black, so — like
            // `clearcoat_roughness` above — these are what scattering looks like
            // the moment it is switched on rather than values that do nothing.
            // Skin's mean free paths, a few millimetres of red down to about one
            // of blue, which is also close enough to marble and wax to be a
            // sensible thing to start from and tune.
            subsurface_radius: Vec3::new(0.0048, 0.0017, 0.0011),
            subsurface_forward_scatter: 12.0,
            subsurface_texture: None,

            height_texture: None,
            // Two centimetres: a brick's face to the back of its mortar, and the
            // depth at which the shortening at grazing angles is not yet what
            // the surface is made of. Read only once a map is authored.
            parallax_depth: 0.02,
            // Eight layers resolve a field crossed head-on; thirty-two is where
            // a wall the camera skims stops showing the layers as terraces. The
            // pair is the cost dial, so the defaults are the ones that look
            // right rather than the ones that are cheapest.
            parallax_min_steps: 8,
            parallax_max_steps: 32,
        }
    }
}

impl Default for SceneLighting {
    fn default() -> Self {
        Self {
            ambient_color: Vec3::new(0.6, 0.7, 1.0),
            ambient_nits: 600.0,
            sun: DirectionalLight {
                direction: Vec3::new(-0.4, -1.0, -0.6).normalize(),
                color: Vec3::new(1.0, 0.97, 0.92),
                illuminance: 20_000.0,
            },
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            shininess: 32.0,
            specular_strength: 0.4,
        }
    }
}

// The seam between the engine and a concrete graphics API: implement for other
// backends (wgpu, D3D12) without touching scene/app code.
pub trait RenderBackend {
    /// Whether this backend decides each opaque view's visible set itself.
    ///
    /// Asked by [`extract_geometry`](crate::systems::extract_geometry), which
    /// otherwise does that work per entity for every view. False for a backend
    /// that has no compute of its own — the headless one — so the sweep answers
    /// the question the way it always did.
    fn gpu_culling(&self) -> bool {
        false
    }

    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle;
    /// Replace what an uploaded mesh's slot draws, for a reimported asset. The
    /// handle keeps its meaning, so nothing naming it changes; the bounds
    /// [`mesh_bounds`](Self::mesh_bounds) reports are the new mesh's.
    fn replace_mesh(&mut self, handle: MeshHandle, mesh: &CpuMesh);
    /// Object-space bounds derived at upload; `None` for a handle this backend
    /// never issued. Mirrored into [`MeshBounds`](crate::scene::MeshBounds) at
    /// load, since culling runs before any backend type is in reach.
    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb>;
    fn load_material(&mut self, material: &Material) -> MaterialHandle;
    /// Replace an uploaded material's parameters in place, for an edit made in
    /// the editor. Every draw using `handle` sees the change from the next
    /// frame; the handle keeps its meaning.
    fn update_material(&mut self, handle: MaterialHandle, material: &Material);
    fn load_texture(&mut self, pixels: &[u8], width: u32, height: u32, srgb: bool)
    -> TextureHandle;
    /// [`replace_mesh`](Self::replace_mesh) for a texture: every material
    /// sampling `handle` samples `texture`'s chain from the next frame. A
    /// block-compressed chain is only ever asked of a backend that
    /// [`samples_bc`](Self::samples_bc).
    fn replace_texture(&mut self, handle: TextureHandle, texture: &TextureLevels);
    /// Whether the device samples BC5 and BC7, so an imported texture can be
    /// uploaded compressed.
    fn samples_bc(&self) -> bool;
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
    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        draws: DrawList<'_>,
        // What the weighted-blended transparency pass accumulates, grouped by
        // (mesh, material) and deliberately not ordered by depth.
        transparent: DrawList<'_>,
        // What the refraction pass draws, grouped the same way and then ordered
        // back to front — which the list above must not be, and this one must.
        refractive: DrawList<'_>,
        // This frame's decals, already ordered back to front by
        // `Decal::sort_order`. Beside the lighting rather than among the three
        // lists above because a decal is not a draw: it enters no queue and
        // produces no `RenderItem` — see [`DecalInstance`].
        decals: &[DecalInstance],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        contact_shadows: &ContactShadowSettings,
        ssr: &SsrSettings,
        subsurface: &SubsurfaceSettings,
        transparency: &TransparencySettings,
        refraction: &RefractionSettings,
        taa: &TaaSettings,
        motion_blur: &MotionBlurSettings,
        dof: &DofSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        fog: &FogSettings,
        dt: f32,
    );
}

#[cfg(test)]
mod draw_list_tests {
    use super::{DrawList, MaterialHandle, MeshHandle, RenderItem};
    use crate::geom::Aabb;
    use glam::{Mat4, Vec3A};

    fn item(mesh: u32, material: u32) -> RenderItem {
        RenderItem {
            model: Mat4::IDENTITY,
            prev_model: Mat4::IDENTITY,
            bounds: Aabb {
                min: Vec3A::splat(-0.5),
                max: Vec3A::splat(0.5),
            },
            mesh: MeshHandle(mesh),
            material: MaterialHandle(material),
            instance: 0,
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
