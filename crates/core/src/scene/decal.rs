use glam::Vec3;
use orrin_registry::Reflect;

use crate::gfx::TextureHandle;

/// A projected decal: a box that stamps maps onto whatever opaque geometry it
/// contains.
///
/// The box is the entity's own transform applied to the unit cube, so a decal is
/// placed, rotated and scaled with the same tools everything else in the scene
/// is, and its extent is its scale in metres. It projects along the entity's
/// **forward**, which is `-Z` — the engine's convention everywhere, and the
/// reason a decal aimed at a wall is authored by pointing at it rather than away
/// from it. The other two axes are the decal's `u` and `v`: `+X` right, `+Y` up.
///
/// Nothing about this is a queue or a pass. A decal is read by the two passes
/// that already rasterise opaque geometry, at the point where they have sampled
/// a surface's maps and before anything is done with them, so what a decal
/// produces *is* the surface as far as the rest of the frame is concerned — it
/// is lit by every light, it is in the reflections, it is under the ambient
/// occlusion. That is what a screen-space stamp composited afterwards cannot be.
///
/// Two consequences worth knowing. It lands on blended and refractive surfaces
/// too, because those shade through the same file. And it lands on *every*
/// surface inside the box, so a decal large enough to swallow a wall's far side
/// stamps that as well — depth is not a test the projection makes, [`angle`] is
/// the tool for it, and a box no deeper than what it should hit is better than
/// either.
#[derive(Clone, Copy, Debug)]
pub struct Decal {
    /// Tints [`Self::albedo`], or is the colour outright when there is no map.
    pub base_color: Vec3,
    /// How much of the decal lands, `0` to `1`. Multiplies the albedo map's own
    /// alpha, which is what actually shapes most decals.
    pub opacity: f32,
    /// `rgb` = colour, `a` = coverage. Both the shape and the weight: a decal
    /// with no map is a solid box-shaped stamp, which is occasionally what a
    /// blood pool or a scorch wants and is almost never what anything else does.
    pub albedo: Option<TextureHandle>,
    /// Tangent-space normals in the *decal's* frame, not the receiver's — a
    /// bullet hole's dent points into the wall it was projected along, whatever
    /// UV layout that wall happens to have.
    pub normal: Option<TextureHandle>,
    /// glTF's packing, as everywhere else: `g` = roughness, `b` = metallic.
    pub metallic_roughness: Option<TextureHandle>,
    /// How far the normal map tilts the surface it lands on, `0` for not at all.
    /// Independent of [`Self::opacity`], because a decal that is mostly a dent —
    /// a scratch, a footprint in dust — wants its normals at full strength under
    /// a colour that barely registers.
    pub normal_strength: f32,
    /// What the receiver's metallic and roughness are driven *to* where the
    /// decal lands, multiplied by the map. Weighted by the same coverage the
    /// colour is, so a decal with no metallic-roughness map still wets or dulls
    /// the surface under it.
    pub metallic: f32,
    pub roughness: f32,
    /// Whether the two above are applied at all. A decal that only stains
    /// colour must not also drag the surface under it to a default roughness,
    /// and there is no value of `roughness` that means "leave it alone" — `0`
    /// is mirror-smooth.
    pub affects_surface: bool,
    /// How far a receiver's normal may point away from the projection direction
    /// before the decal fades off it, in **degrees**.
    ///
    /// This is what keeps a stamp off the walls of the hole it is projected
    /// into: a box that reaches the floor also reaches the skirting board, and
    /// without an angle test the decal smears down it in the stretched streaks
    /// that are the effect's signature failure. Faded rather than clipped, over
    /// the last quarter of the range, because a hard cutoff is its own artefact.
    pub angle: f32,
    /// Painter's order among decals over the same pixel: lower goes down first.
    /// Only meaningful between decals — every one of them lands under the
    /// lighting, never over it.
    pub sort_order: i32,
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            base_color: Vec3::ONE,
            opacity: 1.0,
            albedo: None,
            normal: None,
            metallic_roughness: None,
            // Only read once a normal map is authored, so — like
            // `Material::clearcoat_roughness` — this is what a decal's normals
            // look like the moment one is dropped in rather than a value that
            // does nothing.
            normal_strength: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            affects_surface: false,
            // Just past a right angle, so a decal lands on a face turned fully
            // side-on to it and on nothing behind that. The default most
            // authoring tools ship.
            angle: 90.0,
            sort_order: 0,
        }
    }
}

/// Whether the frame projects decals at all.
///
/// Unlike [`TransparencySettings`](crate::scene::TransparencySettings) and the
/// rest, this is *not* frame structure: decals add no pass and no render target,
/// so switching it changes nothing about the graph and reallocates nothing. It
/// is here as the A/B — "is this a decal?" is a question the offscreen captures
/// answer by rendering the scene with this false — and as the cost dial, since
/// off means the per-fragment loop is skipped on a branch that is uniform across
/// the whole frame.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct DecalSettings {
    pub enabled: bool,
}

impl Default for DecalSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}
