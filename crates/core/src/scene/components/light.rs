use glam::Vec3;
use orrin_registry::Reflect;
use std::f32::consts::{PI, TAU};

/// Variant names are part of the on-disk format, exactly like a component's id:
/// renaming `Directional` orphans every saved light that used it, and nothing
/// catches that at compile time. Add variants freely; rename them never.
///
/// Field *names* are the same promise, and adding one to a variant that already
/// ships is the same hazard from the other direction: a scene saved before it
/// existed has no such field, and the derive reports that as a broken entity
/// rather than an old one. `casts_shadows` therefore carries
/// `#[reflect(default = true)]`, which is what lets a point light written by an
/// earlier build load as the shadow caster it would have been.
///
/// # Units
///
/// Every quantity here is photometric, and the field name is the unit — which is
/// also why the old `intensity` is gone rather than reinterpreted. A scene
/// authored against that field fails to load and says which field it is missing;
/// silently reading `intensity = 3` as three lumens would load a scene that is
/// black for a reason nothing reports. Retrofitting this later would
/// re-tune every scene anyone had authored.
///
/// The unit differs per light type because the *fixture* differs, which is the
/// convention glTF, Filament and Unity's HDRP all share:
///
/// - A sun is infinitely far away, so it has no power to speak of and no
///   distance to fall off over. What it has is the illuminance it lays on a
///   surface facing it: **lux**. Full noon sun is about 100 000, an overcast day
///   20 000, a late sunset under 1 000.
/// - A bulb radiates in every direction, and what its box prints is total
///   luminous power: **lumens**. Domestic bulbs are 400–1 600, a shop light
///   3 000–6 000, an architectural flood tens of thousands.
///
/// Both become the candela the shader needs via the conversions below, and both
/// assume **one world unit is one metre** — an inverse-square falloff has to know
/// what the square is of.
#[derive(Clone, Copy, Debug, Reflect)]
pub enum Light {
    Directional {
        color: Vec3,
        /// Illuminance on a surface facing the light, in lux.
        lux: f32,
    },
    Point {
        color: Vec3,
        /// Total luminous power radiated in every direction, in lumens.
        lumens: f32,
        range: f32,
        /// Whether this light writes into the shadow atlas. The atlas holds a
        /// fixed number of tiles and a point light spends six of them, so this
        /// is the switch for a fill light that should never cost that — and
        /// what the importance sort spends its budget on when more lights ask
        /// than fit.
        #[reflect(default = true)]
        casts_shadows: bool,
    },
    /// A cone. `inner_angle` is where it is still at full brightness and
    /// `outer_angle` where it has fallen to nothing, both as the *half* angle
    /// from the axis in degrees — the same convention glTF uses, and the one
    /// that makes the cone's field of view `2 * outer_angle`.
    ///
    /// Its axis is the entity's forward, `-Z`, matching `Directional`.
    Spot {
        color: Vec3,
        /// Total luminous power the fixture radiates, in lumens — the same
        /// quantity a point light carries, so swapping one for the other keeps
        /// the bulb and changes only the housing around it.
        lumens: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        /// Whether the housing concentrates the bulb's output into the cone.
        ///
        /// With a reflector, narrowing `outer_angle` puts the same lumens
        /// through a smaller solid angle and the beam gets brighter, which is
        /// what the metal cup behind a real spot does. Without one, the fixture
        /// emits as a bare bulb and simply discards what falls outside the cone,
        /// so the angle is a mask and brightness is independent of it.
        ///
        /// The first is physical and the default; the second is what an artist
        /// wants while aiming a light they have already exposed for. HDRP calls
        /// the same switch "Reflector".
        #[reflect(default = true)]
        reflector: bool,
        /// One tile rather than a point light's six, because a cone is a single
        /// frustum — which makes a spot the cheapest thing in the atlas by a
        /// factor of six.
        #[reflect(default = true)]
        casts_shadows: bool,
    },
}

impl Light {
    #[inline]
    pub fn directional(color: Vec3, lux: f32) -> Self {
        Self::Directional { color, lux }
    }

    #[inline]
    pub fn point(color: Vec3, lumens: f32, range: f32) -> Self {
        Self::Point {
            color,
            lumens,
            range,
            casts_shadows: true,
        }
    }

    #[inline]
    pub fn spot(color: Vec3, lumens: f32, range: f32, inner_angle: f32, outer_angle: f32) -> Self {
        Self::Spot {
            color,
            lumens,
            range,
            inner_angle,
            outer_angle,
            reflector: true,
            casts_shadows: true,
        }
    }

    /// Luminous intensity, in candela, of a point light radiating `lumens`
    /// uniformly in every direction.
    ///
    /// A sphere subtends 4π steradian and the power is spread evenly over it, so
    /// this is the whole derivation. It lives here rather than in the extraction
    /// because a second integrator — the path tracer §3.4 wants — has to agree
    /// with the rasteriser about how bright a bulb is, and two copies of a
    /// constant divisor is exactly the drift that reads as a lighting bug.
    #[inline]
    pub fn point_candela(lumens: f32) -> f32 {
        lumens / (4.0 * PI)
    }

    /// Luminous intensity, in candela, of a spot light radiating `lumens` into a
    /// cone of half angle `outer_angle` degrees.
    ///
    /// With a reflector the power is confined to the cone's solid angle,
    /// `2π(1 - cos θ)`, so a narrow beam is an intense one. Without, the fixture
    /// is a bare bulb behind an aperture: it radiates over the full sphere and
    /// the cone only decides what escapes, which is [`point_candela`].
    ///
    /// The cone is clamped to the same range the extraction clamps
    /// `outer_angle` to. At θ → 0 the solid angle goes to zero and the intensity
    /// to infinity, which is not a light anyone authored — it is a divisor
    /// nobody guarded.
    #[inline]
    pub fn spot_candela(lumens: f32, outer_angle: f32, reflector: bool) -> f32 {
        if !reflector {
            return Self::point_candela(lumens);
        }
        let cos_outer = outer_angle
            .clamp(MIN_CONE_ANGLE, MAX_CONE_ANGLE)
            .to_radians()
            .cos();
        lumens / (TAU * (1.0 - cos_outer))
    }
}

/// The half angles a spot's outer cone is held between, shared by the extraction
/// and by [`Light::spot_candela`] so the angle a light is *shaded* with is the
/// angle it was *normalised* against. Diverging by even a degree puts energy in
/// the cone that the cone's own falloff then removes.
pub const MIN_CONE_ANGLE: f32 = 1.0;
pub const MAX_CONE_ANGLE: f32 = 89.0;

impl Default for Light {
    fn default() -> Self {
        // A heavily overcast day: bright enough to read as daylight, dim enough
        // that a fixture placed beside it still shows.
        Self::directional(Vec3::ONE, 20_000.0)
    }
}

/// Uniform light arriving from every direction, in **nits** (cd/m²) — a
/// luminance rather than a power, because a constant sky has no fixture and no
/// distance, only a brightness it is everywhere.
///
/// Only in force when no environment is loaded, which is what it stands in for:
/// the renderer folds it into the irradiance probe's constant band and into the
/// specular tint, so a scene with a real HDRI ignores it entirely.
#[derive(Clone, Copy, Debug)]
pub struct AmbientLight {
    pub color: Vec3,
    /// Luminance of that uniform sky, in cd/m². A clear zenith is a few
    /// thousand, an overcast one under a thousand, a blue-hour sky tens.
    pub nits: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.6, 0.7, 1.0),
            // About a tenth of the default sun's illuminance once integrated
            // over the hemisphere, which is roughly the sky-to-sun ratio on the
            // overcast day `Light::default` describes.
            nits: 600.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition of a candela: one lumen per steradian. A source radiating
    /// 4π lumens over the sphere's 4π steradian is therefore exactly 1 cd, and
    /// that is the whole conversion — worth pinning because every punctual light
    /// in the engine is scaled by it.
    #[test]
    fn a_sphere_of_lumens_is_one_candela() {
        assert!((Light::point_candela(4.0 * PI) - 1.0).abs() < 1e-5);
    }

    /// A reflector conserves power: narrowing the cone puts the same lumens
    /// through less solid angle, so the beam has to get brighter by exactly the
    /// ratio of those angles. Halving the solid angle doubles the candela.
    #[test]
    fn a_narrower_reflector_is_a_brighter_beam() {
        let wide = Light::spot_candela(1000.0, 60.0, true);
        let narrow = Light::spot_candela(1000.0, 20.0, true);
        assert!(narrow > wide, "{narrow} cd should exceed {wide} cd");

        // The ratio is the solid angles', not the angles' — an eyeballed falloff
        // would pass the comparison above and fail this.
        let solid = |deg: f32| TAU * (1.0 - deg.to_radians().cos());
        let expected = solid(60.0) / solid(20.0);
        assert!((narrow / wide - expected).abs() < 1e-3);
    }

    /// A hemisphere is half a sphere in both directions: a reflector opened to
    /// 90° confines the same lumens to 2π steradian, which is twice the
    /// intensity of the bare bulb that spread them over 4π. The two conversions
    /// meeting where the geometry says they must is what says neither has a
    /// stray factor in it.
    #[test]
    fn a_hemispherical_reflector_doubles_a_bare_bulb() {
        let bare = Light::spot_candela(1000.0, 89.0, false);
        let hemisphere = Light::spot_candela(1000.0, MAX_CONE_ANGLE, true);
        // 89° rather than 90° because that is where the cone clamps, so this is
        // the hemisphere the engine can actually author.
        assert!((hemisphere / bare - 2.0).abs() < 0.05);
    }

    /// Without a reflector the cone is a mask, so aiming a light must not change
    /// how bright it is — the entire reason the switch exists.
    #[test]
    fn a_bare_bulb_ignores_its_cone() {
        let narrow = Light::spot_candela(1000.0, 5.0, false);
        let wide = Light::spot_candela(1000.0, 80.0, false);
        assert_eq!(narrow, wide);
        assert_eq!(narrow, Light::point_candela(1000.0));
    }

    /// A cone authored at zero divides by a zero solid angle. The clamp is the
    /// only thing between that and an infinity in the light buffer.
    #[test]
    fn a_degenerate_cone_stays_finite() {
        assert!(Light::spot_candela(1000.0, 0.0, true).is_finite());
        assert!(Light::spot_candela(1000.0, -30.0, true).is_finite());
    }
}
