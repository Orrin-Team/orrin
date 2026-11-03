use glam::Vec3;
use orrin_registry::Reflect;

/// The air, described once for both of the ways the frame draws it.
///
/// [`density`](Self::density), [`height_falloff`](Self::height_falloff) and
/// [`height`](Self::height) say what the medium *is*; everything else says how
/// much of it the frame is willing to simulate. The froxel volume covers the
/// first [`distance`](Self::distance) metres in front of the camera and the
/// analytic integral takes over past it, so the same three numbers describe both
/// segments of every view ray and turning [`volumetric`](Self::volumetric) off
/// changes how the fog is computed rather than how much of it there is.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct FogSettings {
    /// What fraction of extinction *scatters* rather than absorbs, per channel —
    /// the medium's single-scattering albedo, near 1.0 for water droplets and
    /// low for smoke.
    ///
    /// Renamed from `color` rather than reinterpreted, for the reason the
    /// lights' `intensity` was: it used to be the radiance the frame lerped
    /// toward, which stopped meaning anything when the frame became photometric.
    /// A constant 0.6 cd/m² is black beside a sunlit wall at several thousand,
    /// so the old field faded distance to night. What the air sends toward the
    /// camera is now derived — albedo times the light actually reaching it —
    /// which is the same quantity the froxels integrate and is why one field can
    /// serve both paths.
    pub albedo: Vec3,
    /// Extinction at `height`, per metre. Zero disables the effect entirely.
    pub density: f32,
    /// How fast density decays with altitude. Larger values make a thinner,
    /// more ground-hugging layer; zero makes the fog uniform at every height.
    pub height_falloff: f32,
    /// World-space altitude that `density` is measured at.
    pub height: f32,
    /// Whether the first `distance` metres are marched as a froxel volume, which
    /// is what puts the shadow maps into the air and gives the sun shafts.
    /// Structural: it registers two compute passes and a pair of volumes.
    pub volumetric: bool,
    /// How far in front of the camera the froxel volume reaches, in metres.
    ///
    /// The volume's texel count is fixed, so this trades reach against
    /// resolution rather than against cost: doubling it makes every froxel twice
    /// as long and buys nothing past where the analytic term was already
    /// adequate. Beyond it the fog is smooth anyway — there is no shadowing left
    /// to resolve at a hundred metres through haze.
    pub distance: f32,
    /// Henyey-Greenstein asymmetry, in `(-1, 1)`. Positive scatters forward, so
    /// the air brightens sharply around the sun; zero is isotropic. Real fog and
    /// haze sit near 0.7–0.8, which is what makes looking toward a low sun
    /// through it so much brighter than looking away.
    pub anisotropy: f32,
    /// How much of the reprojected history each froxel keeps. The volume is
    /// jittered along its depth axis every frame, and this is what turns that
    /// jitter into smooth light rather than crawling slices — the same trade
    /// `TaaSettings::feedback` makes, one dimension deeper.
    pub feedback: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            // Slightly blue-biased white: droplets scatter almost everything they
            // intercept, and the small tilt is Rayleigh's, not the droplets'.
            albedo: Vec3::new(0.92, 0.94, 1.0),
            density: 0.0,
            height_falloff: 0.1,
            height: 0.0,
            volumetric: true,
            distance: 64.0,
            anisotropy: 0.7,
            feedback: 0.9,
        }
    }
}
