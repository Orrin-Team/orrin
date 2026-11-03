use glam::Vec3;
use orrin_registry::Reflect;

/// Subsurface scattering: light that enters a surface, scatters inside it and
/// leaves somewhere else.
///
/// Two mechanisms answer to this one switch, and they are not alternatives so much
/// as the same physics measured at two scales. `Material::subsurface_radius` and
/// `Material::thickness` drive an analytic wrap and a transmittance term in
/// `shading.glsl`, which run in every queue and need no passes at all. On top of
/// that, `enabled` here adds the screen-space diffusion: the forward pass splits
/// the diffusible radiance into a target of its own and two separable blurs spread
/// it across the image before it is added back.
///
/// Turning it off is not turning scattering off — it drops the diffusion and the
/// wrap widens to stand in for it, which is why a scene looks similar either way
/// and different in the detail. What the diffusion buys is the part no per-pixel
/// term can reach: light crossing a silhouette, and a shadow terminator softened
/// by transport rather than by a curve fitted to it.
///
/// Notice what is *not* here: a strength. The forward pass routes this light out of
/// the colour target rather than adding it, so the composite has to add all of it
/// back or the same scene would lose energy by switching a pass on. The dials that
/// belong to the effect are on the material, where they describe a medium.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct SubsurfaceSettings {
    /// Off drops three passes *and* the second colour target the forward pass
    /// resolves into, which is why it is part of the frame's structure rather than
    /// a flag read at record time.
    pub enabled: bool,
    /// The widest kernel a pixel may open, in pixels.
    ///
    /// A performance control and nothing else: the physical width comes from the
    /// material's mean free path and the surface's distance from the camera, and a
    /// scattering face pressed against the lens would otherwise open a kernel the
    /// width of the frame. Every pixel that hits this clamp is scattering less far
    /// than its material asked for.
    pub max_radius: f32,
    /// How far each channel reaches relative to the one that reaches furthest.
    ///
    /// Normalised on the way to the GPU, so only the ratios matter. Frame-wide
    /// rather than per material, and that is a real limitation: the per-pixel
    /// target carries one radius, so the *width* of the diffusion is per material
    /// but its *chromaticity* is shared. Skin, wax and marble all fall off red to
    /// blue in much the same proportion, which is why one profile carries a scene;
    /// a scene that genuinely needs two wants the indexed profile table HDRP has,
    /// and the index would ride in the target beside the radius.
    ///
    /// The analytic half has no such limit — it reads the material's own
    /// per-channel radius directly.
    pub profile: Vec3,
}

impl Default for SubsurfaceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            // Wide enough for a face filling the frame at the default skin mean
            // free path, and far short of the cost of an unclamped kernel.
            max_radius: 48.0,
            // `Material::default`'s own radii, normalised. The two are the same
            // medium described twice — once per material and once per frame — so
            // they agree until someone changes one, which is the whole reason the
            // default is derived from the other rather than picked.
            profile: Vec3::new(1.0, 0.354, 0.229),
        }
    }
}

impl SubsurfaceSettings {
    /// The profile with its widest channel at one, which is the form the blur
    /// wants: the kernel's extent comes from the per-pixel radius, and this only
    /// says how much of it each channel uses.
    ///
    /// A profile with no positive channel would scatter nothing anywhere and read
    /// as the feature being broken rather than as the setting being empty, so it
    /// falls back to scattering every channel the full width.
    pub fn normalised_profile(&self) -> Vec3 {
        let widest = self.profile.max_element();
        if widest <= 0.0 {
            return Vec3::ONE;
        }
        (self.profile / widest).clamp(Vec3::ZERO, Vec3::ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the ratios matter, so scaling the whole profile has to change nothing
    /// the blur sees — otherwise the setting would double as a strength dial, and
    /// this effect deliberately has none.
    #[test]
    fn the_profile_is_scale_invariant() {
        let base = SubsurfaceSettings::default();
        let scaled = SubsurfaceSettings {
            profile: base.profile * 7.5,
            ..base
        };
        let a = base.normalised_profile();
        let b = scaled.normalised_profile();
        assert!((a - b).length() < 1e-6, "{a} != {b}");
    }

    /// The widest channel spans the kernel the per-pixel radius opened. If it came
    /// back below one, every material's authored mean free path would be quietly
    /// scaled down.
    #[test]
    fn the_widest_channel_spans_the_kernel() {
        assert_eq!(
            SubsurfaceSettings::default()
                .normalised_profile()
                .max_element(),
            1.0
        );
    }

    /// A profile nobody filled in must not switch the effect off by the back door:
    /// the honest way to get no scattering is a black `subsurface_color`, and a
    /// zeroed profile here should read as "no preference" instead.
    #[test]
    fn an_empty_profile_scatters_every_channel() {
        let settings = SubsurfaceSettings {
            profile: Vec3::ZERO,
            ..SubsurfaceSettings::default()
        };
        assert_eq!(settings.normalised_profile(), Vec3::ONE);
    }
}
