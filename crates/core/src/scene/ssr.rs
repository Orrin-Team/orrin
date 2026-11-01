use orrin_registry::Reflect;

/// Screen-space reflections: what the frame reflects off itself.
///
/// A stopgap, and worth saying so where the settings live. The trace can only
/// reflect what the screen already contains, so anything off-frame, behind the
/// camera, or hidden behind what it reflects has no radiance to give and falls
/// back to the environment cube. Every field below except `enabled` is about
/// making that fallback happen gracefully rather than about accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct SsrSettings {
    /// Off drops the three passes rather than tracing zero-length rays, which
    /// is why it is part of the frame's structure.
    pub enabled: bool,
    /// How much of the traced reflection to believe, as a multiplier on the
    /// per-ray confidence. One is the physical answer; lower is an art dial for
    /// scenes where the screen-space failure cases are conspicuous.
    pub intensity: f32,
    /// Past this roughness the surface is left to the environment cube. One ray
    /// per pixel cannot resolve a wide lobe, and a rough reflection is close to
    /// the prefiltered environment anyway — which is exactly why the cutoff
    /// costs so little.
    pub max_roughness: f32,
    /// How deep a surface is assumed to be, in metres, when deciding whether a
    /// ray landed on it or passed behind it. The depth buffer records a front
    /// face and nothing else, so this is the one number standing in for every
    /// object's thickness.
    pub thickness: f32,
    /// How far a ray travels before giving up, in metres.
    pub max_distance: f32,
    /// Steps the march may take. With the depth pyramid a step is a whole cell
    /// rather than a texel, so this is a budget for crossing the frame, not for
    /// resolution.
    pub max_steps: u32,
}

impl Default for SsrSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 1.0,
            // Wet asphalt and polished concrete sit around 0.3; past 0.6 a
            // single ray is mostly noise for a result the environment already
            // approximates well.
            max_roughness: 0.6,
            thickness: 0.5,
            max_distance: 50.0,
            max_steps: 64,
        }
    }
}

impl SsrSettings {
    /// How much of the traced reflection survives at `roughness`, before the
    /// per-ray confidence the trace itself computes.
    ///
    /// Mirrored by the fade in `ssr_trace.comp`, which has to apply it per ray:
    /// this is here so the cutoff can be reasoned about — and tested — without
    /// a GPU.
    pub fn roughness_fade(&self, roughness: f32) -> f32 {
        let cutoff = self.max_roughness.clamp(0.01, 1.0);
        let start = cutoff * 0.6;
        if roughness <= start {
            return 1.0;
        }
        if roughness >= cutoff {
            return 0.0;
        }
        let t = (roughness - start) / (cutoff - start);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mirror gets the whole reflection and anything past the cutoff gets
    /// none. The cutoff is what keeps the noise budget bounded, so a fade that
    /// let anything through past it would be spending rays on a result the
    /// environment already has.
    #[test]
    fn the_fade_spans_the_cutoff() {
        let settings = SsrSettings::default();
        assert_eq!(settings.roughness_fade(0.0), 1.0);
        assert_eq!(settings.roughness_fade(settings.max_roughness), 0.0);
        assert_eq!(settings.roughness_fade(1.0), 0.0);
    }

    /// And it has to get there smoothly, or a surface with a roughness gradient
    /// across it shows the cutoff as a line.
    #[test]
    fn the_fade_never_climbs() {
        let settings = SsrSettings::default();
        let mut previous = 1.0;
        for step in 0..=100 {
            let fade = settings.roughness_fade(step as f32 / 100.0);
            assert!(fade <= previous + 1e-6, "fade rose at {step}");
            assert!((0.0..=1.0).contains(&fade));
            previous = fade;
        }
    }

    /// The cutoff is a setting, so the fade has to follow it rather than the
    /// default it was tuned against.
    #[test]
    fn a_tighter_cutoff_moves_the_whole_fade() {
        let settings = SsrSettings {
            max_roughness: 0.2,
            ..SsrSettings::default()
        };
        assert_eq!(settings.roughness_fade(0.2), 0.0);
        assert!(settings.roughness_fade(0.15) < 1.0);
        assert_eq!(settings.roughness_fade(0.1), 1.0);
    }
}
