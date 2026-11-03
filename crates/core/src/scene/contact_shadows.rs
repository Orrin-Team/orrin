use orrin_registry::Reflect;

/// Contact shadows: a short screen-space march toward the sun, filling the gap
/// a cascade always leaves under an object.
///
/// The gap is not a bug in the cascades, it is what they are. A cascade texel
/// covers real world space — tens of centimetres in the nearest slice of a
/// hundred-metre range — and the normal-offset bias that keeps that texel from
/// self-shadowing pushes the lookup a texel's width off the surface. Everything
/// closer to the contact than that width is lit whatever the map says, so a
/// chair leg meets the floor in a shadow that starts a hand's breadth away.
/// Marching the depth buffer recovers exactly that band, because it is the one
/// scale at which screen space has more resolution than the map does.
///
/// It shadows the sun and nothing else. The march reads a depth buffer, which
/// only records what the camera can see, so everything about the settings below
/// is about failing quietly where it cannot: the ray is short, it fades with
/// distance, and it fades at the frame's edge.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ContactShadowSettings {
    /// Off drops the pass rather than marching a zero-length ray, which is why
    /// it is part of the frame's structure.
    pub enabled: bool,
    /// How far the ray travels, in metres. This is the width of the band the
    /// cascades cannot resolve, so it wants to be about a near cascade's texel
    /// and no more — a longer ray spends steps re-deriving shadows the maps
    /// already have, at a resolution that runs out the moment the occluder
    /// leaves the screen.
    pub ray_length: f32,
    /// Samples along that ray. The march has no acceleration structure, so this
    /// is resolution rather than reach: the shortest occluder it can find is one
    /// covering `ray_length / steps` of the ray.
    pub steps: u32,
    /// How far in front of the ray a surface has to be before it counts as an
    /// occluder, in metres. The floor under a grazing view is only a few
    /// millimetres of depth away from a ray skimming along it, and quantisation
    /// there is what a march without this reads as self-shadowing.
    pub bias: f32,
    /// How deep a surface is assumed to be past that, in metres. The depth
    /// buffer records a front face and nothing else, so a ray landing further
    /// behind one than this has passed through space the buffer cannot see
    /// rather than into the object that face belongs to. Measured from the
    /// bias rather than from the surface, so the two dials stay independent —
    /// a thickness under the bias would otherwise leave no window at all.
    pub thickness: f32,
    /// How dark a contact-shadowed fragment gets. One is what the march found;
    /// less is an art dial, the same one `ShadowSettings::strength` is.
    pub intensity: f32,
    /// Where the effect is gone entirely, as a radial distance from the camera
    /// in metres. Past a certain range the band this recovers is thinner than a
    /// pixel, and all a march there can produce is noise for TAA to smear.
    pub fade_distance: f32,
}

impl Default for ContactShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // About one texel of the nearest cascade at the default settings
            // (a 2048-texel map over the first slice of 100 m), which is the
            // band the maps cannot resolve.
            ray_length: 0.25,
            steps: 16,
            bias: 0.02,
            thickness: 0.2,
            intensity: 1.0,
            fade_distance: 25.0,
        }
    }
}

/// Where the fade begins, as a fraction of [`ContactShadowSettings::fade_distance`].
///
/// Not a setting: two dials for one ramp is two ways to invert it, and the only
/// thing a start distance buys over a fixed fraction is a shorter ramp — which
/// is the thing that makes the fade visible as a band on the ground.
const FADE_START: f32 = 0.75;

impl ContactShadowSettings {
    /// How much of the march survives at `view_distance` metres from the camera.
    ///
    /// Mirrored by the same smoothstep in `contact_shadows.frag`, which has to
    /// apply it per pixel; this is here so the ramp can be reasoned about — and
    /// tested — without a GPU. Radial distance rather than view-space depth, for
    /// the reason the cascade selection uses it: turning the camera should not
    /// move the fade across the ground.
    pub fn distance_fade(&self, view_distance: f32) -> f32 {
        let end = self.fade_distance.max(1e-4);
        let start = end * FADE_START;
        if view_distance <= start {
            return 1.0;
        }
        if view_distance >= end {
            return 0.0;
        }
        let t = (view_distance - start) / (end - start);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing near the camera is faded and nothing past the range survives. The
    /// far end is the one that matters: a march that kept going past it would be
    /// spending sixteen depth fetches per pixel on a band thinner than the pixel
    /// it is shading.
    #[test]
    fn the_fade_spans_the_range() {
        let settings = ContactShadowSettings::default();
        assert_eq!(settings.distance_fade(0.0), 1.0);
        assert_eq!(settings.distance_fade(settings.fade_distance), 0.0);
        assert_eq!(settings.distance_fade(1e6), 0.0);
    }

    /// And it has to get there smoothly. A hard cutoff on a floor receding from
    /// the camera is a line across it, in the one place the effect exists to
    /// look natural.
    #[test]
    fn the_fade_never_climbs() {
        let settings = ContactShadowSettings::default();
        let mut previous = 1.0;
        for step in 0..=200 {
            let distance = step as f32 * 0.25;
            let fade = settings.distance_fade(distance);
            assert!(fade <= previous + 1e-6, "fade rose at {distance} m");
            assert!((0.0..=1.0).contains(&fade));
            previous = fade;
        }
    }

    /// The range is a setting, so the whole ramp has to follow it rather than
    /// the default it was tuned against.
    #[test]
    fn a_shorter_range_moves_the_whole_ramp() {
        let settings = ContactShadowSettings {
            fade_distance: 4.0,
            ..ContactShadowSettings::default()
        };
        assert_eq!(settings.distance_fade(3.0), 1.0);
        assert!(settings.distance_fade(3.5) < 1.0);
        assert_eq!(settings.distance_fade(4.0), 0.0);
    }

    /// A zero range is a slider dragged to its end, not a division by zero.
    #[test]
    fn a_zero_range_fades_everything() {
        let settings = ContactShadowSettings {
            fade_distance: 0.0,
            ..ContactShadowSettings::default()
        };
        assert_eq!(settings.distance_fade(0.0), 1.0);
        assert_eq!(settings.distance_fade(1.0), 0.0);
    }
}
