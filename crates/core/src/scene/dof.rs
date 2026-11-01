use orrin_registry::Reflect;

/// Defocus blur, described the way a lens is.
///
/// The dials here are the ones on a camera — an aperture, a focus distance, a
/// sensor — rather than a blur radius, for the reason [`HdrSettings`] is in
/// EV100 rather than in multipliers: they compose. Stopping down two stops
/// halves the circle of confusion whatever the scene is, and zooming in
/// shallows the depth of field on its own, because the focal length is
/// *derived* from the camera's `fov_y` rather than stated again here. There is
/// deliberately no second field of view to keep in step.
///
/// [`HdrSettings`]: crate::scene::HdrSettings
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct DofSettings {
    /// Off drops the four passes rather than passing a zero radius through
    /// them, which is why it is part of the frame's structure.
    pub enabled: bool,
    /// What the lens is focused on, in metres. Everything else is blurred in
    /// proportion to how far it sits from this plane.
    pub focus_distance: f32,
    /// The aperture, as the f-number engraved on a lens: f/1.4 is wide open and
    /// very shallow, f/16 is nearly everything in focus. Larger is a *smaller*
    /// opening, which is the one thing about the scale that surprises people.
    pub f_number: f32,
    /// Sensor height in millimetres; 24 is a full-frame stills camera and 14.2
    /// is Super 35. It sets the scale the whole system works at — the same
    /// f-number on a smaller sensor is a deeper image, exactly as it is in life.
    pub sensor_height: f32,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            focus_distance: 4.0,
            // Two stops down from wide open on a fast lens: shallow enough to
            // read as defocus without putting the whole scene into the near
            // field the moment it is switched on.
            f_number: 2.8,
            sensor_height: 24.0,
        }
    }
}

impl DofSettings {
    /// The focal length the camera's vertical field of view implies, in metres.
    ///
    /// This is the direction the derivation has to run. A renderer states a
    /// field of view and a real lens states a focal length, and they are the
    /// same fact seen from either end of `sensor_height`; deriving the other way
    /// would leave two numbers that could disagree about how wide the frame is.
    pub fn focal_length(&self, fov_y: f32) -> f32 {
        let sensor_height = self.sensor_height * 1e-3;
        sensor_height / (2.0 * (fov_y * 0.5).tan())
    }

    /// Everything about the circle of confusion that does not vary per pixel, so
    /// the shader is left with `coc = scale * (1 - focus_distance / depth)`.
    ///
    /// The full expression is the thin-lens one: a point at distance `D` images
    /// as a circle of diameter `|D - S| / D * f² / (N * (S - f))`. Every factor
    /// but `(D - S) / D` is constant across the frame, so it is folded here,
    /// converted from millimetres on the sensor to pixels on the screen, and
    /// halved — the gather works in radii.
    ///
    /// Returns zero where the lens cannot form an image at all: focused at or
    /// inside its own focal length, which the divisor would otherwise blow up
    /// on.
    pub fn coc_radius_scale(&self, fov_y: f32, frame_height: u32) -> f32 {
        let focal_length = self.focal_length(fov_y);
        let sensor_height = self.sensor_height * 1e-3;
        if self.focus_distance <= focal_length || self.f_number <= 0.0 || sensor_height <= 0.0 {
            return 0.0;
        }
        let diameter =
            focal_length * focal_length / (self.f_number * (self.focus_distance - focal_length));
        diameter / sensor_height * frame_height as f32 * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOV: f32 = std::f32::consts::FRAC_PI_3;
    const HEIGHT: u32 = 1080;

    /// The circle of confusion at a given distance, as the shader computes it.
    fn coc(settings: &DofSettings, depth: f32) -> f32 {
        settings.coc_radius_scale(FOV, HEIGHT) * (1.0 - settings.focus_distance / depth)
    }

    /// A 60° vertical field of view on a full-frame sensor is a 20.8mm lens.
    /// Pinning one known pair is what catches the derivation being inverted or
    /// off by the half-angle.
    #[test]
    fn the_focal_length_matches_the_field_of_view() {
        let focal_length = DofSettings::default().focal_length(FOV);
        assert!(
            (focal_length - 0.0208).abs() < 1e-4,
            "60° on 24mm should be a ~20.8mm lens, got {}mm",
            focal_length * 1e3,
        );
    }

    /// The focus plane is the definition of sharp: whatever the aperture, a
    /// point exactly at `focus_distance` must image as a point.
    #[test]
    fn the_focus_plane_is_perfectly_sharp() {
        let settings = DofSettings::default();
        assert!(coc(&settings, settings.focus_distance).abs() < 1e-6);
    }

    /// Nearer than focus is the near field and further is the far field, and the
    /// two composite differently — a near-field surface bleeds over what it
    /// occludes and a far-field one does not. The sign is what tells them apart,
    /// so it is not an internal detail.
    #[test]
    fn the_near_field_is_signed_apart_from_the_far_field() {
        let settings = DofSettings::default();
        assert!(coc(&settings, settings.focus_distance * 0.5) < 0.0);
        assert!(coc(&settings, settings.focus_distance * 2.0) > 0.0);
    }

    /// A stop is a halving of the opening, so f/2.8 must blur exactly twice as
    /// much as f/5.6 at the same distance. This is the property that makes the
    /// dial behave like a lens rather than like a slider, and the one an
    /// eyeballed curve would lose — the same reason `HdrSettings` is in stops.
    #[test]
    fn one_stop_halves_the_circle_of_confusion() {
        let wide = DofSettings {
            f_number: 2.8,
            ..Default::default()
        };
        let stopped_down = DofSettings {
            f_number: 5.6,
            ..wide
        };
        let ratio = coc(&wide, 20.0) / coc(&stopped_down, 20.0);
        assert!(
            (ratio - 2.0).abs() < 1e-4,
            "a stop changed the blur by {ratio}"
        );
    }

    /// Blur must grow with distance from focus and level off, never reverse:
    /// the far field approaches the scale as `focus / depth` vanishes, so a
    /// sign flip out there would mean distant geometry snapping back to sharp.
    #[test]
    fn the_far_field_grows_towards_a_limit() {
        let settings = DofSettings::default();
        let scale = settings.coc_radius_scale(FOV, HEIGHT);
        let near = coc(&settings, 10.0);
        let far = coc(&settings, 1000.0);
        assert!(near > 0.0 && far > near);
        assert!(far < scale && (far - scale).abs() < scale * 0.01);
    }

    /// A lens focused at or inside its own focal length forms no image, and the
    /// thin-lens divisor goes to zero there. A dragged slider reaches it, so it
    /// has to be a flat zero rather than an infinity that becomes a NaN radius.
    #[test]
    fn a_degenerate_focus_blurs_nothing() {
        let settings = DofSettings {
            focus_distance: 0.0,
            ..Default::default()
        };
        assert_eq!(settings.coc_radius_scale(FOV, HEIGHT), 0.0);

        let settings = DofSettings {
            f_number: 0.0,
            ..Default::default()
        };
        assert_eq!(settings.coc_radius_scale(FOV, HEIGHT), 0.0);
    }

    /// The same aperture on a smaller sensor is a deeper image, because the
    /// shorter lens that keeps the field of view has a shorter focal length
    /// squared in the numerator. Getting this backwards would make the sensor
    /// dial an inverted blur slider.
    #[test]
    fn a_smaller_sensor_is_deeper_at_the_same_aperture() {
        let full_frame = DofSettings::default();
        let super_35 = DofSettings {
            sensor_height: 14.2,
            ..full_frame
        };
        assert!(coc(&super_35, 20.0) < coc(&full_frame, 20.0));
    }
}
