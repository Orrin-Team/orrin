use orrin_registry::Reflect;

/// Camera and object motion blur, described the way a shutter is.
///
/// One dial, because a shutter really is one number once the frame rate is
/// fixed. The prepass already writes how far each pixel travelled over one
/// frame; the shutter angle says what fraction of that frame the shutter was
/// open for, and therefore what fraction of that travel lands on the sensor.
///
/// That also makes the effect behave like film rather than like a slider: a
/// scene running at 30 fps blurs twice as far as the same scene at 60 fps for
/// the same angle, which is exactly what a real camera does and what a
/// frame-rate-independent "strength" would hide.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct MotionBlurSettings {
    /// Off drops the three passes rather than running them with a zero shutter,
    /// which is why it is part of the frame's structure.
    pub enabled: bool,
    /// How far round the rotating disc the opening runs, in degrees. 180° is the
    /// film standard and the one nearly every game ships; 360° is a shutter that
    /// never closes, which is the most blur the frame's motion vectors can
    /// justify. Anything past that would be inventing motion nothing measured.
    pub shutter_angle: f32,
}

impl Default for MotionBlurSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            shutter_angle: 180.0,
        }
    }
}

impl MotionBlurSettings {
    /// The fraction of a frame's motion that reaches the sensor.
    ///
    /// Clamped to a closed shutter at one end and an open one at the other: the
    /// motion vectors describe exactly one frame of travel, so a longer exposure
    /// than that is not something this buffer can answer.
    pub fn shutter_fraction(&self) -> f32 {
        (self.shutter_angle / 360.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The film standard is half the frame, and the gather centres its taps on
    /// the pixel — so 180° spans a quarter frame either side, which is what
    /// makes a 180° pan look like footage rather than like a smear.
    #[test]
    fn the_film_standard_is_half_a_frame() {
        assert_eq!(
            MotionBlurSettings {
                shutter_angle: 180.0,
                ..Default::default()
            }
            .shutter_fraction(),
            0.5,
        );
    }

    /// A shutter that never closes is the most the velocity buffer can justify,
    /// and a dragged slider must not read past it into motion nothing measured.
    #[test]
    fn the_shutter_cannot_open_past_a_whole_frame() {
        let wide_open = MotionBlurSettings {
            shutter_angle: 720.0,
            ..Default::default()
        };
        assert_eq!(wide_open.shutter_fraction(), 1.0);

        let closed = MotionBlurSettings {
            shutter_angle: -10.0,
            ..Default::default()
        };
        assert_eq!(closed.shutter_fraction(), 0.0);
    }
}
