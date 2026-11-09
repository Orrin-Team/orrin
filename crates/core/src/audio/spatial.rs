//! Where a sound is, as a gain for each ear.
//!
//! Two independent terms. **Distance** is an attenuation curve over the
//! source's own minimum and maximum distance ([`attenuation`]). **Direction**
//! is a constant-power pan ([`pan`]): the left and right gains are the cosine
//! and sine of one angle, so their squares always sum to one and a source
//! circling the listener keeps its loudness rather than dipping by 3 dB as it
//! passes the front, which is what a linear pan does. The price is that a
//! centred spatial source is 3 dB below a flat one of the same volume; a flat
//! sound is simply not panned.
//!
//! Stereo cannot tell in front from behind, so nothing here tries. No HRTF, no
//! Doppler, no occlusion: each would be a stage after this one, not a change to
//! it.

use glam::{Mat4, Vec3};

use crate::scene::{Camera, Rolloff};

/// Where sound is heard from, reduced to what a stereo pan needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    pub position: Vec3,
    /// Unit length, pointing out of the right ear.
    pub right: Vec3,
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            right: Vec3::X,
        }
    }
}

impl Listener {
    /// An entity's world matrix: its position, and its local +X as the right
    /// ear, with any scale taken out.
    pub fn from_matrix(matrix: Mat4) -> Self {
        Self {
            position: matrix.w_axis.truncate(),
            right: matrix.x_axis.truncate().try_normalize().unwrap_or(Vec3::X),
        }
    }

    /// The camera the frame is drawn from, its right ear `forward × up`.
    pub fn from_camera(camera: &Camera) -> Self {
        let forward = camera.target - camera.position;
        Self {
            position: camera.position,
            right: forward.cross(camera.up).try_normalize().unwrap_or(Vec3::X),
        }
    }
}

/// A source's distance curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Falloff {
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: Rolloff,
}

impl Default for Falloff {
    fn default() -> Self {
        let source = crate::scene::AudioSource::default();
        Self {
            min_distance: source.min_distance,
            max_distance: source.max_distance,
            rolloff: source.rolloff,
        }
    }
}

impl Falloff {
    /// The distances as the curves use them: a minimum above zero and a
    /// maximum no nearer than it. An authored zero or a crossed pair is a
    /// slider dragged too far, not a request for a division by zero.
    pub fn reach(self) -> (f32, f32) {
        let min = if self.min_distance.is_finite() {
            self.min_distance.max(MIN_DISTANCE)
        } else {
            MIN_DISTANCE
        };
        let max = if self.max_distance.is_finite() {
            self.max_distance.max(min)
        } else {
            min
        };
        (min, max)
    }
}

const MIN_DISTANCE: f32 = 1e-3;

/// The gain distance alone leaves a source at, from 1 inside its minimum
/// distance.
pub fn attenuation(distance: f32, falloff: Falloff) -> f32 {
    let (min, max) = falloff.reach();
    let distance = distance.clamp(min, max);
    match falloff.rolloff {
        Rolloff::Inverse => min / distance,
        Rolloff::Linear if max > min => 1.0 - (distance - min) / (max - min),
        Rolloff::Linear => 1.0,
    }
}

/// How far out a source has fallen to `gain`, or `None` if its curve stops
/// before it gets that quiet. The inverse of [`attenuation`], for drawing it.
pub fn distance_at(gain: f32, falloff: Falloff) -> Option<f32> {
    let (min, max) = falloff.reach();
    let distance = match falloff.rolloff {
        Rolloff::Inverse if gain > 0.0 => min / gain,
        Rolloff::Linear if max > min => min + (1.0 - gain) * (max - min),
        _ => return None,
    };
    (distance <= max).then_some(distance)
}

/// Where a source sits between the ears, from -1 (left) to 1 (right).
///
/// The sine of its azimuth — except that inside the minimum distance the
/// offset is measured against that distance rather than the true one, so a
/// source passing through the listener's head slides across the middle instead
/// of snapping from one ear to the other.
pub fn pan(listener: &Listener, position: Vec3, min_distance: f32) -> f32 {
    let offset = position - listener.position;
    let reach = offset.length().max(min_distance).max(MIN_DISTANCE);
    (offset.dot(listener.right) / reach).clamp(-1.0, 1.0)
}

/// The left and right gains of a source at `position`, before its volume.
pub fn gains(listener: &Listener, position: Vec3, falloff: Falloff) -> [f32; 2] {
    let (min, _) = falloff.reach();
    let gain = attenuation(position.distance(listener.position), falloff);
    let angle = (pan(listener, position, min) + 1.0) * std::f32::consts::FRAC_PI_4;
    [angle.cos() * gain, angle.sin() * gain]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn falloff(rolloff: Rolloff) -> Falloff {
        Falloff {
            min_distance: 2.0,
            max_distance: 32.0,
            rolloff,
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn inside_the_minimum_distance_a_source_is_at_full_volume() {
        for rolloff in [Rolloff::Inverse, Rolloff::Linear] {
            assert_eq!(attenuation(0.0, falloff(rolloff)), 1.0);
            assert_eq!(attenuation(2.0, falloff(rolloff)), 1.0);
        }
    }

    /// The inverse-distance law: amplitude halves with every doubling.
    #[test]
    fn inverse_rolloff_halves_with_each_doubling_and_stops_at_the_maximum() {
        let curve = falloff(Rolloff::Inverse);
        assert!(close(attenuation(4.0, curve), 0.5));
        assert!(close(attenuation(8.0, curve), 0.25));
        assert!(close(attenuation(32.0, curve), 1.0 / 16.0));
        assert_eq!(attenuation(500.0, curve), attenuation(32.0, curve));
    }

    #[test]
    fn linear_rolloff_reaches_silence_at_the_maximum() {
        let curve = falloff(Rolloff::Linear);
        assert!(close(attenuation(17.0, curve), 0.5));
        assert_eq!(attenuation(32.0, curve), 0.0);
        assert_eq!(attenuation(100.0, curve), 0.0);
    }

    #[test]
    fn distance_at_is_where_attenuation_reaches_that_gain() {
        for rolloff in [Rolloff::Inverse, Rolloff::Linear] {
            for gain in [1.0, 0.5, 0.25, 0.1] {
                let distance = distance_at(gain, falloff(rolloff)).unwrap();
                assert!(
                    close(attenuation(distance, falloff(rolloff)), gain),
                    "{rolloff:?} {gain}"
                );
            }
        }
        assert_eq!(
            distance_at(0.01, falloff(Rolloff::Inverse)),
            None,
            "past the maximum"
        );
    }

    #[test]
    fn a_degenerate_falloff_is_still_a_number() {
        for (min, max) in [(0.0, 0.0), (10.0, 1.0), (f32::NAN, f32::INFINITY)] {
            for rolloff in [Rolloff::Inverse, Rolloff::Linear] {
                let curve = Falloff {
                    min_distance: min,
                    max_distance: max,
                    rolloff,
                };
                for distance in [0.0, 0.5, 5.0, 1e6] {
                    let gain = attenuation(distance, curve);
                    assert!(
                        (0.0..=1.0).contains(&gain),
                        "{curve:?} at {distance}: {gain}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_source_ahead_is_centred_and_one_to_the_side_is_in_that_ear() {
        let listener = Listener::default();
        assert!(close(pan(&listener, Vec3::new(0.0, 0.0, -10.0), 1.0), 0.0));
        assert!(close(pan(&listener, Vec3::new(10.0, 0.0, 0.0), 1.0), 1.0));
        assert!(close(pan(&listener, Vec3::new(-10.0, 0.0, 0.0), 1.0), -1.0));

        let [left, right] = gains(
            &listener,
            Vec3::new(0.0, 0.0, -1.0),
            falloff(Rolloff::Inverse),
        );
        assert!(close(left, std::f32::consts::FRAC_1_SQRT_2) && close(left, right));
        let [left, right] = gains(
            &listener,
            Vec3::new(2.0, 0.0, 0.0),
            falloff(Rolloff::Inverse),
        );
        assert!(close(left, 0.0) && close(right, 1.0), "{left} {right}");
    }

    /// Constant power: the squares of the two gains sum to the square of the
    /// attenuation wherever the source is.
    #[test]
    fn panning_keeps_the_power_distance_leaves() {
        let listener = Listener::default();
        let curve = falloff(Rolloff::Inverse);
        for step in 0..16 {
            let angle = step as f32 / 16.0 * std::f32::consts::TAU;
            let position = Vec3::new(angle.cos(), 0.3, angle.sin()) * 6.0;
            let [left, right] = gains(&listener, position, curve);
            let expected = attenuation(position.length(), curve);
            assert!(
                close(left * left + right * right, expected * expected),
                "at {position}: {left} {right} against {expected}"
            );
        }
    }

    #[test]
    fn a_source_passing_through_the_head_slides_across_the_middle() {
        let listener = Listener::default();
        let near = pan(&listener, Vec3::new(0.1, 0.0, 0.0), 1.0);
        assert!(close(near, 0.1), "{near}");
        assert_eq!(pan(&listener, Vec3::ZERO, 1.0), 0.0);
    }

    /// Right-handed and -Z forward: a camera at the origin looking down -Z
    /// hears +X in its right ear, and turning it to face +X moves that source
    /// to the middle.
    #[test]
    fn the_camera_hears_with_the_engine_handedness() {
        let mut camera = Camera {
            position: Vec3::ZERO,
            target: Vec3::NEG_Z,
            up: Vec3::Y,
            ..Camera::default()
        };
        let source = Vec3::new(5.0, 0.0, 0.0);
        assert!(close(
            pan(&Listener::from_camera(&camera), source, 1.0),
            1.0
        ));
        camera.target = Vec3::X;
        assert!(close(
            pan(&Listener::from_camera(&camera), source, 1.0),
            0.0
        ));

        let turned =
            Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(3.0));
        let listener = Listener::from_matrix(turned);
        assert!(close(listener.right.length(), 1.0));
        assert!(close(pan(&listener, Vec3::new(0.0, 0.0, 5.0), 1.0), -1.0));
    }
}
