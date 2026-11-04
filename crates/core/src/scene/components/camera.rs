use orrin_registry::{Reflect, Value, ValueError, take};

/// A camera the game can draw from. Where it stands and which way it looks are
/// the entity's world transform, -Z forward and +Y up like a spot light; this
/// is only the lens in front of it.
///
/// Not the engine's `scene::Camera` resource, which is the one view a frame is
/// drawn from: while the game runs, the active camera entity is copied into it
/// (see `scene::camera`), and in edit mode the editor's fly camera owns it.
///
/// Field names are the on-disk format and, as with `Light`, the unit:
/// `fov_degrees` is the *vertical* field of view in degrees, `near` and `far`
/// are metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraComponent {
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    /// Which camera draws when nothing has been selected: the highest wins.
    pub priority: i32,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            fov_degrees: 60.0,
            near: 0.1,
            far: 1000.0,
            priority: 0,
        }
    }
}

impl CameraComponent {
    /// Why this lens cannot make a projection, naming the field. Every lens
    /// that arrives from outside Rust passes through here: a zero near plane or
    /// a far plane in front of it is a matrix of infinities, and by the time
    /// that is visible it has gone through every pass that reads depth.
    pub fn check(&self) -> Result<(), ValueError> {
        let fov = self.fov_degrees;
        if !(fov > 0.0 && fov < 180.0) {
            return Err(
                ValueError::invalid("an angle between 0 and 180", format!("{fov}"))
                    .at_field("fov_degrees"),
            );
        }
        if !(self.near > 0.0 && self.near.is_finite()) {
            return Err(
                ValueError::invalid("a positive distance", format!("{}", self.near))
                    .at_field("near"),
            );
        }
        if !(self.far > self.near && self.far.is_finite()) {
            return Err(ValueError::invalid(
                "a distance beyond `near`",
                format!("{} (near is {})", self.far, self.near),
            )
            .at_field("far"));
        }
        Ok(())
    }
}

/// Written by hand rather than derived so a scene file, which is untrusted
/// input, is held to [`CameraComponent::check`] where it enters.
impl Reflect for CameraComponent {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("fov_degrees", self.fov_degrees.to_value()),
            ("near", self.near.to_value()),
            ("far", self.far.to_value()),
            ("priority", self.priority.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let camera = Self {
            fov_degrees: take(value, "fov_degrees")?,
            near: take(value, "near")?,
            far: take(value, "far")?,
            priority: take(value, "priority")?,
        };
        camera.check()?;
        Ok(camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_camera_reads_back_what_it_wrote() {
        let camera = CameraComponent {
            fov_degrees: 75.0,
            near: 0.05,
            far: 250.0,
            priority: -2,
        };
        assert_eq!(
            CameraComponent::from_value(&camera.to_value()).unwrap(),
            camera
        );
    }

    #[test]
    fn a_lens_that_cannot_project_is_refused_by_the_field_at_fault() {
        let refused = |camera: CameraComponent| {
            CameraComponent::from_value(&camera.to_value())
                .unwrap_err()
                .to_string()
        };
        let lens = CameraComponent::default();
        for (camera, field) in [
            (
                CameraComponent {
                    fov_degrees: 0.0,
                    ..lens
                },
                "fov_degrees",
            ),
            (
                CameraComponent {
                    fov_degrees: 180.0,
                    ..lens
                },
                "fov_degrees",
            ),
            (CameraComponent { near: 0.0, ..lens }, "near"),
            (
                CameraComponent {
                    near: f32::NAN,
                    ..lens
                },
                "near",
            ),
            (CameraComponent { far: 0.1, ..lens }, "far"),
            (
                CameraComponent {
                    far: f32::INFINITY,
                    ..lens
                },
                "far",
            ),
        ] {
            let error = refused(camera);
            assert!(error.contains(field), "{camera:?}: {error}");
        }
    }
}
