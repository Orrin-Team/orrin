//! Which shadow casters are standing still.
//!
//! A cached shadow layer holds the depth of the casters that do not move, so
//! something has to say which those are. It is derived rather than authored: a
//! caster is static once its transform, mesh and material have gone unchanged
//! for [`STATIC_AFTER`] frames, and dynamic again the frame any of them
//! changes. An authored flag would be a promise nothing checks — a "static"
//! crate that a script moves leaves its shadow where it stood — and a scene
//! built without one would cache nothing at all.
//!
//! A mesh may still say how it casts ([`ShadowCasting`]), and what it says is
//! about cost, never about whether its shadow is right: `Static` skips the
//! wait and `Dynamic` never ends it, but a caster told it is static and then
//! moved is noticed exactly as any other is, and takes its cached depth with
//! it.
//!
//! What this module tracks beyond the flag is what each change *costs*: a
//! caster entering or leaving the static set has made every cached layer it
//! stands in out of date, so its bounds are reported for the planner to find
//! those layers by. Nothing here takes a `Device`.

use glam::Mat4;

use crate::geom::Aabb;
use crate::gfx::DrawList;
use crate::scene::ShadowCasting;

/// How many frames a caster has to hold still before it is baked into the
/// cached layers. More than one, because baking is a redraw of every layer the
/// caster stands in: something that was only between two steps — a spawned
/// projectile's first frame, a body the solver has not yet moved — must not buy
/// one on its way in and another on its way out.
pub const STATIC_AFTER: u16 = 8;

/// The bit of an instance's flags that says it is in the cached layers, and the
/// one that says it is not. One or the other, or neither for an instance that
/// casts no shadow, so a view's filter is a mask.
pub const FLAG_STATIC: u32 = 1;
pub const FLAG_DYNAMIC: u32 = 2;
/// Set on every instance, which is what the camera's view asks for: whether
/// something casts has nothing to do with whether it is seen.
pub const FLAG_VISIBLE: u32 = 4;

/// How many changes of the static set a frame reports one by one. Past it the
/// frame says only that everything changed, which is what loading a scene is.
const MAX_CHANGES: usize = 128;

/// How many moving casters a frame reports one by one. Past it every cached
/// view treats its whole tile as touched, which costs fill rate rather than a
/// test per caster per view.
const MAX_DYNAMIC: usize = 512;

#[derive(Clone, Copy)]
struct Row {
    /// Where it stood when last offered. Its own copy rather than the item's
    /// `prev_model`, which is the extraction's memory and is the caster's own
    /// position for one nobody extracted last frame — a teleport across a
    /// frame the extraction forgot would read as standing still.
    model: Mat4,
    mesh: u32,
    material: u32,
    channels: u32,
    mode: ShadowCasting,
    /// The frame this row was last offered.
    seen: u32,
    /// Frames since anything about it changed.
    still: u16,
    is_static: bool,
    /// World bounds it was baked with, while static.
    bounds: Aabb,
}

impl Default for Row {
    fn default() -> Self {
        Self {
            model: Mat4::ZERO,
            mesh: u32::MAX,
            material: u32::MAX,
            channels: 0,
            mode: ShadowCasting::Auto,
            seen: 0,
            still: 0,
            is_static: false,
            bounds: Aabb::EMPTY,
        }
    }
}

/// Every opaque caster's standing, by the row its matrices live in.
pub struct CasterTracker {
    rows: Vec<Row>,
    /// The rows that are static, so that one which stops being offered — a
    /// despawn, a hidden entity — is noticed without a sweep of every row.
    baked: Vec<u32>,
    frame: u32,
    changed: Vec<Aabb>,
    changed_everywhere: bool,
    dynamic: Vec<Aabb>,
    dynamic_everywhere: bool,
}

impl Default for CasterTracker {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            baked: Vec::new(),
            // Past the stamp a fresh row carries, so that no row reads as
            // offered on a frame before the first.
            frame: 1,
            changed: Vec::new(),
            changed_everywhere: false,
            dynamic: Vec::new(),
            dynamic_everywhere: false,
        }
    }
}

impl CasterTracker {
    /// Take this frame's casters. `bounds` is a mesh's object-space box, or
    /// `None` for one that is not measurable.
    pub fn update(&mut self, casters: DrawList<'_>, bounds: impl Fn(u32) -> Option<Aabb>) {
        self.frame += 1;
        let frame = self.frame;
        self.changed.clear();
        self.dynamic.clear();
        self.changed_everywhere = false;
        self.dynamic_everywhere = false;

        for slot in 0..casters.len() {
            let item = casters.item(slot);
            let index = item.instance as usize;
            if index >= self.rows.len() {
                self.rows.resize(index + 1, Row::default());
            }
            let row = &mut self.rows[index];
            let moved = row.model != item.model
                || row.mesh != item.mesh.0
                || row.material != item.material.0
                // A caster in other channels casts into other lights' views,
                // and one told to cast differently is another caster.
                || row.channels != item.channels
                || row.mode != item.shadows
                // A row nobody offered last frame is a new caster, whatever the
                // slot's last occupant was doing.
                || row.seen != frame - 1;
            row.seen = frame;
            row.model = item.model;
            row.mesh = item.mesh.0;
            row.material = item.material.0;
            row.channels = item.channels;
            row.mode = item.shadows;

            if moved {
                row.still = 0;
                if row.is_static {
                    row.is_static = false;
                    report(
                        &mut self.changed,
                        &mut self.changed_everywhere,
                        row.bounds,
                        MAX_CHANGES,
                    );
                }
            } else {
                row.still = row.still.saturating_add(1);
            }

            if row.is_static || row.mode == ShadowCasting::Off {
                continue;
            }
            let world = bounds(item.mesh.0).map(|bounds| bounds.transformed(&item.model));
            let settled = match row.mode {
                ShadowCasting::Static => true,
                ShadowCasting::Dynamic => false,
                _ => row.still >= STATIC_AFTER,
            };
            match world {
                Some(world) if settled => {
                    row.is_static = true;
                    row.bounds = world;
                    self.baked.push(item.instance);
                    report(
                        &mut self.changed,
                        &mut self.changed_everywhere,
                        world,
                        MAX_CHANGES,
                    );
                }
                // A caster that cannot be measured cannot be found again when
                // it leaves, so it is never baked, and it may be anywhere.
                None => self.dynamic_everywhere = true,
                Some(world) => {
                    report(
                        &mut self.dynamic,
                        &mut self.dynamic_everywhere,
                        world,
                        MAX_DYNAMIC,
                    );
                }
            }
        }

        let rows = &mut self.rows;
        let (changed, everywhere) = (&mut self.changed, &mut self.changed_everywhere);
        self.baked.retain(|&index| {
            let row = &mut rows[index as usize];
            if row.is_static && row.seen != frame {
                row.is_static = false;
                report(changed, everywhere, row.bounds, MAX_CHANGES);
            }
            row.is_static
        });
    }

    /// Call every caster dynamic again, because something changed what a
    /// caster's depth is without changing which caster it is: a mesh's
    /// geometry, a material's cutout.
    pub fn reset(&mut self) {
        for &index in &self.baked {
            self.rows[index as usize].is_static = false;
            self.rows[index as usize].still = 0;
        }
        self.baked.clear();
        self.changed_everywhere = true;
    }

    /// The flags the cull dispatch tests a view's filter against.
    pub fn flags(&self, row: u32) -> u32 {
        FLAG_VISIBLE
            | match self.rows.get(row as usize) {
                Some(row) if row.mode == ShadowCasting::Off => 0,
                Some(row) if row.is_static => FLAG_STATIC,
                _ => FLAG_DYNAMIC,
            }
    }

    /// World bounds of the casters that joined or left the static set this
    /// frame, or `None` when too many did to list, or something was
    /// [reset](Self::reset): every cached layer is then out of date.
    pub fn changed(&self) -> Option<&[Aabb]> {
        (!self.changed_everywhere).then_some(&self.changed)
    }

    /// World bounds of this frame's moving casters, or `None` when there are
    /// too many to list, or one of them cannot be measured.
    pub fn dynamic(&self) -> Option<&[Aabb]> {
        (!self.dynamic_everywhere).then_some(&self.dynamic)
    }
}

fn report(list: &mut Vec<Aabb>, everywhere: &mut bool, bounds: Aabb, limit: usize) {
    if list.len() < limit {
        list.push(bounds);
    } else {
        *everywhere = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::RenderItem;
    use crate::scene::{MaterialHandle, MeshHandle};
    use glam::{Mat4, Vec3, Vec3A};

    fn unit() -> Option<Aabb> {
        Some(Aabb {
            min: Vec3A::splat(-0.5),
            max: Vec3A::splat(0.5),
        })
    }

    /// A caster whose extraction remembers nothing of last frame, which is the
    /// case the tracker must not depend on: `prev_model` is `model`.
    fn item(instance: u32, at: Vec3) -> RenderItem {
        RenderItem {
            model: Mat4::from_translation(at),
            prev_model: Mat4::from_translation(at),
            bounds: Aabb::EMPTY,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            instance,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    fn update(tracker: &mut CasterTracker, items: &[RenderItem]) {
        let order: Vec<u32> = (0..items.len() as u32).collect();
        tracker.update(DrawList::new(items, &order), |_| unit());
    }

    /// The whole life of a caster: dynamic while new, baked once it has held
    /// still, reported where it stood the frame it moves, and reported again
    /// the frame it is gone — each report being a cached layer that has to be
    /// redrawn, so a missing one is a shadow left behind and a spare one is a
    /// redraw for nothing.
    #[test]
    fn a_caster_is_reported_when_it_settles_when_it_moves_and_when_it_goes() {
        let mut tracker = CasterTracker::default();
        let rest = Vec3::new(3.0, 0.0, 0.0);
        let still = [item(7, rest)];

        for frame in 0..STATIC_AFTER {
            update(&mut tracker, &still);
            assert_eq!(
                tracker.flags(7),
                FLAG_VISIBLE | FLAG_DYNAMIC,
                "frame {frame}"
            );
            assert!(tracker.changed().unwrap().is_empty(), "frame {frame}");
            assert_eq!(tracker.dynamic().unwrap().len(), 1);
        }
        update(&mut tracker, &still);
        assert_eq!(tracker.flags(7), FLAG_VISIBLE | FLAG_STATIC);
        assert_eq!(tracker.changed().unwrap().len(), 1, "settling bakes it in");
        assert!(tracker.dynamic().unwrap().is_empty());

        update(&mut tracker, &still);
        assert!(tracker.changed().unwrap().is_empty(), "nothing happened");

        let moved = [item(7, rest + Vec3::X)];
        update(&mut tracker, &moved);
        assert_eq!(tracker.flags(7), FLAG_VISIBLE | FLAG_DYNAMIC);
        let changed = tracker.changed().unwrap();
        assert_eq!(changed.len(), 1);
        assert!(
            changed[0].center().abs_diff_eq(Vec3A::from(rest), 1e-5),
            "the layer to redraw is where it was baked, not where it went"
        );

        // Settled again, then despawned.
        let settled = [item(7, rest + Vec3::X)];
        for _ in 0..=STATIC_AFTER {
            update(&mut tracker, &settled);
        }
        assert_eq!(tracker.flags(7), FLAG_VISIBLE | FLAG_STATIC);
        update(&mut tracker, &[]);
        assert_eq!(tracker.changed().unwrap().len(), 1, "a despawn un-bakes it");
        assert_eq!(tracker.flags(7), FLAG_VISIBLE | FLAG_DYNAMIC);
    }

    /// What a mesh says about its own shadow outranks what its movement does.
    /// Told to be static it is baked the frame it appears; told to be dynamic
    /// it never is, however long it rests; told to cast nothing it is in no
    /// shadow view's mask at all — and still in the camera's.
    #[test]
    fn a_caster_told_how_to_cast_is_taken_at_its_word() {
        let mut tracker = CasterTracker::default();
        let told = |instance, shadows| RenderItem {
            shadows,
            ..item(instance, Vec3::ZERO)
        };
        let items = [
            told(0, ShadowCasting::Static),
            told(1, ShadowCasting::Dynamic),
            told(2, ShadowCasting::Off),
        ];
        update(&mut tracker, &items);
        assert_eq!(tracker.flags(0), FLAG_VISIBLE | FLAG_STATIC);
        assert_eq!(tracker.changed().unwrap().len(), 1, "baked on sight");

        for _ in 0..STATIC_AFTER * 2 {
            update(&mut tracker, &items);
        }
        assert_eq!(tracker.flags(1), FLAG_VISIBLE | FLAG_DYNAMIC);
        assert_eq!(tracker.flags(2), FLAG_VISIBLE);
        assert_eq!(
            tracker.dynamic().unwrap().len(),
            1,
            "only the dynamic one moves"
        );
    }

    /// A slot reused by another entity standing still must not inherit the
    /// stillness of the one before it.
    #[test]
    fn a_caster_that_was_not_offered_last_frame_starts_over() {
        let mut tracker = CasterTracker::default();
        let still = [item(2, Vec3::ZERO)];
        for _ in 0..STATIC_AFTER - 1 {
            update(&mut tracker, &still);
        }
        update(&mut tracker, &[]);
        update(&mut tracker, &still);
        update(&mut tracker, &still);
        assert_eq!(tracker.flags(2), FLAG_VISIBLE | FLAG_DYNAMIC);
    }
}
