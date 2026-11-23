//! The irradiance volumes' probe field: where each probe stands, where its
//! data lives, and which cascade a frame brings up to date.
//!
//! A probe is a point that knows how much light arrives at it from every
//! direction, and how far it can see in each (Majercik et al. 2019). A surface
//! reads the eight probes round it, each weighted by whether it can see the
//! surface, which is what keeps light from leaking through a wall. The
//! probes stand in [`CASCADES`] nested grids round the camera, the layout id
//! Tech 8 ships (Sousa, SIGGRAPH 2025): each grid is `probes` a side, the
//! finest spaced at the setting's spacing and each one after it twice the one
//! before, so a probe near the camera is a metre from the next and one across
//! a valley thirty-two.
//!
//! # Scrolling
//!
//! Each grid is snapped to its own spacing and centred on the camera, so it
//! moves a whole probe at a time and a probe never moves at all: probe `w`
//! stands at `w * spacing` for as long as the grid holds it. Where its data
//! lives is decided by `w` alone — the grid is a torus over the atlas, [`slot`]
//! is `w` modulo the grid's width — so a grid that scrolls by one probe hands
//! one slab of slots to the probes that came into it and leaves every other
//! probe's history where it was. Nothing is copied and nothing swims.
//!
//! A slot does not know on its own whether it holds the probe that is there
//! now or the one that left. Each probe's state carries a [`stamp`] of the
//! coordinate it was last written for, and a probe whose stamp disagrees is
//! read as having no data: the update starts it afresh and a surface skips
//! it, falling back to the next grid out.
//!
//! # The atlas
//!
//! Each probe is an octahedral tile of [`TILE`] texels a side with a one-texel
//! border, so a bilinear fetch at a tile's edge reads the texel across the
//! octahedron's fold rather than a neighbour's. Tiles are laid out a grid's
//! `x` and `z` along the atlas's width and its `y` and the cascade down its
//! height. One atlas of irradiance and one of the mean and mean square of
//! the distance to whatever each direction hits.
//!
//! # Which cascade a frame updates
//!
//! One a frame, so the cost is flat: [`Schedule`] picks the grid that scrolled
//! since its last update, and otherwise the one updated longest ago. A grid
//! that has been waiting [`OVERDUE`] frames is taken whatever else scrolled,
//! so one moving fast cannot starve the rest. Each update's rays are turned by
//! a rotation of their own, so successive updates of a probe look in
//! different directions and its history averages them.
//!
//! `slot`, `probe_index`, `tile` and `stamp` here and their `gi_` twins in
//! `gi.glsl` are a lock-step pair, as are the constants.

use glam::{IVec3, Mat3, Quat, UVec2, UVec3, Vec3};

pub const CASCADES: usize = 6;
/// Interior texels a side of one probe's octahedral tile.
pub const TILE: u32 = 8;
/// The tile with its one-texel border on each side.
pub const TILE_BORDER: u32 = TILE + 2;
/// The most rays a probe traces in an update: what `gi_update.comp` holds in
/// shared memory.
pub const MAX_RAYS: u32 = 64;
/// How many frames a grid may go without an update before it is taken ahead
/// of one that scrolled.
pub const OVERDUE: u64 = 2 * CASCADES as u64;

/// One grid of probes: the coordinate of its lowest probe, and the distance
/// between neighbours in metres. Probe `w` stands at `w * spacing`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascade {
    pub base: IVec3,
    pub spacing: f32,
}

impl Cascade {
    /// The grid of `probes` a side and `spacing` apart round `camera`, snapped
    /// to its own spacing.
    pub fn around(camera: Vec3, spacing: f32, probes: u32) -> Self {
        Self {
            base: (camera / spacing).floor().as_ivec3() - IVec3::splat(probes as i32 / 2),
            spacing,
        }
    }

    /// The probe at the lowest corner of the cell `point` is in, when all
    /// eight round the point are in this grid.
    pub fn cell_of(&self, point: Vec3, probes: u32) -> Option<IVec3> {
        let world = (point / self.spacing).floor().as_ivec3();
        let local = world - self.base;
        let inside =
            local.cmpge(IVec3::ZERO).all() && local.cmplt(IVec3::splat(probes as i32 - 1)).all();
        inside.then_some(world)
    }

    /// Where the probe `local` steps from the grid's lowest one keeps its data.
    pub fn slot_of(&self, local: IVec3, probes: u32) -> UVec3 {
        slot(self.base + local, probes)
    }
}

/// Every grid round `camera`, finest first.
pub fn cascades(camera: Vec3, first_spacing: f32, probes: u32) -> [Cascade; CASCADES] {
    std::array::from_fn(|level| {
        Cascade::around(camera, first_spacing * (1u32 << level) as f32, probes)
    })
}

/// Where probe `w` keeps its data in a grid of `probes` a side, whichever way
/// the grid has scrolled.
pub fn slot(world: IVec3, probes: u32) -> UVec3 {
    world.rem_euclid(IVec3::splat(probes as i32)).as_uvec3()
}

/// A slot's row in the probe state buffer.
pub fn probe_index(level: usize, slot: UVec3, probes: u32) -> u32 {
    level as u32 * probes * probes * probes + (slot.z * probes + slot.y) * probes + slot.x
}

/// The texel at the top left of a slot's tile, border included.
pub fn tile(level: usize, slot: UVec3, probes: u32) -> UVec2 {
    UVec2::new(slot.x + slot.z * probes, slot.y + level as u32 * probes) * TILE_BORDER
}

/// Both atlases' extent, in texels.
pub fn atlas_extent(probes: u32) -> [u32; 2] {
    [
        probes * probes * TILE_BORDER,
        probes * CASCADES as u32 * TILE_BORDER,
    ]
}

/// What a probe's state records of the coordinate it was written for: nine
/// bits an axis, so a slot can tell its own probe from the one that left.
pub fn stamp(world: IVec3) -> u32 {
    (world.x & 511) as u32 | ((world.y & 511) as u32) << 9 | ((world.z & 511) as u32) << 18
}

/// One frame's update: which grid, and how its rays are turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Update {
    pub level: usize,
    pub rotation: Mat3,
}

/// Which grid each frame updates. Kept between frames.
#[derive(Default)]
pub struct Schedule {
    /// Each grid as it stood when it was last updated, `None` for never.
    updated: [Option<IVec3>; CASCADES],
    /// The update each grid last had.
    last: [u64; CASCADES],
    updates: u64,
}

impl Schedule {
    /// The grid this frame updates, given where the grids stand now.
    pub fn next(&mut self, cascades: &[Cascade; CASCADES]) -> Update {
        let age = |level: usize| self.updates - self.last[level];
        let stale = |level: usize| self.updated[level] != Some(cascades[level].base);
        let oldest = |pick: &dyn Fn(usize) -> bool| {
            (0..CASCADES)
                .filter(|&level| pick(level))
                .max_by_key(|&level| (age(level), std::cmp::Reverse(level)))
        };
        let level = oldest(&|level| age(level) >= OVERDUE)
            .or_else(|| oldest(&stale))
            .or_else(|| oldest(&|_| true))
            .expect("there is always a grid");

        self.updates += 1;
        self.updated[level] = Some(cascades[level].base);
        self.last[level] = self.updates;
        Update {
            level,
            rotation: rotation(self.updates),
        }
    }
}

/// A rotation drawn uniformly from all of them (Shoemake 1992), the same one
/// for the same `n`: a capture is the same picture however often it is taken.
fn rotation(n: u64) -> Mat3 {
    let mut state = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut next = || {
        // SplitMix64, one step.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
    };
    let (u1, u2, u3) = (next(), next(), next());
    let tau = std::f32::consts::TAU;
    let (a, b) = ((1.0 - u1).sqrt(), u1.sqrt());
    Mat3::from_quat(
        Quat::from_xyzw(
            a * (tau * u2).sin(),
            a * (tau * u2).cos(),
            b * (tau * u3).sin(),
            b * (tau * u3).cos(),
        )
        .normalize(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 % 100_000) as f32 / 100_000.0
        }
        fn within(&mut self, reach: f32) -> Vec3 {
            Vec3::new(self.next(), self.next(), self.next()) * 2.0 * reach - reach
        }
    }

    /// The property scrolling rests on: a point the grid holds before and
    /// after the camera moves reads the same probes from the same tiles, and
    /// no two probes a grid holds share a tile. Cameras on both sides of every
    /// axis's origin, steps from a fraction of a probe to past a whole grid,
    /// both grid widths and every cascade.
    #[test]
    fn a_point_keeps_its_probe_and_its_tile_as_the_cascades_scroll() {
        let mut rng = Rng(0x2545_f491);
        let mut checked = 0;
        for probes in [12, 16] {
            for round in 0..60 {
                let before = rng.within(900.0);
                let after = before + rng.within([0.3, 4.0, 40.0][round % 3]);
                let (old, new) = (
                    cascades(before, 0.75, probes),
                    cascades(after, 0.75, probes),
                );
                for level in 0..CASCADES {
                    let reach = new[level].spacing * probes as f32;
                    for _ in 0..80 {
                        let point = after + rng.within(reach * 0.6);
                        let (Some(a), Some(b)) = (
                            old[level].cell_of(point, probes),
                            new[level].cell_of(point, probes),
                        ) else {
                            continue;
                        };
                        assert_eq!(a, b, "{point} changed probe as the camera moved");
                        // Each corner as the shader reaches it: by its place in
                        // the grid, which the scroll changed.
                        for corner in 0..8 {
                            let offset = IVec3::new(corner & 1, (corner >> 1) & 1, corner >> 2);
                            let s = old[level].slot_of(a - old[level].base + offset, probes);
                            let t = new[level].slot_of(b - new[level].base + offset, probes);
                            assert_eq!(tile(level, s, probes), tile(level, t, probes));
                            assert_eq!(
                                probe_index(level, s, probes),
                                probe_index(level, t, probes)
                            );
                        }
                        checked += 1;
                    }

                    let per_grid = probes * probes * probes;
                    let mut tiles = std::collections::HashSet::new();
                    let mut rows = std::collections::HashSet::new();
                    for z in 0..probes as i32 {
                        for y in 0..probes as i32 {
                            for x in 0..probes as i32 {
                                let s = new[level].slot_of(IVec3::new(x, y, z), probes);
                                assert!(tiles.insert(tile(level, s, probes)));
                                let row = probe_index(level, s, probes);
                                assert!(row / per_grid == level as u32 && rows.insert(row));
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 15_000, "only {checked} points were checked");
    }
}
