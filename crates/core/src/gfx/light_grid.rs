//! Which punctual lights can reach which part of the world: the light lists a
//! ray's hit loops over, wherever it lands.
//!
//! [`clusters`](super::clusters) answers the same question for a fragment, by
//! cutting up the view frustum. A ray is traced to find what the frustum does
//! not hold, so its hit has no cluster to read: it needs lists keyed by where
//! the hit *is*. This is that — a clustered light list in world space, the
//! shape id built for DOOM Eternal's ray tracing and kept (Sousa, SIGGRAPH
//! 2025).
//!
//! [`CASCADES`] nested boxes round the camera, each [`CELLS`] cells a side and
//! each twice the size of the one inside it: [`FIRST_CELL`]-metre cells out to
//! sixteen metres, then four-metre cells to thirty-two, and on to cells of 256
//! metres two kilometres out. A point reads the finest cascade that contains
//! it, so a hit near the camera loops over the lights of a two-metre cube and
//! one across a valley over those of a district, which is also about what
//! each can resolve.
//!
//! Binned on the CPU and conservative, for the reasons `clusters` gives for
//! both: a light is listed in every cell its bounding sphere's box touches, so
//! a cell may list a light that misses it and never omits one that reaches it,
//! and the shader still applies each light's own range. Which is why nothing
//! here can show in the picture: the lit result is the same wherever the grid
//! stands. The cascades are snapped to their own cells all the same, so a
//! camera standing still, or moving inside one cell, bins to the same lists
//! and [`bin`](LightGrid::bin) can say that nothing changed.
//!
//! Past the last cascade a point reads no punctual lights at all. At two
//! kilometres a lamp's range has ended long before.
//!
//! What it costs is the lists of the finest cascades, where a light of range
//! `r` is listed in about `r^3` cells. Measured on one thread: sixteen lamps
//! bin in 0.04 ms, most of it the fixed walk over the cells; 256 of eight
//! metres in 0.2 ms and 0.5 MB; 256 of twenty-five metres in 1.9 ms and 3.9
//! MB. Paid only by a frame in which a light moved or the camera left a
//! two-metre cell, and a frame in which neither did compares the lights and
//! returns in under a microsecond.
//!
//! `cell_of` here and `world_light_cell` in `light_grid.glsl` are a lock-step
//! pair, as are the three grid constants.

use glam::Vec3;

use super::{MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, SceneLighting};

pub const CELLS: u32 = 16;
pub const CASCADES: usize = 8;
/// The edge of a cell in the first cascade, in metres.
pub const FIRST_CELL: f32 = 2.0;
pub const CELL_COUNT: usize = (CELLS * CELLS * CELLS) as usize * CASCADES;

/// One cascade's box: its minimum corner and the edge of one of its cells.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cascade {
    pub origin: Vec3,
    pub cell: f32,
}

impl Cascade {
    /// The cell of this cascade `point` is in, per axis, or `None` outside it.
    fn cell_at(&self, point: Vec3) -> Option<[u32; 3]> {
        let local = ((point - self.origin) / self.cell).floor();
        let inside = local.cmpge(Vec3::ZERO).all() && local.cmplt(Vec3::splat(CELLS as f32)).all();
        inside.then_some([local.x as u32, local.y as u32, local.z as u32])
    }
}

/// The cascades round `camera`, finest first, each snapped to a whole number
/// of its own cells so that it only moves when the camera leaves a cell.
pub fn cascades(camera: Vec3) -> [Cascade; CASCADES] {
    std::array::from_fn(|level| {
        let cell = FIRST_CELL * (1u32 << level) as f32;
        Cascade {
            origin: ((camera / cell).floor() - Vec3::splat((CELLS / 2) as f32)) * cell,
            cell,
        }
    })
}

fn index(level: usize, [x, y, z]: [u32; 3]) -> usize {
    level * (CELLS * CELLS * CELLS) as usize + ((z * CELLS + y) * CELLS + x) as usize
}

/// `world_light_cell` in `light_grid.glsl`: the cell a world position reads,
/// in the finest cascade that contains it, or `None` past the last.
pub fn cell_of(cascades: &[Cascade; CASCADES], point: Vec3) -> Option<usize> {
    cascades
        .iter()
        .enumerate()
        .find_map(|(level, cascade)| Some(index(level, cascade.cell_at(point)?)))
}

/// Every cell's lights. `ranges[c]` is where the cell's list starts in
/// `indices` and how long it is: the point lights' indices, then the spots',
/// with the two counts packed sixteen bits each, points low. Half a cluster's
/// four words, because there are ten times as many cells and a hit reads one
/// at random.
///
/// Kept between frames for its allocations, and for the comparison that lets a
/// frame in which nothing moved upload nothing.
#[derive(Default)]
pub struct LightGrid {
    pub cascades: [Cascade; CASCADES],
    pub ranges: Vec<[u32; 2]>,
    pub indices: Vec<u32>,
    /// What the lists were last built from: each light's kind, position and
    /// range.
    binned: Vec<(bool, Vec3, f32)>,
    spans: Vec<Span>,
    counts: Vec<[u32; 2]>,
}

/// The cells of one cascade a light's box touches.
#[derive(Clone, Copy)]
struct Span {
    light: u32,
    spot: bool,
    level: usize,
    lo: [u32; 3],
    hi: [u32; 3],
}

impl LightGrid {
    /// Bin every light in `lighting` round `camera`. Returns whether the lists
    /// differ from the ones already here, which is whether they need
    /// uploading.
    pub fn bin(&mut self, lighting: &SceneLighting, camera: Vec3) -> bool {
        let cascades = cascades(camera);
        let spheres = || {
            // No further than the light tables go: an index past the end of
            // one is a read past the end of a uniform block.
            lighting
                .point_lights
                .iter()
                .take(MAX_POINT_LIGHTS)
                .map(|light| (false, light.position, light.range))
                .chain(
                    lighting
                        .spot_lights
                        .iter()
                        .take(MAX_SPOT_LIGHTS)
                        .map(|light| (true, light.position, light.range)),
                )
        };
        if !self.ranges.is_empty()
            && cascades == self.cascades
            && spheres().eq(self.binned.iter().copied())
        {
            return false;
        }
        self.cascades = cascades;
        self.binned.clear();
        self.binned.extend(spheres());

        self.spans.clear();
        let (mut points, mut spots) = (0, 0);
        for &(spot, position, range) in &self.binned {
            let counter = if spot { &mut spots } else { &mut points };
            let light = *counter;
            *counter += 1;
            let radius = Vec3::splat(range.max(1e-4));
            for (level, cascade) in cascades.iter().enumerate() {
                let lo = ((position - radius - cascade.origin) / cascade.cell).floor();
                let hi = ((position + radius - cascade.origin) / cascade.cell).floor();
                let last = (CELLS - 1) as f32;
                if hi.cmplt(Vec3::ZERO).any() || lo.cmpgt(Vec3::splat(last)).any() {
                    continue;
                }
                let cells = |v: Vec3| {
                    let v = v.clamp(Vec3::ZERO, Vec3::splat(last));
                    [v.x as u32, v.y as u32, v.z as u32]
                };
                self.spans.push(Span {
                    light,
                    spot,
                    level,
                    lo: cells(lo),
                    hi: cells(hi),
                });
            }
        }

        self.counts.clear();
        self.counts.resize(CELL_COUNT, [0; 2]);
        for span in &self.spans {
            for_each_cell(span, |cell| self.counts[cell][span.spot as usize] += 1);
        }

        self.ranges.clear();
        self.ranges.reserve(CELL_COUNT);
        let mut cursor = 0;
        for [points, spots] in &self.counts {
            debug_assert!(*points <= 0xffff && *spots <= 0xffff);
            self.ranges.push([cursor, 0]);
            cursor += points + spots;
        }
        self.indices.clear();
        self.indices.resize(cursor as usize, 0);
        for span in &self.spans {
            let (ranges, indices) = (&mut self.ranges, &mut self.indices);
            for_each_cell(span, |cell| {
                let [offset, packed] = &mut ranges[cell];
                // Points are binned before spots, so a cell's points are
                // complete by the time its first spot is placed after them.
                let placed = (*packed & 0xffff) + (*packed >> 16);
                indices[(*offset + placed) as usize] = span.light;
                *packed += if span.spot { 1 << 16 } else { 1 };
            });
        }
        true
    }
}

fn for_each_cell(span: &Span, mut visit: impl FnMut(usize)) {
    for z in span.lo[2]..=span.hi[2] {
        for y in span.lo[1]..=span.hi[1] {
            for x in span.lo[0]..=span.hi[0] {
                visit(index(span.level, [x, y, z]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::{PointLight, SpotLight};

    fn point(position: Vec3, range: f32) -> PointLight {
        PointLight {
            id: 0,
            position,
            color: Vec3::ONE,
            candela: 100.0,
            range,
            casts_shadows: false,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    fn spot(position: Vec3, range: f32) -> SpotLight {
        SpotLight {
            id: 0,
            position,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            candela: 100.0,
            range,
            inner_cos: 0.9,
            outer_cos: 0.8,
            casts_shadows: false,
            channels: crate::scene::ALL_CHANNELS,
            shadows: Default::default(),
        }
    }

    /// A cell's two lists, unpacked as `light_grid.glsl` unpacks them.
    fn lights_of(grid: &LightGrid, cell: usize) -> (&[u32], &[u32]) {
        let [offset, packed] = grid.ranges[cell];
        let (points, spots) = (packed & 0xffff, packed >> 16);
        let list = &grid.indices[offset as usize..(offset + points + spots) as usize];
        list.split_at(points as usize)
    }

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 % 100_000) as f32 / 100_000.0
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next()
        }
        fn within(&mut self, reach: f32) -> Vec3 {
            Vec3::new(
                self.range(-reach, reach),
                self.range(-reach, reach),
                self.range(-reach, reach),
            )
        }
    }

    /// The property everything rests on: any point a light can reach is in a
    /// cell whose list names that light, whichever cascade the point falls in
    /// and wherever the camera stands in its cell. Lights are scattered from
    /// beside the camera to past the last cascade, small and large, and each
    /// point is looked up exactly as the shader would.
    #[test]
    fn every_point_a_light_reaches_is_in_a_cell_that_lists_it() {
        let mut rng = Rng(0x9e37_79b9);
        let camera = Vec3::new(13.7, -4.2, 250.9);
        let mut lighting = SceneLighting::default();
        for reach in [8.0, 40.0, 300.0, 1500.0, 2600.0] {
            for _ in 0..40 {
                let range = rng.range(0.3, reach * 0.25);
                lighting
                    .point_lights
                    .push(point(camera + rng.within(reach), range));
            }
            for _ in 0..10 {
                let range = rng.range(0.3, reach * 0.25);
                lighting
                    .spot_lights
                    .push(spot(camera + rng.within(reach), range));
            }
        }
        let mut grid = LightGrid::default();
        assert!(grid.bin(&lighting, camera));

        let spheres = lighting
            .point_lights
            .iter()
            .map(|l| (false, l.position, l.range))
            .enumerate()
            .chain(
                lighting
                    .spot_lights
                    .iter()
                    .map(|l| (true, l.position, l.range))
                    .enumerate(),
            );
        let (mut checked, mut cascades_read) = (0, [false; CASCADES]);
        for (light, (spot, position, range)) in spheres {
            for _ in 0..200 {
                let reached = position + rng.within(1.0).normalize_or_zero() * range * rng.next();
                let Some(cell) = cell_of(&grid.cascades, reached) else {
                    continue;
                };
                let (points, spots) = lights_of(&grid, cell);
                let list = if spot { spots } else { points };
                assert!(
                    list.contains(&(light as u32)),
                    "light {light} reaches {reached} but cell {cell} omits it"
                );
                cascades_read[cell / (CELLS * CELLS * CELLS) as usize] = true;
                checked += 1;
            }
        }
        assert!(checked > 30_000, "only {checked} points were checked");
        assert_eq!(cascades_read, [true; CASCADES], "a cascade was never read");
    }

    /// A camera moving inside one cell of the finest cascade moves no cascade,
    /// so the lists are the ones already there and a frame uploads nothing;
    /// leaving the cell, or a light moving, rebuilds them.
    #[test]
    fn the_lists_are_rebuilt_only_when_a_cascade_or_a_light_moves() {
        let mut lighting = SceneLighting::default();
        lighting
            .point_lights
            .push(point(Vec3::new(3.0, 1.0, -2.0), 5.0));
        let mut grid = LightGrid::default();
        assert!(grid.bin(&lighting, Vec3::new(0.1, 0.1, 0.1)));
        assert!(!grid.bin(&lighting, Vec3::new(1.9, 1.9, 1.9)));
        assert!(grid.bin(&lighting, Vec3::new(2.1, 1.9, 1.9)));
        lighting.point_lights[0].position.x += 0.5;
        assert!(grid.bin(&lighting, Vec3::new(2.1, 1.9, 1.9)));
    }
}
