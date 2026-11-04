//! Which punctual lights can reach which part of the view: the clustered light
//! lists the surface shaders loop over instead of every light in the scene.
//!
//! The view frustum is cut into [`TILES_X`] x [`TILES_Y`] screen tiles and
//! [`SLICES`] depth slices, exponentially spaced between the near and far
//! planes so a slice is about as deep as it is wide everywhere. Each light is
//! binned into every cluster its sphere of influence can touch, and a fragment
//! loops over its own cluster's list — so the per-pixel cost follows the lights
//! that actually reach the pixel rather than the number in the scene, and the
//! count stops being capped by a uniform block.
//!
//! Binned on the CPU, not in a compute pass. Hundreds of lights bin in
//! microseconds, the result is a few tens of kilobytes uploaded with the frame,
//! and the binning stays device-free — so the property the whole scheme rests
//! on, that every point a light reaches lands in a cluster listing it, is
//! asserted in CI rather than inferred from a picture. It also adds no pass,
//! which this frame's budget counts in. A compute binning would read the same
//! layout if light counts ever outgrow this.
//!
//! The test is conservative: a light's bounding sphere is boxed in view space
//! and that box projected, and a sphere crossing the near plane claims every
//! tile of its slices. A cluster may therefore list a light that misses it,
//! which costs that cluster's pixels one attenuation test; it never omits one
//! that reaches it. The shader still applies each light's own range test.
//!
//! `cluster_of` here and `light_cluster` in `shading.glsl` are a lock-step pair,
//! as are the three grid constants.

use glam::{Mat4, Vec3, Vec4Swizzles};

use super::SceneLighting;

pub const TILES_X: u32 = 16;
pub const TILES_Y: u32 = 9;
pub const SLICES: u32 = 24;
pub const CLUSTERS: usize = (TILES_X * TILES_Y * SLICES) as usize;

/// The grid's depth mapping, as the shader is handed it: slice `k` begins at
/// `near * (far / near)^(k / SLICES)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthSlicing {
    /// `SLICES / ln(far / near)`.
    pub scale: f32,
    /// `ln(near) * scale`.
    pub bias: f32,
}

impl DepthSlicing {
    pub fn new(near: f32, far: f32) -> Self {
        let scale = SLICES as f32 / (far / near).ln();
        Self {
            scale,
            bias: near.ln() * scale,
        }
    }

    /// The slice a view depth (metres along the camera's forward axis) falls
    /// in, clamped into the grid.
    pub fn slice_of(&self, view_depth: f32) -> u32 {
        let slice = (view_depth.max(1e-6).ln() * self.scale - self.bias).floor();
        slice.clamp(0.0, (SLICES - 1) as f32) as u32
    }
}

/// `light_cluster` in `shading.glsl`: the cluster a fragment at framebuffer
/// position `frag` (pixels, origin top-left) and `view_depth` reads.
pub fn cluster_of(
    frag: [f32; 2],
    extent: [u32; 2],
    view_depth: f32,
    slicing: &DepthSlicing,
) -> usize {
    let tx = ((frag[0] / extent[0] as f32) * TILES_X as f32)
        .floor()
        .clamp(0.0, (TILES_X - 1) as f32) as u32;
    let ty = ((frag[1] / extent[1] as f32) * TILES_Y as f32)
        .floor()
        .clamp(0.0, (TILES_Y - 1) as f32) as u32;
    index(tx, ty, slicing.slice_of(view_depth))
}

fn index(x: u32, y: u32, z: u32) -> usize {
    ((z * TILES_Y + y) * TILES_X + x) as usize
}

/// Every cluster's lights. `ranges[c]` is `(point offset, point count, spot
/// offset, spot count)` into `indices`, which holds light indices.
///
/// Kept between frames for its allocations, like every per-frame list here.
#[derive(Default)]
pub struct LightClusters {
    pub ranges: Vec<[u32; 4]>,
    pub indices: Vec<u32>,
    /// The cluster box each light was binned into, reused between frames.
    boxes: Vec<ClusterBox>,
    counts: Vec<[u32; 2]>,
}

#[derive(Clone, Copy)]
struct ClusterBox {
    light: u32,
    spot: bool,
    x: [u32; 2],
    y: [u32; 2],
    z: [u32; 2],
}

impl LightClusters {
    /// Bin every light in `lighting` for a camera with `view` and the
    /// unjittered `proj` it was rasterised through, between `near` and `far`.
    pub fn bin(&mut self, lighting: &SceneLighting, view: Mat4, proj: Mat4, near: f32, far: f32) {
        let slicing = DepthSlicing::new(near, far);
        self.boxes.clear();
        let spheres = lighting
            .point_lights
            .iter()
            .map(|light| (false, light.position, light.range))
            .enumerate()
            .chain(
                lighting
                    .spot_lights
                    .iter()
                    .map(|light| (true, light.position, light.range))
                    .enumerate(),
            );
        for (light, (spot, position, range)) in spheres {
            if let Some(found) =
                cluster_box(view, proj, near, far, &slicing, position, range.max(1e-4))
            {
                self.boxes.push(ClusterBox {
                    light: light as u32,
                    spot,
                    ..found
                });
            }
        }

        self.counts.clear();
        self.counts.resize(CLUSTERS, [0; 2]);
        for found in &self.boxes {
            for_each_cluster(found, |cluster| {
                self.counts[cluster][found.spot as usize] += 1
            });
        }

        self.ranges.clear();
        self.ranges.reserve(CLUSTERS);
        let mut cursor = 0;
        for [points, spots] in &self.counts {
            self.ranges.push([cursor, 0, cursor + points, 0]);
            cursor += points + spots;
        }
        self.indices.clear();
        self.indices.resize(cursor as usize, 0);
        for found in &self.boxes {
            let (ranges, indices) = (&mut self.ranges, &mut self.indices);
            for_each_cluster(found, |cluster| {
                let range = &mut ranges[cluster];
                let (offset, count) = if found.spot {
                    (range[2], &mut range[3])
                } else {
                    (range[0], &mut range[1])
                };
                indices[(offset + *count) as usize] = found.light;
                *count += 1;
            });
        }
    }
}

fn for_each_cluster(found: &ClusterBox, mut visit: impl FnMut(usize)) {
    for z in found.z[0]..=found.z[1] {
        for y in found.y[0]..=found.y[1] {
            for x in found.x[0]..=found.x[1] {
                visit(index(x, y, z));
            }
        }
    }
}

/// The clusters a sphere can touch, or `None` if it lies wholly behind the
/// near plane or past the far one.
fn cluster_box(
    view: Mat4,
    proj: Mat4,
    near: f32,
    far: f32,
    slicing: &DepthSlicing,
    position: Vec3,
    radius: f32,
) -> Option<ClusterBox> {
    let center = view.transform_point3(position);
    let depth = -center.z;
    if depth + radius < near || depth - radius > far {
        return None;
    }
    let z = [
        slicing.slice_of((depth - radius).max(near)),
        slicing.slice_of((depth + radius).min(far)),
    ];

    // A sphere reaching the near plane can cover any part of the screen, and
    // its box would have corners behind the eye that project nowhere sensible.
    let (x, y) = if depth - radius <= near {
        ([0, TILES_X - 1], [0, TILES_Y - 1])
    } else {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for corner in 0..8 {
            let offset = Vec3::new(
                if corner & 1 == 0 { -radius } else { radius },
                if corner & 2 == 0 { -radius } else { radius },
                if corner & 4 == 0 { -radius } else { radius },
            );
            let clip = proj * (center + offset).extend(1.0);
            let ndc = clip.xy() / clip.w;
            for axis in 0..2 {
                lo[axis] = lo[axis].min(ndc[axis]);
                hi[axis] = hi[axis].max(ndc[axis]);
            }
        }
        if lo[0] > 1.0 || hi[0] < -1.0 || lo[1] > 1.0 || hi[1] < -1.0 {
            return None;
        }
        let tile = |ndc: f32, tiles: u32| {
            (((ndc * 0.5 + 0.5) * tiles as f32).floor()).clamp(0.0, (tiles - 1) as f32) as u32
        };
        (
            [tile(lo[0], TILES_X), tile(hi[0], TILES_X)],
            [tile(lo[1], TILES_Y), tile(hi[1], TILES_Y)],
        )
    };
    Some(ClusterBox {
        light: 0,
        spot: false,
        x,
        y,
        z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::{PointLight, SpotLight};
    use crate::scene::Camera;

    const EXTENT: [u32; 2] = [1600, 900];

    fn camera() -> Camera {
        Camera {
            position: Vec3::new(0.0, 2.0, 6.0),
            target: Vec3::new(0.0, 0.5, 0.0),
            ..Camera::default()
        }
    }

    fn aspect() -> f32 {
        EXTENT[0] as f32 / EXTENT[1] as f32
    }

    fn point(position: Vec3, range: f32) -> PointLight {
        PointLight {
            position,
            color: Vec3::ONE,
            candela: 100.0,
            range,
            casts_shadows: false,
        }
    }

    fn lights_of(clusters: &LightClusters, cluster: usize) -> (&[u32], &[u32]) {
        let [po, pc, so, sc] = clusters.ranges[cluster];
        (
            &clusters.indices[po as usize..(po + pc) as usize],
            &clusters.indices[so as usize..(so + sc) as usize],
        )
    }

    fn bin(lighting: &SceneLighting) -> LightClusters {
        let camera = camera();
        let mut clusters = LightClusters::default();
        clusters.bin(
            lighting,
            camera.view(),
            camera.projection(aspect()),
            camera.near,
            camera.far,
        );
        clusters
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
    }

    /// The property everything rests on: any visible point a light can reach
    /// is in a cluster whose list names that light. Checked by throwing points
    /// into the spheres of lights scattered through and around the view —
    /// some straddling the near plane, some half off screen — and looking each
    /// point up exactly as the shader would.
    #[test]
    fn every_point_a_light_reaches_is_in_a_cluster_that_lists_it() {
        let mut rng = Rng(0x1234_5678);
        let mut lighting = SceneLighting::default();
        for _ in 0..200 {
            let position = Vec3::new(
                rng.range(-30.0, 30.0),
                rng.range(-5.0, 10.0),
                rng.range(-60.0, 8.0),
            );
            lighting
                .point_lights
                .push(point(position, rng.range(0.5, 12.0)));
        }
        lighting.spot_lights.push(SpotLight {
            position: Vec3::new(1.0, 3.0, 2.0),
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            candela: 100.0,
            range: 6.0,
            inner_cos: 0.9,
            outer_cos: 0.8,
            casts_shadows: false,
        });
        let clusters = bin(&lighting);

        let camera = camera();
        let (view, proj) = (camera.view(), camera.projection(aspect()));
        let slicing = DepthSlicing::new(camera.near, camera.far);
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
        let mut checked = 0;
        for (index, (spot, position, range)) in spheres {
            for _ in 0..200 {
                let dir = Vec3::new(
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                );
                let p = position + dir.normalize_or_zero() * range * rng.next();
                let depth = -view.transform_point3(p).z;
                let ndc = proj.project_point3(view.transform_point3(p));
                if depth < camera.near
                    || depth > camera.far
                    || ndc.x.abs() > 1.0
                    || ndc.y.abs() > 1.0
                {
                    continue;
                }
                let frag = [
                    (ndc.x * 0.5 + 0.5) * EXTENT[0] as f32,
                    (ndc.y * 0.5 + 0.5) * EXTENT[1] as f32,
                ];
                let cluster = cluster_of(frag, EXTENT, depth, &slicing);
                let (points, spots) = lights_of(&clusters, cluster);
                let list = if spot { spots } else { points };
                assert!(
                    list.contains(&(index as u32)),
                    "light {index} reaches {p} but cluster {cluster} omits it"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 5_000,
            "only {checked} visible points were checked"
        );
    }

    /// The other half of the point: a light far off to one side is not on
    /// the far side's list, so the pixels there do not pay for it.
    #[test]
    fn a_small_light_is_listed_only_near_where_it_is() {
        let mut lighting = SceneLighting::default();
        lighting
            .point_lights
            .push(point(Vec3::new(-6.0, 0.5, 0.0), 1.0));
        let clusters = bin(&lighting);
        let listed = (0..CLUSTERS)
            .filter(|&c| !lights_of(&clusters, c).0.is_empty())
            .count();
        assert!(listed > 0);
        assert!(
            listed < CLUSTERS / 50,
            "{listed} of {CLUSTERS} clusters list a one-metre light"
        );
        let right_edge = cluster_of(
            [EXTENT[0] as f32 - 1.0, EXTENT[1] as f32 * 0.5],
            EXTENT,
            6.0,
            &DepthSlicing::new(0.1, 1000.0),
        );
        assert!(lights_of(&clusters, right_edge).0.is_empty());
    }

    #[test]
    fn a_light_behind_the_camera_is_listed_nowhere() {
        let mut lighting = SceneLighting::default();
        lighting
            .point_lights
            .push(point(camera().position + Vec3::new(0.0, 0.0, 20.0), 2.0));
        let clusters = bin(&lighting);
        assert!(clusters.indices.is_empty());
    }

    /// The slices tile the whole depth range in order, so the lookup and the
    /// binning cannot disagree about which slice a depth is in.
    #[test]
    fn the_slices_run_from_near_to_far_in_order() {
        let slicing = DepthSlicing::new(0.1, 1000.0);
        assert_eq!(slicing.slice_of(0.1), 0);
        assert_eq!(slicing.slice_of(999.0), SLICES - 1);
        let mut last = 0;
        for i in 0..1000 {
            let slice = slicing.slice_of(0.1 * 10_000f32.powf(i as f32 / 1000.0));
            assert!(slice >= last);
            last = slice;
        }
    }
}
