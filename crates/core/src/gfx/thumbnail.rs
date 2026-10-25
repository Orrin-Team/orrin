//! Small still renders of one mesh or one material, for the editor's asset
//! slots and pickers.
//!
//! A CPU rasteriser rather than a pass in the render graph, on purpose. A
//! thumbnail is rendered once per asset and then reused for the session, so
//! what matters is that it costs nothing per frame and needs nothing the frame
//! has — no device, no descriptor, no graph node to keep in step with the real
//! passes. It is also what lets a mesh's thumbnail be rendered at import, the
//! one moment its vertices are still on the CPU, and lets CI with no GPU test
//! it like any other function.
//!
//! One renderer, two subjects: a mesh in neutral clay, and a material on a
//! sphere. Both go through the same triangle loop, camera and light rig, so a
//! mesh and a material thumbnail sit together in a list as one family.
//!
//! What it deliberately does not do: normal maps, alpha, clearcoat, sheen or
//! any of the layered lobes. A thumbnail answers "which one is this?", and
//! base colour, metal and roughness answer it; the rest would be cost spent on
//! a 64-pixel image that could not show it.

use glam::{Vec2, Vec3};

/// The edge of every thumbnail, in pixels. Drawn at 36 points in the editor,
/// so 64 is sharp up to a 1.75 display scale: a mesh's is 8 KB, a material's
/// 16 KB.
pub const SIZE: u32 = 64;

/// Samples per pixel along each axis. Four per pixel is what turns a
/// rasteriser's staircase into an edge at this size.
const SUPERSAMPLE: u32 = 2;

/// A rendered thumbnail. Colour is sRGB-encoded and *not* premultiplied.
#[derive(Clone, Debug, PartialEq)]
pub struct Thumbnail {
    pub size: u32,
    pub pixels: Pixels,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    /// One grey and one coverage byte per pixel: a mesh in clay has no colour
    /// to store, and a model brings meshes by the thousand.
    GreyAlpha(Vec<u8>),
    Rgba(Vec<u8>),
}

impl Thumbnail {
    /// `[r, g, b, a]` at `(x, y)` from the top-left.
    pub fn rgba(&self, x: u32, y: u32) -> [u8; 4] {
        let index = (y * self.size + x) as usize;
        match &self.pixels {
            Pixels::GreyAlpha(px) => {
                let [l, a] = [px[2 * index], px[2 * index + 1]];
                [l, l, l, a]
            }
            Pixels::Rgba(px) => [
                px[4 * index],
                px[4 * index + 1],
                px[4 * index + 2],
                px[4 * index + 3],
            ],
        }
    }

    /// What this thumbnail holds in memory, for the budget it is kept to.
    pub fn bytes(&self) -> usize {
        match &self.pixels {
            Pixels::GreyAlpha(px) | Pixels::Rgba(px) => px.len(),
        }
    }
}

/// The few material parameters a thumbnail can show. Mirrors the uploaded
/// material, which the CPU does not otherwise keep.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    /// Linear.
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
    /// In nits, as the material declares it.
    pub emissive: Vec3,
    /// A tiny copy of the albedo map, so a textured material's sphere shows
    /// the texture's colours rather than the white its factor usually is.
    pub albedo: Option<Albedo>,
}

impl Default for Surface {
    fn default() -> Self {
        Self {
            base_color: Vec3::splat(0.8),
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            albedo: None,
        }
    }
}

/// An albedo map reduced to a few texels, sRGB-encoded like its source.
#[derive(Clone, Debug, PartialEq)]
pub struct Albedo {
    size: u32,
    texels: Vec<[u8; 3]>,
}

impl Albedo {
    /// The edge of the reduced map. A sphere at 64 pixels shows about this
    /// much of a texture's structure.
    pub const SIZE: u32 = 16;

    /// Box-filter `rgba` (tightly packed RGBA8, `width` x `height`) down to
    /// [`Self::SIZE`] square. `None` for an image with no pixels.
    ///
    /// Averaged in the stored sRGB values rather than in linear light: at this
    /// size the difference is a shade of a colour nobody will compare, and it
    /// keeps a 4K map's reduction to one pass of integer adds.
    pub fn reduce(rgba: &[u8], width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            return None;
        }
        let size = Self::SIZE;
        let mut sums = vec![[0u64; 4]; (size * size) as usize];
        for y in 0..height {
            let ty = (y * size / height) as usize;
            for x in 0..width {
                let tx = (x * size / width) as usize;
                let i = ((y * width + x) * 4) as usize;
                let cell = &mut sums[ty * size as usize + tx];
                cell[0] += rgba[i] as u64;
                cell[1] += rgba[i + 1] as u64;
                cell[2] += rgba[i + 2] as u64;
                cell[3] += 1;
            }
        }
        let texels = sums
            .into_iter()
            .map(|[r, g, b, n]| {
                let n = n.max(1);
                [(r / n) as u8, (g / n) as u8, (b / n) as u8]
            })
            .collect();
        Some(Self { size, texels })
    }

    /// Linear colour at `uv`, nearest texel, wrapping like a repeat sampler.
    fn sample(&self, uv: Vec2) -> Vec3 {
        let size = self.size as f32;
        let x = (uv.x.rem_euclid(1.0) * size) as u32 % self.size;
        let y = (uv.y.rem_euclid(1.0) * size) as u32 % self.size;
        let [r, g, b] = self.texels[(y * self.size + x) as usize];
        Vec3::new(decode(r), decode(g), decode(b))
    }
}

fn decode(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn encode(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let c = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0 + 0.5) as u8
}

/// The clay a mesh is shown in: a light grey dielectric, rough enough that the
/// form reads from the shading rather than from a highlight.
fn clay() -> Surface {
    Surface {
        base_color: Vec3::splat(0.42),
        metallic: 0.0,
        roughness: 0.65,
        emissive: Vec3::ZERO,
        albedo: None,
    }
}

/// A mesh, framed to fill the square from a three-quarter view above and to
/// the front-right, in clay.
pub fn mesh(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u32]) -> Thumbnail {
    let samples = rasterise(positions, normals, None, indices);
    let rgba = resolve(&samples, &clay());
    let pixels = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[grey, _, _, alpha]| [grey, alpha])
        .collect();
    Thumbnail {
        size: SIZE,
        pixels: Pixels::GreyAlpha(pixels),
    }
}

/// A material on a sphere, from the same view and under the same lights.
pub fn material(surface: &Surface) -> Thumbnail {
    let sphere = sphere();
    let samples = rasterise(
        &sphere.positions,
        &sphere.positions,
        Some(&sphere.uvs),
        &sphere.indices,
    );
    Thumbnail {
        size: SIZE,
        pixels: Pixels::Rgba(resolve(&samples, surface)),
    }
}

/// The rasterised image before shading: per sample, the view-space normal and
/// texture coordinate of the nearest surface, or nothing.
struct Samples {
    edge: u32,
    hits: Vec<Option<(Vec3, Vec2)>>,
}

/// The view every thumbnail shares, as an orthonormal basis: from the front
/// right and above, the three-quarter view in which a box shows three faces
/// and a sphere its terminator. `toward` points from the subject at the eye.
struct View {
    right: Vec3,
    up: Vec3,
    toward: Vec3,
}

impl View {
    fn new() -> Self {
        let toward = Vec3::new(0.6, 0.8, 1.0).normalize();
        let right = Vec3::Y.cross(toward).normalize();
        let up = toward.cross(right);
        Self { right, up, toward }
    }

    fn to_view(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.right), v.dot(self.up), v.dot(self.toward))
    }
}

/// How much of the frame the subject's larger extent fills, leaving a margin
/// so a thumbnail never reads as cropped.
const FILL: f32 = 0.86;

/// Orthographic: a thumbnail has no viewer distance worth simulating, and a
/// perspective fit has to guess a near plane for every mesh it is handed.
fn rasterise(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
) -> Samples {
    let edge = SIZE * SUPERSAMPLE;
    let mut samples = Samples {
        edge,
        hits: vec![None; (edge * edge) as usize],
    };
    let view = View::new();
    let projected: Vec<Vec3> = positions
        .iter()
        .map(|&p| view.to_view(Vec3::from(p)))
        .collect();

    // Framed on the vertices indices actually reach, which for an imported
    // primitive is all of them and for a hand-built one need not be.
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for &index in indices {
        let Some(p) = projected.get(index as usize) else {
            continue;
        };
        lo = lo.min(p.truncate());
        hi = hi.max(p.truncate());
    }
    let extent = (hi - lo).max_element();
    if !extent.is_finite() || extent <= f32::EPSILON {
        return samples;
    }
    let centre = (lo + hi) * 0.5;
    let half = edge as f32 * 0.5;
    let scale = half * FILL * 2.0 / extent;
    let screen = |p: Vec3| {
        Vec3::new(
            (p.x - centre.x) * scale + half,
            half - (p.y - centre.y) * scale,
            p.z,
        )
    };

    let mut depth = vec![f32::NEG_INFINITY; (edge * edge) as usize];
    for triangle in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = triangle.map(|i| i as usize);
        let (Some(&p0), Some(&p1), Some(&p2)) =
            (projected.get(i0), projected.get(i1), projected.get(i2))
        else {
            continue;
        };
        let [a, b, c] = [screen(p0), screen(p1), screen(p2)];
        let area = cross(b - a, c - a);
        if area.abs() < 1e-8 {
            continue;
        }

        let x0 = a.x.min(b.x).min(c.x).floor().max(0.0) as u32;
        let y0 = a.y.min(b.y).min(c.y).floor().max(0.0) as u32;
        let x1 = (a.x.max(b.x).max(c.x).ceil() as i64).clamp(0, edge as i64) as u32;
        let y1 = (a.y.max(b.y).max(c.y).ceil() as i64).clamp(0, edge as i64) as u32;

        let face = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        let normal = |i: usize| {
            normals
                .get(i)
                .map(|&n| view.to_view(Vec3::from(n)))
                .filter(|n| n.length_squared() > 0.0)
                .unwrap_or(face)
        };
        let [n0, n1, n2] = [normal(i0), normal(i1), normal(i2)];
        let uv = |i: usize| {
            uvs.and_then(|uvs| uvs.get(i))
                .map_or(Vec2::ZERO, |&uv| Vec2::from(uv))
        };
        let [t0, t1, t2] = [uv(i0), uv(i1), uv(i2)];

        for y in y0..y1 {
            for x in x0..x1 {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let w0 = cross(c - b, p - b) / area;
                let w1 = cross(a - c, p - c) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * a.z + w1 * b.z + w2 * c.z;
                let slot = (y * edge + x) as usize;
                if z <= depth[slot] {
                    continue;
                }
                depth[slot] = z;
                let mut n = (n0 * w0 + n1 * w1 + n2 * w2).normalize_or_zero();
                // Double-sided: a face seen from behind shows its front's shading.
                if n.z < 0.0 {
                    n = -n;
                }
                samples.hits[slot] = Some((n, t0 * w0 + t1 * w1 + t2 * w2));
            }
        }
    }
    samples
}

/// The z of the 2-D cross product: twice the signed area of the triangle the
/// two vectors span, on screen.
fn cross(a: Vec3, b: Vec3) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Box-filter down to [`SIZE`], coverage into alpha, and shade once per pixel.
///
/// Shaded once at the samples' mean normal rather than once per sample: the
/// samples exist to antialias the silhouette, which coverage already does, and
/// shading was three quarters of a thumbnail's cost for a difference no 64-pixel
/// image shows.
fn resolve(samples: &Samples, surface: &Surface) -> Vec<u8> {
    let mut out = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut normal = Vec3::ZERO;
            let mut uv = Vec2::ZERO;
            let mut covered = 0u32;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let slot =
                        ((y * SUPERSAMPLE + sy) * samples.edge + x * SUPERSAMPLE + sx) as usize;
                    if let Some((n, t)) = samples.hits[slot] {
                        normal += n;
                        uv += t;
                        covered += 1;
                    }
                }
            }
            if covered == 0 {
                continue;
            }
            let normal = normal.try_normalize().unwrap_or(Vec3::Z);
            let colour = tonemap(shade(surface, normal, uv / covered as f32));
            let i = ((y * SIZE + x) * 4) as usize;
            out[i] = encode(colour.x);
            out[i + 1] = encode(colour.y);
            out[i + 2] = encode(colour.z);
            out[i + 3] = ((covered * 255) / (SUPERSAMPLE * SUPERSAMPLE)) as u8;
        }
    }
    out
}

/// The rig, in view space (x right, y up, z toward the eye): a warm key from
/// the upper left, a cool fill from the right and a rim from behind, which is
/// the three-light setup that reads form at a glance.
const LIGHTS: [(Vec3, Vec3); 3] = [
    (Vec3::new(-0.55, 0.65, 0.55), Vec3::new(1.9, 1.85, 1.7)),
    (Vec3::new(0.75, 0.05, 0.6), Vec3::new(0.10, 0.12, 0.15)),
    (Vec3::new(0.3, 0.45, -0.85), Vec3::new(0.6, 0.6, 0.65)),
];

const SKY: Vec3 = Vec3::new(0.16, 0.17, 0.20);
const GROUND: Vec3 = Vec3::new(0.02, 0.02, 0.02);

/// Nits that map to the top of the thumbnail's range. A thumbnail has no
/// exposure to meter, so emission is scaled against a bright bulb's surface
/// instead: anything that bright reads as glowing, anything far dimmer barely.
const EMISSIVE_WHITE: f32 = 200.0;

/// The light arriving from `direction`, for the diffuse fill: a dim sky above
/// the horizon, a dark floor below.
fn environment(direction: Vec3) -> Vec3 {
    GROUND.lerp(SKY, (direction.y * 0.5 + 0.5).clamp(0.0, 1.0))
}

/// What a reflection sees, which is brighter than what lights the diffuse: a
/// metal has nothing *but* its reflections, and a dim studio turns gold into
/// olive. A bright upper hemisphere, a dark floor, and a softbox where the key
/// light is, widened by roughness so a rough metal shows a broad sheen rather
/// than a pin of light.
fn studio(direction: Vec3, roughness: f32) -> Vec3 {
    let horizon = smoothstep(-0.15, 0.45, direction.y);
    let room = Vec3::splat(0.03).lerp(Vec3::new(0.62, 0.66, 0.72), horizon);
    let key = LIGHTS[0].0.normalize();
    let width = 0.12 + 0.6 * roughness;
    room + Vec3::splat(2.2) * smoothstep(1.0 - width, 1.0, direction.dot(key))
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear radiance leaving `n` toward the eye. Lambert plus a normalised
/// Blinn-Phong lobe per light, and the studio reflected through Schlick's
/// Fresnel — not the engine's BRDF, but the same parameters pulling in the
/// same directions, which is all a thumbnail has to agree on.
fn shade(surface: &Surface, n: Vec3, uv: Vec2) -> Vec3 {
    let albedo = surface.base_color
        * surface
            .albedo
            .as_ref()
            .map_or(Vec3::ONE, |albedo| albedo.sample(uv));
    let metallic = surface.metallic.clamp(0.0, 1.0);
    let roughness = surface.roughness.clamp(0.04, 1.0);
    let alpha = roughness * roughness;
    let shininess = (2.0 / (alpha * alpha) - 2.0).clamp(1.0, 4096.0);
    let f0 = Vec3::splat(0.04).lerp(albedo, metallic);
    let diffuse = albedo * (1.0 - metallic);

    let v = Vec3::Z;
    let nv = n.dot(v).max(1e-3);
    let mut colour = diffuse * environment(n);
    for (direction, intensity) in LIGHTS {
        let l = direction.normalize();
        let nl = n.dot(l);
        if nl <= 0.0 {
            continue;
        }
        let h = (l + v).normalize();
        let specular = f0 * (shininess + 8.0) / 8.0 * n.dot(h).max(0.0).powf(shininess);
        colour += intensity * nl * (diffuse + specular);
    }

    let fresnel = f0 + (Vec3::ONE - f0) * (1.0 - nv).powi(5);
    let reflected = 2.0 * nv * n - v;
    colour += fresnel * studio(reflected, roughness);

    colour + surface.emissive / EMISSIVE_WHITE
}

/// Narkowicz's fit of the ACES curve: a shoulder rather than a clip, so a
/// highlight on chrome stays a highlight instead of a flat white patch.
fn tonemap(c: Vec3) -> Vec3 {
    let f = |x: f32| (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

struct Sphere {
    positions: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// The unit sphere every material is shown on, built once. Its positions are
/// its normals, which is why `material` passes them twice.
fn sphere() -> &'static Sphere {
    static SPHERE: std::sync::OnceLock<Sphere> = std::sync::OnceLock::new();
    SPHERE.get_or_init(|| {
        const SEGMENTS: u32 = 48;
        const RINGS: u32 = 24;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        for ring in 0..=RINGS {
            let v = ring as f32 / RINGS as f32;
            let theta = v * std::f32::consts::PI;
            for segment in 0..=SEGMENTS {
                let u = segment as f32 / SEGMENTS as f32;
                let phi = u * std::f32::consts::TAU;
                positions.push([
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                ]);
                // Twice round the equator, so a texture's detail is at a scale
                // a 64-pixel sphere can still show.
                uvs.push([u * 2.0, v]);
            }
        }
        let mut indices = Vec::new();
        let stride = SEGMENTS + 1;
        for ring in 0..RINGS {
            for segment in 0..SEGMENTS {
                let a = ring * stride + segment;
                let b = a + stride;
                indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Sphere {
            positions,
            uvs,
            indices,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::CpuMesh;

    fn of(mesh: &CpuMesh) -> Thumbnail {
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.normal).collect();
        super::mesh(&positions, &normals, &mesh.indices)
    }

    fn coverage(thumbnail: &Thumbnail) -> f32 {
        let mut covered = 0.0;
        for y in 0..thumbnail.size {
            for x in 0..thumbnail.size {
                covered += thumbnail.rgba(x, y)[3] as f32 / 255.0;
            }
        }
        covered / (thumbnail.size * thumbnail.size) as f32
    }

    /// The mean of `[r, g, b]` over the covered pixels inside `region` (x0, y0,
    /// x1, y1, exclusive), weighted by coverage.
    fn mean(thumbnail: &Thumbnail, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut weight = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                let [r, g, b, a] = thumbnail.rgba(x, y);
                let a = a as f32 / 255.0;
                sum += Vec3::new(r as f32, g as f32, b as f32) * a;
                weight += a;
            }
        }
        sum / weight.max(1e-6)
    }

    /// Framed, not lost: the subject fills a good part of the square and
    /// leaves its corners empty — a thumbnail of a speck, or of a crop, says
    /// nothing about the asset.
    #[test]
    fn a_mesh_fills_the_frame_without_touching_its_corners() {
        let thumbnail = of(&CpuMesh::cube());
        let filled = coverage(&thumbnail);
        assert!(
            filled > 0.35,
            "the cube covers only {filled:.2} of the frame"
        );
        let last = SIZE - 1;
        for (x, y) in [(0, 0), (last, 0), (0, last), (last, last)] {
            assert_eq!(thumbnail.rgba(x, y)[3], 0, "corner ({x}, {y}) is covered");
        }
    }

    /// A plane has no height at all, which is the case a fit to the bounds'
    /// largest extent along one axis divides by zero on.
    #[test]
    fn a_flat_mesh_is_still_framed() {
        let filled = coverage(&of(&CpuMesh::plane()));
        assert!(filled > 0.2, "the plane covers only {filled:.2}");
    }

    /// An empty or degenerate mesh is a blank thumbnail, not a panic: the
    /// importer renders whatever a file brings.
    #[test]
    fn an_empty_mesh_renders_nothing() {
        let thumbnail = super::mesh(&[], &[], &[]);
        assert_eq!(coverage(&thumbnail), 0.0);
        let point = super::mesh(&[[1.0, 1.0, 1.0]; 3], &[[0.0, 1.0, 0.0]; 3], &[0, 1, 2]);
        assert_eq!(point.size, SIZE);
    }

    /// The key light is above and to the left, so the sphere's upper left is
    /// brighter than its lower right — shading, not a flat disc.
    #[test]
    fn a_sphere_is_lit_from_the_upper_left() {
        let thumbnail = material(&Surface::default());
        let half = SIZE / 2;
        let lit = mean(&thumbnail, (0, 0, half, half));
        let shade = mean(&thumbnail, (half, half, SIZE, SIZE));
        assert!(
            lit.element_sum() > shade.element_sum() * 1.3,
            "lit {lit} vs shade {shade}"
        );
    }

    /// The point of a material thumbnail: its colour reads as the material's.
    #[test]
    fn a_red_material_renders_red() {
        let thumbnail = material(&Surface {
            base_color: Vec3::new(0.8, 0.05, 0.05),
            ..Surface::default()
        });
        let colour = mean(&thumbnail, (0, 0, SIZE, SIZE));
        assert!(
            colour.x > colour.y * 2.0 && colour.x > colour.z * 2.0,
            "{colour}"
        );
    }

    /// A metal has no diffuse lobe; its colour is in its reflections. A rough
    /// dielectric and a polished metal of the same base colour must not render
    /// alike, or the thumbnail cannot tell chrome from paint.
    #[test]
    fn metal_and_paint_render_differently() {
        let paint = material(&Surface {
            base_color: Vec3::splat(0.8),
            roughness: 0.9,
            ..Surface::default()
        });
        let chrome = material(&Surface {
            base_color: Vec3::splat(0.8),
            metallic: 1.0,
            roughness: 0.1,
            ..Surface::default()
        });
        let contrast = |t: &Thumbnail| {
            let mut lo = 255u8;
            let mut hi = 0u8;
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let [r, _, _, a] = t.rgba(x, y);
                    if a == 255 {
                        lo = lo.min(r);
                        hi = hi.max(r);
                    }
                }
            }
            hi.saturating_sub(lo)
        };
        assert_ne!(paint, chrome);
        assert!(
            contrast(&chrome) > contrast(&paint),
            "chrome should be the higher-contrast of the two"
        );
    }

    /// An emissive material glows on its shadowed side too.
    #[test]
    fn emission_lights_the_shadowed_side() {
        let dark = material(&Surface {
            base_color: Vec3::splat(0.05),
            ..Surface::default()
        });
        let glowing = material(&Surface {
            base_color: Vec3::splat(0.05),
            emissive: Vec3::new(400.0, 400.0, 400.0),
            ..Surface::default()
        });
        let half = SIZE / 2;
        let region = (half, half, SIZE - 4, SIZE - 4);
        assert!(mean(&glowing, region).element_sum() > mean(&dark, region).element_sum() + 60.0);
    }

    /// A textured material's sphere shows the texture's colour, multiplied by
    /// the factor, as the real shader does.
    #[test]
    fn an_albedo_map_tints_the_sphere() {
        let green: Vec<u8> = [30, 200, 40, 255].repeat(64 * 64);
        let albedo = Albedo::reduce(&green, 64, 64).expect("reduces");
        let thumbnail = material(&Surface {
            base_color: Vec3::ONE,
            albedo: Some(albedo),
            ..Surface::default()
        });
        let colour = mean(&thumbnail, (0, 0, SIZE, SIZE));
        assert!(
            colour.y > colour.x * 1.5 && colour.y > colour.z * 1.5,
            "{colour}"
        );
    }

    #[test]
    fn reducing_an_albedo_keeps_its_average_colour() {
        // Left half black, right half white: the reduction must average to
        // mid-grey, not pick one side.
        let mut rgba = Vec::new();
        for _y in 0..32 {
            for x in 0..32 {
                let v = if x < 16 { 0 } else { 255 };
                rgba.extend([v, v, v, 255]);
            }
        }
        let albedo = Albedo::reduce(&rgba, 32, 32).unwrap();
        let mean: f32 =
            albedo.texels.iter().map(|t| t[0] as f32).sum::<f32>() / albedo.texels.len() as f32;
        assert!((mean - 127.5).abs() < 2.0, "{mean}");
        assert!(Albedo::reduce(&[], 0, 0).is_none());
    }

    /// Thumbnails are rendered by the thousand at import: the same input must
    /// give the same bytes, or the cache could not be trusted, and no frame
    /// may pay for a mesh that is rendered again.
    #[test]
    fn rendering_is_deterministic() {
        let sphere = CpuMesh::sphere(32, 16);
        assert_eq!(of(&sphere), of(&sphere));
        assert_eq!(material(&Surface::default()), material(&Surface::default()));
    }

    /// The budget that keeps import fast: a mesh the size of the library's
    /// sphere renders in well under a millisecond in a release build. Debug is
    /// far slower, so the bound here only catches an order-of-magnitude slip.
    #[test]
    fn a_thumbnail_costs_well_under_a_frame() {
        let sphere = CpuMesh::sphere(32, 16);
        let start = std::time::Instant::now();
        for _ in 0..10 {
            std::hint::black_box(of(&sphere));
            std::hint::black_box(material(&Surface::default()));
        }
        let each = start.elapsed() / 20;
        assert!(
            each < std::time::Duration::from_millis(20),
            "{each:?} per thumbnail"
        );
    }

    /// Drawn at 36 points, a thumbnail has to carry 54 pixels for a 1.5x
    /// display without being stretched.
    #[test]
    fn a_thumbnail_is_sharp_on_a_high_density_display() {
        assert!(SIZE as f32 >= 36.0 * 1.5);
    }

    #[test]
    fn a_mesh_thumbnail_is_two_bytes_a_pixel() {
        assert_eq!(of(&CpuMesh::cube()).bytes(), (SIZE * SIZE * 2) as usize);
        assert_eq!(
            material(&Surface::default()).bytes(),
            (SIZE * SIZE * 4) as usize
        );
    }
}
