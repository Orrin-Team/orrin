// One reflection ray: its direction drawn from the GGX visible-normal lobe, its
// march through the depth pyramid, and the radiance it finds.
//
// Shared by the two passes that trace. `ssr_trace.comp` traces one ray per 2x2
// quad for every surface the reflections reach; `ssr_temporal.comp` traces the
// other three of a near-mirror's quad itself, at the pixel that needs it, so a
// mirror's reflection is as sharp as the frame. One function, so the two rates
// are the same estimator and a roughness gradient that crosses from one to the
// other changes how often a ray is traced, never what a ray finds.
//
// The including shader declares, by these names: the `Ssr` block as `ssr`,
// `u_hiz` (the nearest-depth pyramid from level 1 up), `u_depth`, `u_source` (the lit frame's
// pyramid, based at half resolution), `u_source_full` (the lit frame itself)
// and `u_blue_noise`.

#ifndef ORRIN_SSR_MARCH_GLSL
#define ORRIN_SSR_MARCH_GLSL

#include "depth.glsl"

const float PI = 3.14159265359;

// Below the first, a surface is a mirror: its ray is the exact reflection, the
// same every frame, so it is the answer rather than a sample of it and keeps
// no history. A lobe of 0.05 is under a fifth of a degree wide, below what the
// frame resolves at any distance on screen, so drawing from it would add
// grain and nothing else. From there up a ray is drawn from the lobe, and
// every drawn ray is accumulated. Up to the second, a pixel still traces its
// own ray beside the quarter-rate gather, for sharpness; past it, only the
// gather. Perceptual roughness.
const float MIRROR_ROUGHNESS = 0.05;
const float GLOSSY_ROUGHNESS = 0.15;

vec3 view_pos(vec2 uv, float depth) {
    vec4 clip = ssr.inv_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return clip.xyz / clip.w;
}

// The pixel a quarter-rate ray starts from: its quad's first, every frame.
//
// Not rotated through the four. Rotating lets the history see each pixel's own
// surface in turn, and measured as more flicker than it bought: the corner
// that started the ray moves every frame, so every silhouette pixel's estimate
// does too, and the history has to average that away as well as the noise.
ivec2 ray_texel(ivec2 quad, ivec2 size) {
    return min(quad * 2, size - 1);
}

// The R2 low-discrepancy sequence (Roberts 2018) over the frame index, shifted
// per pixel by a blue-noise tile — a Cranley-Patterson rotation. Low
// discrepancy in time, blue in space, and each for its own filter.
//
// In time, it is the difference between a pixel drawing twelve *independent*
// samples across the history's accumulation window and drawing twelve that
// cover the lobe evenly. Independent samples leave clumps and gaps, and a gap
// on a bright reflection is exactly the flicker that survives a temporal
// filter.
//
// In space, it decides what the gather is left with. Neighbouring offsets from
// the tile are as far apart as the tile allows, so the nine rays a pixel
// gathers sample nine different parts of the lobe; a white hash would hand some
// neighbourhoods nine similar offsets, and those survive the gather as blotches
// the history then has to average away. Shifting every pixel by the same
// amount each frame keeps the tile blue: it only moves where its values sit
// on the circle.
vec2 stratified(ivec2 pixel, uint frame) {
    const vec2 alpha = vec2(0.7548776662, 0.5698402909);
    vec2 offset = texelFetch(u_blue_noise, pixel % textureSize(u_blue_noise, 0), 0).rg;
    return fract(alpha * float(frame) + offset);
}

// Heitz 2018, sampling the GGX distribution of visible normals. Sampling D
// alone would generate half-vectors the viewer cannot see and then throw the
// samples away; this generates only visible ones, which is what makes a single
// ray per pixel worth tracing at all.
vec3 sample_vndf(vec3 v, float alpha, vec2 u) {
    // Stretch the view direction so the anisotropic lobe becomes a hemisphere.
    vec3 vh = normalize(vec3(alpha * v.x, alpha * v.y, v.z));

    // An orthonormal basis around vh, built to stay stable as vh.z -> 1.
    float len2 = vh.x * vh.x + vh.y * vh.y;
    vec3 t1 = len2 > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(len2) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(vh, t1);

    // A point on the projected hemisphere: a disc, squashed toward the silhouette.
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;

    vec3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    // Unstretch back into the original frame.
    return normalize(vec3(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-6)));
}

// The ray from the surface at `uv` (view-space normal `n`), with lobe sample
// `u`: rgb = the radiance it found, a = how much of it to believe. Zero for a
// miss, with `ray_length` — metres from the surface to the hit — zero too.
vec4 trace_reflection(vec2 uv, float depth, vec3 n, float roughness, vec2 u, out float ray_length) {
    ray_length = 0.0;
    vec3 p = view_pos(uv, depth);
    vec3 v = normalize(-p); // toward the camera, view space

    // Sample the lobe in a frame where the normal is +Z, which is the frame the
    // visible-normal derivation is written in.
    vec3 t1 = abs(n.z) < 0.999 ? normalize(cross(vec3(0.0, 0.0, 1.0), n)) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(n, t1);
    mat3 tangent_to_view = mat3(t1, t2, n);
    vec3 v_local = vec3(dot(v, t1), dot(v, t2), dot(v, n));

    float alpha = roughness * roughness;
    vec3 h = roughness < MIRROR_ROUGHNESS ? n : tangent_to_view * sample_vndf(v_local, alpha, u);
    vec3 l = reflect(-v, h);

    // Below the horizon: the sampled microfacet reflects into the surface.
    if (dot(l, n) <= 0.0) {
        return vec4(0.0);
    }

    // Offset along the normal before projecting, so the ray does not begin
    // inside the very surface it left.
    vec3 origin = p + n * max(0.02, 0.004 * -p.z);
    vec3 target = origin + l * ssr.trace.y;
    // Clip to the near plane: a point behind the camera projects to a position
    // that means nothing, and the segment through it is not the ray.
    float near_z = -1e-3 + view_pos(vec2(0.5), DEPTH_NEAR).z;
    if (target.z > near_z) {
        target = origin + l * ((near_z - origin.z) / max(l.z, 1e-6));
    }

    vec4 clip0 = ssr.proj * vec4(origin, 1.0);
    vec4 clip1 = ssr.proj * vec4(target, 1.0);
    if (clip0.w <= 0.0 || clip1.w <= 0.0) {
        return vec4(0.0);
    }
    vec3 s0 = vec3((clip0.xy / clip0.w) * 0.5 + 0.5, clip0.z / clip0.w);
    vec3 s1 = vec3((clip1.xy / clip1.w) * 0.5 + 0.5, clip1.z / clip1.w);
    vec3 dir = s1 - s0;

    // Marching (uv, z) linearly is exact rather than an approximation: NDC z is
    // affine in 1/view_z, and 1/view_z is itself linear across the screen, so a
    // straight line in the world stays a straight line in this space.
    // The pyramid holds levels 1 and up; level 0 is the depth buffer itself.
    int max_level = int(ssr.trace.z);
    int level = 0;
    // Start a couple of full-resolution texels along, past the depth values the
    // origin surface itself wrote.
    float t = min(2.0 / max(length(dir.xy * ssr.extent.xy), 1e-4), 0.25);
    bool hit = false;

    for (int i = 0; i < int(ssr.trace.x); ++i) {
        if (t >= 1.0) {
            break;
        }
        vec3 pos = s0 + dir * t;
        if (any(lessThan(pos.xy, vec2(0.0))) || any(greaterThan(pos.xy, vec2(1.0)))) {
            break;
        }

        // Cells are laid on the depth buffer's own texel grid: a cell at level
        // `n` is exactly the 2^n x 2^n block `hiz_build.comp` reduced into it.
        // Not `pos * textureSize(level)`, which is the same grid only when 2^n
        // divides the extent: Vulkan floors each level's size, so on any other
        // viewport that grid drifts against the blocks by up to a cell across
        // the frame, and the march tests a ray against the depth of a block
        // beside the one it is in — a skipped surface, at cell granularity.
        float texels = float(1 << level);
        vec2 cell = floor(pos.xy * ssr.extent.xy / texels);
        // The far edge of this cell along each axis, and how far along the ray
        // that is. A zero component gives an infinity, which `min` discards.
        vec2 boundary = (cell + max(sign(dir.xy), vec2(0.0))) * texels * ssr.extent.zw;
        vec2 axis_t = (boundary - pos.xy) / dir.xy;
        float step_t = min(axis_t.x, axis_t.y);
        if (!(step_t > 0.0) || isinf(step_t)) {
            // The ray barely moves across the screen — a reflection almost
            // straight down the view axis. Nothing to skip; take the rest.
            step_t = 1.0;
        }
        // A nudge past the boundary, or the next iteration lands on the same
        // cell and the march stalls.
        float t_next = min(t + step_t + 1e-5, 1.0);

        float cell_z;
        if (level == 0) {
            cell_z = texelFetch(u_depth, min(ivec2(cell), ivec2(ssr.extent.xy) - 1), 0).r;
        } else {
            // The strip along the right and bottom edges that no stored texel
            // covers, where the extent is not a multiple of the cell: nothing
            // says it is empty, so look closer rather than skip it.
            if (any(greaterThanEqual(ivec2(cell), textureSize(u_hiz, level - 1)))) {
                level--;
                continue;
            }
            cell_z = texelFetch(u_hiz, ivec2(cell), level - 1).r;
        }
        float z_far = s0.z + dir.z * t_next;
        // The deepest the ray gets inside this cell. If that stays in front of
        // everything the cell contains, the cell is empty for this ray.
        float z_deep = depth_farthest(pos.z, z_far);

        if (!depth_nearer(z_deep, cell_z)) {
            if (level == 0) {
                // Land on the crossing itself rather than the cell edge, so the
                // hit UV is where the surface is.
                float cross_t = abs(dir.z) > 1e-8 ? (cell_z - s0.z) / dir.z : t;
                t = clamp(cross_t, t, t_next);
                hit = true;
                break;
            }
            // Something is in the way somewhere in this cell; look closer
            // without advancing.
            level--;
        } else {
            t = t_next;
            level = min(level + 1, max_level);
        }
    }

    if (!hit) {
        return vec4(0.0);
    }

    vec3 hit_screen = s0 + dir * t;
    vec2 hit_uv = hit_screen.xy;

    // The pyramid says a surface is in front of the ray, not that the ray ended
    // on it: a thin object the ray passed behind reports the same crossing. The
    // depth buffer is what decides, at the texel the march actually stopped on.
    float surface_depth = texture(u_depth, hit_uv).r;
    float surface_z = view_pos(hit_uv, surface_depth).z;
    float ray_z = view_pos(hit_uv, hit_screen.z).z;
    float gap = abs(ray_z - surface_z);
    if (gap > ssr.params.z) {
        return vec4(0.0);
    }

    // The depth buffer vouches for a hit only to within the ray's own depth
    // across one texel: that is how far apart a real crossing and the texel
    // the march stopped on can be. Further behind, the ray went behind the
    // surface rather than into it, and only `thickness` says something is
    // there — a floating sphere's hidden underside, read in the colour of its
    // lit top. A mirror shows that guess texel for texel and, where it runs
    // out, ends on a ragged edge that flickers with the jitter and leaves
    // slabs of the wrong surface, so a mirror's hit fades across the window
    // instead. A rough lobe blurs that edge by itself and keeps the plain cut:
    // its loose hits are most of the contact reflection under an object.
    vec3 next = hit_screen + dir / max(length(dir.xy * ssr.extent.xy), 1e-4);
    float slack = min(abs(view_pos(next.xy, next.z).z - ray_z), 0.5 * ssr.params.z);
    float sharp = 1.0 - smoothstep(MIRROR_ROUGHNESS, GLOSSY_ROUGHNESS, roughness);
    float behind = mix(1.0, 1.0 - smoothstep(slack, ssr.params.z, gap), sharp);

    // Three more things the trace cannot know anything about, faded rather than
    // cut so a reflection dissolves instead of ending on a line.
    vec2 border = min(hit_uv, 1.0 - hit_uv);
    float edge = smoothstep(0.0, 0.08, min(border.x, border.y));
    // A ray travelling back toward the camera is reflecting something the screen
    // does not contain — it is behind the viewer, or hidden by the surface doing
    // the reflecting.
    float facing = 1.0 - smoothstep(0.25, 0.75, dot(l, v));
    // And the lobe stops being resolvable by one ray long before the cutoff.
    float smoothness = 1.0 - smoothstep(ssr.params.y * 0.6, ssr.params.y, roughness);

    float confidence = behind * edge * facing * smoothness;

    // The ray is really a cone, and this is where that stops being a figure of
    // speech. A lobe of angular width ~alpha covers a patch of the world after
    // travelling `distance`; reading one texel of that patch is the variance
    // that shows up as grain on every surface that is not a mirror, and reading
    // the pyramid level whose texel *is* the patch costs the same fetch.
    //
    // Half the lobe, not all of it: the direction was already sampled *from*
    // the lobe, so prefiltering over its full width would blur by the roughness
    // twice. Half puts the effective width about a tenth over the truth and
    // takes most of the variance with it.
    const float CONE_SCALE = 0.5;
    vec3 hit_view = view_pos(hit_uv, surface_depth);
    ray_length = distance(hit_view, p);
    float cone = ray_length * alpha * CONE_SCALE;
    // World units to texels of the full-resolution frame: a length at view
    // depth z spans `proj[1][1] / |z|` of NDC, and NDC spans two across it.
    float texels = cone * (ssr.proj[1][1] / max(abs(hit_view.z), 1e-4)) * ssr.extent.y * 0.5;
    float lod = log2(max(texels, 1.0));

    // A cone under two texels wide reads the frame itself, and the pyramid —
    // whose first level is already half the frame — takes over from there,
    // cross-faded over that first octave so a cone that widens along a surface
    // never steps from one to the other. A mirror's cone is under a texel for
    // any distance on screen, which is what makes its reflection as sharp as
    // the frame it reflects.
    vec3 radiance = textureLod(u_source, hit_uv, max(lod - 1.0, 0.0)).rgb;
    if (lod < 1.0) {
        radiance = mix(textureLod(u_source_full, hit_uv, 0.0).rgb, radiance, lod);
    }
    return vec4(radiance, confidence);
}

#endif
