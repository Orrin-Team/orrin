// One reflection ray: the exact reflection of the view ray about the surface
// normal, its march through the depth pyramid, and the radiance it finds —
// blurred by the surface's roughness at the hit.
//
// Shared by the two passes that trace. `ssr_trace.comp` traces one ray per 2x2
// quad for every surface the reflections reach; `ssr_resolve.comp` traces the
// other three of a smooth surface's quad itself, at the pixel that needs it, so
// a mirror's reflection is as sharp as the frame. One function, so the two
// rates find the same thing and a roughness gradient that crosses from one to
// the other changes how often a ray is traced, never what a ray finds.
//
// The ray is the same every frame for the same surface, so nothing here varies
// from frame to frame but what the frame itself does — see `ssr.rs` for why that
// is the design, and what it gives up.
//
// The including shader declares, by these names: the `Ssr` block as `ssr`,
// `u_hiz` (the nearest-depth pyramid from level 1 up), `u_depth`, `u_source`
// (the lit frame's pyramid, based at half resolution) and `u_source_full` (the
// lit frame itself).
//
// With `ORRIN_RAY_FALLBACK` defined, what the screen cannot answer is traced
// against the scene's acceleration structure instead: the same ray, in the
// world, shaded where it lands. The includer then includes `ray_shade.glsl`
// first. The order is the one id Tech 8 gives: the screen when it can, then a
// traced ray, then the environment.

#ifndef ORRIN_SSR_MARCH_GLSL
#define ORRIN_SSR_MARCH_GLSL

#include "depth.glsl"

// Below the first, a pixel's own full-rate ray is all its reflection is; past
// the second, the quarter-rate rays its neighbours traced, gathered; between
// them the two cross-fade, so a roughness gradient shows no line. The gather is
// a blur a quad or two wide, which a mirror cannot afford and a surface at 0.15
// no longer shows — by then the blur at the hit is wider than a quad.
// Perceptual roughness.
const float FULL_RATE_ROUGHNESS = 0.05;
const float QUARTER_RATE_ROUGHNESS = 0.15;

vec3 view_pos(vec2 uv, float depth) {
    vec4 clip = ssr.inv_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return clip.xyz / clip.w;
}

// The pixel a quarter-rate ray starts from: its quad's first, every frame.
ivec2 ray_texel(ivec2 quad, ivec2 size) {
    return min(quad * 2, size - 1);
}

// The pyramid at fractional `level`, through a cubic B-spline rather than the
// sampler's tent. A box-reduced level read bilinearly shows its texels as
// blocks once they are tens of pixels wide, and a reflection that slides
// across them steps; the B-spline is smooth through each texel boundary.
// Four bilinear taps placed between texels, as Sigg and Hadwiger do it, each
// trilinear, with the offsets of the finer of the two levels it blends.
vec3 pyramid_smooth(vec2 uv, float level) {
    vec2 size = vec2(textureSize(u_source, int(level)));
    vec2 position = uv * size - 0.5;
    vec2 centre = floor(position);
    vec2 f = position - centre;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1;
    vec2 g1 = 1.0 - g0;
    vec2 at0 = (centre + 0.5 + w1 / g0 - 1.0) / size;
    vec2 at1 = (centre + 0.5 + w3 / g1 + 1.0) / size;
    return g0.y * (g0.x * textureLod(u_source, vec2(at0.x, at0.y), level).rgb
                 + g1.x * textureLod(u_source, vec2(at1.x, at0.y), level).rgb)
         + g1.y * (g0.x * textureLod(u_source, vec2(at0.x, at1.y), level).rgb
                 + g1.x * textureLod(u_source, vec2(at1.x, at1.y), level).rgb);
}

// What the screen holds of the reflection of the surface at `uv` (view-space
// normal `n`): rgb = the radiance found, a = how much of it the screen can
// vouch for, zero for a miss.
vec4 march_reflection(vec2 uv, float depth, vec3 n, float roughness) {
    vec3 p = view_pos(uv, depth);
    vec3 v = normalize(-p); // toward the camera, view space
    vec3 l = reflect(-v, n);

    // Below the horizon: a normal map tilted past the view ray.
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
    if (gap > ssr.params.y) {
        return vec4(0.0);
    }

    // The depth buffer vouches for a hit only to within the ray's own depth
    // across one texel: that is how far apart a real crossing and the texel
    // the march stopped on can be. Further behind, the ray went behind the
    // surface rather than into it, and only `thickness` says something is
    // there — a floating sphere's hidden underside, read in the colour of its
    // lit top. An exact ray shows that guess texel for texel and, where it
    // runs out, ends on a ragged edge that flickers with the jitter and leaves
    // slabs of the wrong surface, so the hit fades across the window instead.
    vec3 next = hit_screen + dir / max(length(dir.xy * ssr.extent.xy), 1e-4);
    float slack = min(abs(view_pos(next.xy, next.z).z - ray_z), 0.5 * ssr.params.y);
    float behind = 1.0 - smoothstep(slack, ssr.params.y, gap);

    // Three more things the trace cannot know anything about, faded rather than
    // cut so a reflection dissolves instead of ending on a line.
    vec2 border = min(hit_uv, 1.0 - hit_uv);
    float edge = smoothstep(0.0, 0.08, min(border.x, border.y));
    // A ray travelling back toward the camera is reflecting something the screen
    // does not contain — it is behind the viewer, or hidden by the surface doing
    // the reflecting.
    float facing = 1.0 - smoothstep(0.25, 0.75, dot(l, v));

    float confidence = behind * edge * facing;

    // The roughness is a cone around the exact ray, and it is read as one: the
    // pyramid level whose texel matches the cone's width at the hit carries the
    // average of what the lobe would have seen. Short rays stay sharp and long
    // ones soften, which is a contact reflection hardening toward the contact.
    //
    // 2.5 alpha is the reflected lobe's full width at half its peak: GGX puts
    // its half-maximum at `tan(theta_h) = 0.64 alpha`, and a reflection turns
    // twice as far as the half-vector does.
    const float CONE_SCALE = 2.5;
    float alpha = roughness * roughness;
    vec3 hit_view = view_pos(hit_uv, surface_depth);
    float cone = distance(hit_view, p) * alpha * CONE_SCALE;
    // World units to texels of the full-resolution frame: a length at view
    // depth z spans `|proj[1][1]| / |z|` of NDC, and NDC spans two across it.
    // The magnitude, because Vulkan's flipped Y makes `proj[1][1]` negative.
    float texels = cone * (abs(ssr.proj[1][1]) / max(abs(hit_view.z), 1e-4)) * ssr.extent.y * 0.5;
    float lod = log2(max(texels, 1.0));

    // A cone under two texels wide reads the frame itself, and the pyramid —
    // whose first level is already half the frame — takes over from there,
    // cross-faded over that first octave so a cone that widens along a surface
    // never steps from one to the other. A mirror's cone is under a texel for
    // any distance on screen, which is what makes its reflection as sharp as
    // the frame it reflects.
    vec3 radiance = pyramid_smooth(hit_uv, max(lod - 1.0, 0.0));
    if (lod < 1.0) {
        radiance = mix(textureLod(u_source_full, hit_uv, 0.0).rgb, radiance, lod);
    }
    return vec4(radiance, confidence);
}

#ifdef ORRIN_RAY_FALLBACK
// Past the first of these a traced hit starts handing back to the environment,
// and by the second it has. A hit is one point's radiance, sharp, where a rough
// lobe wants a blur of everything round it, and nothing here can blur what is
// not on the screen: on a mirror that is exact, and on a rough surface it is a
// crisp image inside a soft one. Perceptual roughness.
const float TRACED_FULL_ROUGHNESS = 0.15;
const float TRACED_NO_ROUGHNESS = 0.3;

// A pixel's own angle, near enough, in metres of cone per metre of ray: what
// a mirror's reflection is filtered by at its hit.
const float PIXEL_SPREAD = 0.0015;

// The part of a reflection the screen could not vouch for, traced against the
// scene itself and shaded where it lands. `found` is what the screen had; the
// result is the two blended by how much each knows. A ray that leaves the
// scene changes nothing: what is out there is the sky, and the environment
// already answers for it.
vec4 trace_past_the_screen(vec4 found, vec2 uv, float depth, vec3 n, float roughness) {
    float reach = (1.0 - found.a)
                * (1.0 - smoothstep(TRACED_FULL_ROUGHNESS, TRACED_NO_ROUGHNESS, roughness));
    if (reach <= 0.0) {
        return found;
    }
    vec3 p = view_pos(uv, depth);
    vec3 l = reflect(normalize(p), n);
    if (dot(l, n) <= 0.0) {
        return found;
    }
    vec3 origin = (ssr.inv_view * vec4(p + n * max(0.02, 0.004 * -p.z), 1.0)).xyz;
    vec3 direction = normalize(mat3(ssr.inv_view) * l);
    RayHit hit = ray_closest(origin, direction, 0.0, RAY_FAR);
    if (!hit.found) {
        return found;
    }
    // The cone the screen's own hit is read through, as a slope.
    float spread = PIXEL_SPREAD + roughness * roughness * 2.5;
    vec3 traced = ray_shade(hit, direction, spread);
    float total = found.a + reach;
    return vec4((found.rgb * found.a + traced * reach) / total, total);
}
#endif

// The reflection of the surface at `uv` (view-space normal `n`): rgb = the
// radiance found, a = how much of it to believe, zero for a miss.
vec4 trace_reflection(vec2 uv, float depth, vec3 n, float roughness) {
    vec4 found = march_reflection(uv, depth, n, roughness);
#ifdef ORRIN_RAY_FALLBACK
    found = trace_past_the_screen(found, uv, depth, n, roughness);
#endif
    // A lobe wide enough to see most of the hemisphere is a reflection the
    // prefiltered environment approximates better than a blur of the screen.
    float smoothness = 1.0 - smoothstep(ssr.params.x * 0.6, ssr.params.x, roughness);
    return vec4(found.rgb, found.a * smoothness);
}

#endif
