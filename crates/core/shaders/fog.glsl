// The air, once: how a froxel maps to the world, how the medium scatters, and
// what a fragment does with the volume the two compute passes left behind.
//
// Included by `fog_scatter.comp`, which fills the volume, and by `shading.glsl`
// and `skybox.frag`, which read it. That is the same argument `brdf.glsl` makes:
// the mapping has to be one function, or the froxel a fragment reads is not the
// froxel that was lit for it, and the fog would slide against the geometry as
// the camera moved.
//
// Nothing here declares a binding. The constants arrive as a [`FogParams`] value
// out of whichever block the includer owns and the volume arrives as a
// parameter, so a compute pass at set 0 and a fragment shader at set 3 share the
// file.

#ifndef ORRIN_FOG_GLSL
#define ORRIN_FOG_GLSL

// For the `Cascades` half of the block below: the scatter pass reads the shadow
// maps, so the same uniform carries what it takes to read them.
#include "cascades.glsl"
#include "depth.glsl"

const float FOG_PI = 3.14159265359;

// Depth slices in the froxel volume. Keep in sync with `FROXEL_SLICES` in
// gfx/vulkan/fog.rs — the integration walks exactly this many.
const int FOG_SLICES = 64;

// Where the volume starts, in metres. Slices are distributed exponentially
// between here and the far distance, so this is what stops a sixty-fourth of the
// budget from being spent on the first centimetre in front of the lens.
const float FOG_NEAR = 0.5;

// Mirrors `GpuFog` in gfx/vulkan/fog.rs. std140 packs this exactly like the Rust
// `#[repr(C)]` struct because every field is a `mat4` or a `vec4`.
struct FogParams {
    // Unjittered, and both of them deliberately. The volume is reprojected
    // against its own history rather than resolved by TAA, so baking the
    // raster's subpixel jitter into it would fight that reprojection — and a
    // froxel is sixteen pixels wide, so the offset it would buy is a sixteenth
    // of a texel.
    mat4 inv_view_proj;
    mat4 prev_view_proj;
    // rgb = single-scattering albedo, w = extinction at the reference height.
    vec4 albedo;
    // x = height falloff, y = reference height, z = far distance, w = anisotropy.
    vec4 params;
    // xyz = direction toward the sun.
    vec4 sun_direction;
    // rgb = sun colour, w = illuminance in lux on a surface facing it.
    vec4 sun_color;
    // rgb = ambient luminance the medium sees, cd/m2.
    vec4 ambient;
    // xyz = camera world position, w = 1.0 when the froxel volume is live.
    vec4 camera_pos;
    // xyz = where the camera was last frame. The reprojection needs it as well
    // as `prev_view_proj`, because a froxel's slice is its *distance* from the
    // camera and a matrix alone cannot say what that was.
    vec4 prev_camera_pos;
    // x = history feedback, y = depth jitter in [0, 1), z = 1.0 to drop the
    // history.
    vec4 temporal;
};

// The whole uniform block, which is `FogParams` plus what the scatter pass needs
// to read the cascades. Declared here so the compute pass at set 0 and the
// shading path at set 3 cannot disagree about its shape; only the `layout` line
// is the includer's.
struct GpuFog {
    FogParams fog;
    // A second copy of what `GpuLighting` already carries, and deliberately: the
    // scatter pass is a dispatch with no lighting set bound, and the alternative
    // is dragging the whole `Lighting` declaration — point lights, spot lights,
    // irradiance — into a shader that wants four fields of it. One Rust
    // expression fills both, so the copies are made together or not at all.
    Cascades cascades;
};

// The view distance a normalised slice coordinate sits at, and its inverse.
//
// Exponential rather than linear: the froxels nearest the camera cover the most
// screen and the most parallax, and a linear distribution spends the same texel
// on the last ten metres of haze that it spends on the first. Both directions
// exist because the scatter pass walks slices forward and every reader arrives
// with a world position to walk back.
float fog_slice_depth(FogParams f, float t) {
    return FOG_NEAR * pow(f.params.z / FOG_NEAR, t);
}

float fog_depth_slice(FogParams f, float view_dist) {
    return clamp(
        log(max(view_dist, FOG_NEAR) / FOG_NEAR) / log(f.params.z / FOG_NEAR),
        0.0,
        1.0
    );
}

// The view ray through a froxel column, as a unit direction in world space.
vec3 fog_ray_direction(FogParams f, vec2 uv) {
    vec4 far = f.inv_view_proj * vec4(uv * 2.0 - 1.0, DEPTH_FAR, 1.0);
    return normalize(far.xyz / far.w - f.camera_pos.xyz);
}

// Extinction at a world position: the same exponential height falloff the
// analytic integral below assumes, evaluated pointwise. One function, so the
// segment the froxels march and the segment past them describe one medium.
float fog_density(FogParams f, vec3 world_pos) {
    return f.albedo.w * exp(-f.params.x * (world_pos.y - f.params.y));
}

// Optical depth of the height layer between two distances along a ray.
//
// The analytic integral the froxels are not marching: `exp(-falloff * y)` along
// `p(s) = origin + dir * s` is an exponential in `s`, so the whole segment is one
// closed form rather than a march. Ranged rather than measured from the camera
// because it has two jobs — the whole ray when the volume is off, and only the
// tail past the volume's far plane when it is on.
float fog_optical_depth(FogParams f, vec3 origin, vec3 dir, float near, float far) {
    float density = f.albedo.w;
    if (density <= 0.0 || far <= near) {
        return 0.0;
    }

    float at_origin = exp(-f.params.x * (origin.y - f.params.y));
    float k = f.params.x * dir.y;
    // The quotient has a removable singularity for rays with no vertical
    // component, where the integral is just the segment's length.
    float segment = abs(k) > 1e-4 ? (exp(-k * near) - exp(-k * far)) / k : (far - near);
    return density * at_origin * max(segment, 0.0);
}

// Henyey-Greenstein. `cos_theta` is between the direction light is travelling
// and the direction it scatters into, which for a view ray `dir` and a direction
// `L` toward the sun is `dot(L, dir)`: looking at the sun through fog is `+1` and
// is the bright case, which is the whole reason the term is here.
float fog_phase(float g, float cos_theta) {
    float g2 = g * g;
    float d = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * FOG_PI * d * sqrt(d));
}

// What the medium sends toward the camera per unit of *scattering*, given how
// much of the sun reaches it. Shared by the froxel pass, which has a shadow
// lookup to hand it, and by the analytic tail, which passes 1.0 — past the
// volume there is no shadow map deep enough to say otherwise.
vec3 fog_inscattered(FogParams f, vec3 dir, float sun_visibility) {
    float phase = fog_phase(f.params.w, dot(f.sun_direction.xyz, dir));
    vec3 sun = f.sun_color.rgb * f.sun_color.w * phase * sun_visibility;
    // The ambient is already a luminance and arrives from every direction, so it
    // takes the isotropic phase rather than the one above.
    return f.albedo.rgb * (sun + f.ambient.rgb);
}

// In-scattered radiance and the transmittance behind it — which is exactly the
// pair a fragment needs, because fog is still the lerp it always was: the
// surface keeps `transmittance` of what it sent and the air adds `inscatter` on
// top.
struct FogTerm {
    vec3 inscatter;
    float transmittance;
};

// The whole ray from the camera to `world_pos`, in the two segments it is made
// of: whatever the froxel volume covered, then the analytic height layer past
// it. Composing them is a product on transmittance and a
// `near + T_near * far` on radiance, which is what makes the volume's far plane
// invisible rather than a ring in the image.
//
// `screen_uv` is the fragment's own, so a froxel is read at the pixel it was lit
// for. Past the volume the lookup clamps to the last slice, which already holds
// the whole volume's integral — so a distant surface and the sky read the same
// value and the tail below does the rest.
FogTerm fog_term(FogParams f, sampler3D volume, vec3 world_pos, vec2 screen_uv) {
    vec3 camera_pos = f.camera_pos.xyz;
    vec3 ray = world_pos - camera_pos;
    float dist = length(ray);
    vec3 dir = ray / max(dist, 1e-4);

    FogTerm term;
    term.inscatter = vec3(0.0);
    term.transmittance = 1.0;

    float tail_near = 0.0;
    if (f.camera_pos.w > 0.5) {
        vec4 v = textureLod(volume, vec3(screen_uv, fog_depth_slice(f, dist)), 0.0);
        term.inscatter = v.rgb;
        term.transmittance = v.a;
        tail_near = min(dist, f.params.z);
    }

    float tau = fog_optical_depth(f, camera_pos, dir, tail_near, dist);
    if (tau > 0.0) {
        float tail = exp(-tau);
        // Unshadowed, and it has to be: the tail is the part of the ray beyond
        // where the cascades have any depth to answer with. It is also the part
        // where a shaft would not be visible anyway — a hundred metres of haze
        // has scattered the sun into itself several times over.
        vec3 radiance = fog_inscattered(f, dir, 1.0);
        term.inscatter += term.transmittance * radiance * (1.0 - tail);
        term.transmittance *= tail;
    }

    return term;
}

#endif // ORRIN_FOG_GLSL
