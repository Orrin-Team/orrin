// What the environment lays on a surface: its irradiance for the diffuse lobe
// and its prefiltered radiance for the specular one. Included by `shading.glsl`
// and by `ray_shade.glsl`, so a hit is under the sky its reflector is.
//
// The includer declares, before including this file, the `lighting` block
// (`lights.glsl`), the environment cube as `u_environment` with
// `u_environment_sampler`, `SPECULAR_MIPS`, and `env_brdf_approx` from
// `brdf.glsl`.

#ifndef ORRIN_AMBIENT_GLSL
#define ORRIN_AMBIENT_GLSL

// Diffuse irradiance from the environment, divided by pi so the result is
// already the diffuse response for unit albedo.
//
// The nine terms are the real spherical-harmonic basis for bands 0..=2, and
// they must match `basis` in gfx/sh.rs term for term and component for
// component — the coefficients were projected against that one. A mismatch is
// not a compile error, it is lighting that is quietly rotated or mirrored.
//
// With no environment loaded these coefficients carry the scene's flat ambient
// in band 0 alone, which evaluates to that ambient for every normal. So there
// is one path here, not two.
// World space into the environment's own frame. The same rotation the skybox
// samples through, so what is drawn behind the scene and what lights it cannot
// disagree.
vec3 to_environment(vec3 v) {
    float s = lighting.environment.x;
    float c = lighting.environment.y;
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

vec3 sh_irradiance(vec3 n) {
    n = to_environment(n);

    vec3 e = lighting.irradiance[0].rgb * 0.282095
           + lighting.irradiance[1].rgb * (0.488603 * n.y)
           + lighting.irradiance[2].rgb * (0.488603 * n.z)
           + lighting.irradiance[3].rgb * (0.488603 * n.x)
           + lighting.irradiance[4].rgb * (1.092548 * n.x * n.y)
           + lighting.irradiance[5].rgb * (1.092548 * n.y * n.z)
           + lighting.irradiance[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + lighting.irradiance[7].rgb * (1.092548 * n.x * n.z)
           + lighting.irradiance[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));

    // Backstop for the ringing the band window in gfx/sh.rs is sized to
    // contain: a truncated series overshoots at a sun disc and undershoots
    // opposite it, and the undershoot can cross zero.
    return max(e, vec3(0.0));
}

// Specular image-based lighting, Karis' split sum: the prefiltered chain is the
// light integral, `env_brdf_approx` the BRDF integral. This is the term that
// gives a metal anything to be — its diffuse response is zero by definition, so
// without it a metal is lit only where it happens to mirror a light.
//
// `energy` is the same multi-scatter compensation the analytic lights get, and
// it belongs here for the same reason: single-scatter GGX loses energy with
// roughness whether the light is a sun or a sky.
// Takes the reflection vector rather than deriving it, because two callers do
// not want `reflect(-V, N)`: an anisotropic surface reflects about a normal bent
// toward its stretch, and a clear coat reflects about its own.
vec3 specular_ibl_along(vec3 R, float n_dot_v, vec3 f0, float perceptual_roughness, vec3 energy) {
    float lod = perceptual_roughness * (SPECULAR_MIPS - 1.0);
    vec3 radiance = textureLod(
        samplerCube(u_environment, u_environment_sampler), to_environment(R), lod
    ).rgb;

    vec2 dfg = env_brdf_approx(perceptual_roughness, n_dot_v);
    return radiance * lighting.env_specular.rgb * (f0 * dfg.x + dfg.y) * energy;
}

vec3 specular_ibl(vec3 N, vec3 V, vec3 f0, float perceptual_roughness, vec3 energy) {
    return specular_ibl_along(
        reflect(-V, N), max(dot(N, V), 1e-4), f0, perceptual_roughness, energy
    );
}

#endif
