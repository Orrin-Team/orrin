// What a surface does with one light: the microfacet terms, the description of
// a surface they are evaluated against, and `brdf`, which sums the lobes for
// one direction. Included by `shading.glsl` and by `ray_shade.glsl`: a
// fragment and a ray's hit differ in how they come by a `Surface` and in which
// lights they loop over, and in nothing below.
//
// Nothing here reads a binding, a varying or a derivative. The includer
// defines `PI` and includes `material.glsl` first, for the feature bits.

#ifndef ORRIN_LOBES_GLSL
#define ORRIN_LOBES_GLSL

// --- Cook-Torrance terms (metallic-roughness workflow) ---

// GGX / Trowbridge-Reitz normal distribution.
float distribution_ggx(float n_dot_h, float a) {
    float a2 = a * a;
    float d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility (already folds in the 1/(4 NoL NoV) denom).
float visibility_smith_ggx(float n_dot_v, float n_dot_l, float a) {
    float a2 = a * a;
    float gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

// Fresnel-Schlick reflectance.
vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

float fresnel_schlick(float v_dot_h, float f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// --- Anisotropic GGX (Burley 2012, in Filament's formulation) ---
//
// The isotropic pair above is this one with `at == ab`; the difference is that
// the halfvector is measured against the tangent frame rather than against the
// normal alone, which is what lets the highlight stretch. A brushed or woven
// surface has grooves running one way, and a lobe that cannot tell the two
// directions apart cannot show them.

// Smallest either roughness may reach. The distribution divides by a quantity
// that goes to zero with `at * ab`, so a fully anisotropic surface at zero
// roughness is a singularity rather than a sharp highlight.
const float MIN_ANISO_ROUGHNESS = 1e-3;

float distribution_ggx_aniso(float n_dot_h, float t_dot_h, float b_dot_h, float at, float ab) {
    float a2 = at * ab;
    vec3 d = vec3(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    float d2 = dot(d, d);
    float w2 = a2 / max(d2, 1e-9);
    return a2 * w2 * w2 / PI;
}

float visibility_smith_ggx_aniso(
    float n_dot_v, float n_dot_l,
    float t_dot_v, float b_dot_v,
    float t_dot_l, float b_dot_l,
    float at, float ab
) {
    float lambda_v = n_dot_l * length(vec3(at * t_dot_v, ab * b_dot_v, n_dot_v));
    float lambda_l = n_dot_v * length(vec3(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

// --- Clear coat ---
//
// Kelemen's visibility rather than Smith's. The coat is a thin, near-smooth
// film, and over that range the two agree closely enough that the cheaper one is
// free accuracy elsewhere — which matters because the coat is a *second* full
// specular evaluation on top of a surface that already paid for one.
float visibility_kelemen(float v_dot_h) {
    return 0.25 / max(v_dot_h * v_dot_h, 1e-5);
}

// --- Sheen (Estevez & Kulla 2017) ---
//
// The Charlie distribution, which unlike GGX peaks *away* from the normal — that
// retroreflective rim is the whole visual signature of cloth, and no amount of
// roughness on a GGX lobe produces it.
float distribution_charlie(float n_dot_h, float a) {
    float inv_a = 1.0 / max(a, 1e-3);
    float cos2 = n_dot_h * n_dot_h;
    // Floored rather than clamped to zero: `pow` of zero with a large exponent
    // is a legal way to get a NaN out of a surface facing exactly edge-on.
    float sin2 = max(1.0 - cos2, 0.0078125);
    return (2.0 + inv_a) * pow(sin2, inv_a * 0.5) / (2.0 * PI);
}

// Ashikhmin's uniform visibility term, as Neubelt & Pettineo use it. Smith's
// derivation assumes GGX microfacets and does not apply to Charlie's.
float visibility_neubelt(float n_dot_v, float n_dot_l) {
    return clamp(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 0.0, 1.0);
}

// Everything about this pixel that does not depend on which light is being
// summed, resolved once by `read_surface` and handed to every lobe below.
//
// A struct rather than a parameter list because the list had already reached
// nine and each of the three extra lobes wants four more. The fields past
// `energy` are only meaningful when the matching bit is set in `flags`; nothing
// reads them otherwise, so a plain material leaves them at whatever
// `read_surface` skipped writing.
struct Surface {
    vec3 N;
    vec3 V;
    /// Where this fragment actually reads the material's maps. `v_uv` until the
    /// parallax march moves it, and every map the material carries has to be
    /// sampled here rather than at the interpolated coordinate — a normal read
    /// one brick along from the albedo is worse than no parallax at all.
    vec2 uv;
    vec3 albedo;
    vec3 f0;
    float metallic;
    /// Linear roughness, already widened for specular antialiasing.
    float a;
    /// Perceptual roughness, which is what the split-sum fits are parameterised
    /// by — keeping both saves squaring and rooting back at every use.
    float perceptual_roughness;
    vec3 energy;
    float n_dot_v;
    uint flags;

    // Anisotropy: the tangent frame the lobe is stretched along, rotated out of
    // the geometric one by the material's angle, and the two roughnesses that
    // stretch it.
    vec3 T;
    vec3 B;
    float at;
    float ab;
    /// Signed strength, kept because the environment lookup needs it after the
    /// two roughnesses above have already absorbed it.
    float anisotropy;

    // Clear coat: its own normal, because an orange-peel film and the grain
    // beneath it are different surfaces.
    vec3 coat_N;
    float coat;
    float coat_a;
    float coat_roughness;

    vec3 sheen_color;
    float sheen_a;

    // Subsurface scattering: what comes back out of the medium, and the two
    // distances that decide where from.
    vec3 scatter_color;
    /// Per-channel mean free path, in metres. Red reaches furthest in every
    /// medium this exists for, which is why it is a vector.
    vec3 scatter_radius;
    /// Metres of medium behind this point. The reference length the wrap below is
    /// measured against, and the distance the transmitted term is absorbed over.
    float thickness;
    float forward_scatter;
    /// How far past the terminator the diffuse lobe reaches, per channel.
    /// Resolved once per fragment by `subsurface_wrap` — it reads a screen-space
    /// derivative, so it cannot be computed per light inside `brdf`.
    vec3 wrap;
};

// The diffuse cosine, widened by the wrap and renormalised so widening it does
// not also brighten it.
//
// The `(1 + w)^2` is what makes this an energy-preserving wrap rather than a
// wrap plus a gain: the numerator's range grows by `1 + w` and the lobe's
// integral over the hemisphere by another factor of it.
vec3 wrapped_diffuse(Surface s, float raw_n_dot_l) {
    vec3 wrapped = (vec3(raw_n_dot_l) + s.wrap) / ((1.0 + s.wrap) * (1.0 + s.wrap));
    return max(wrapped, vec3(0.0));
}

// Light that entered the far side of the surface and left through this one.
//
// Two things multiply: what survives `thickness` of medium — Beer-Lambert
// against the mean free path, per channel, which is what turns a thick limb
// opaque and leaves an ear glowing — and a phase function that is mostly
// forward. The forward lobe is why a leaf lights up when the camera is nearly
// looking into the sun through it and goes flat when it steps aside, and the
// `wrap_back` floor underneath keeps a surface lit from directly behind from
// going dark off-axis.
//
// Deliberately not multiplied by the surface's own albedo: this light never
// reflected off the boundary, it came *through*, so what tints it is the medium's
// colour and nothing else. The same rule `KHR_materials_diffuse_transmission`
// states.
vec3 subsurface_transmission(Surface s, vec3 L, float raw_n_dot_l) {
    vec3 through = exp(-s.thickness / s.scatter_radius);
    float forward = exp2(clamp(dot(s.V, -L), 0.0, 1.0) * s.forward_scatter - s.forward_scatter);
    float wrap_back = clamp(-raw_n_dot_l, 0.0, 1.0);
    return s.scatter_color * through * mix(wrap_back, 1.0, forward) / PI;
}

// One light's contribution, split by what the frame is allowed to spread.
//
// `direct` is everything whose position on screen is the surface it came off:
// every specular lobe, and the diffuse of a material that does not scatter.
// `diffusible` is the part that physically left the surface somewhere other than
// where it arrived, which is exactly the licence the screen-space passes need to
// move it — and is zero unless the material scatters, so a frame with no
// subsurface material in it has an empty second target.
struct Lobes {
    vec3 direct;
    vec3 diffusible;
};

// The base layer's specular lobe: GGX, stretched along the tangent frame when
// the material asked for it.
vec3 base_specular(Surface s, vec3 L, vec3 H, float n_dot_l, float n_dot_h) {
    float D;
    float Vis;
    if ((s.flags & MATERIAL_ANISOTROPY) != 0u) {
        D = distribution_ggx_aniso(n_dot_h, dot(s.T, H), dot(s.B, H), s.at, s.ab);
        Vis = visibility_smith_ggx_aniso(
            s.n_dot_v, n_dot_l,
            dot(s.T, s.V), dot(s.B, s.V),
            dot(s.T, L), dot(s.B, L),
            s.at, s.ab
        );
    } else {
        D = distribution_ggx(n_dot_h, s.a);
        Vis = visibility_smith_ggx(s.n_dot_v, n_dot_l, s.a);
    }
    return vec3(D * Vis);
}

// Outgoing radiance toward the camera from one light direction L.
// Takes the surface's linear roughness rather than its perceptual one: the
// caller filters it once for specular antialiasing, and this runs once per light.
//
// The two visibilities are the light's own shadow term, applied here rather than
// by the caller because a scattering surface needs two different ones and only
// this function knows which term wants which. `visibility` is whether the light
// reaches *this* point; `back_visibility` is whether it reaches the far side of
// the medium, which is the question the transmitted lobe is actually asking — an
// ear lit from behind is in shadow by the first measure and lit by the second,
// and answering it with the first is how subsurface scattering ends up invisible
// on exactly the geometry it exists for.
Lobes brdf(Surface s, vec3 L, vec3 radiance, float visibility, float back_visibility) {
    Lobes lobes;
    lobes.direct = vec3(0.0);
    lobes.diffusible = vec3(0.0);

    bool scatters = (s.flags & MATERIAL_SUBSURFACE) != 0u;
    float raw_n_dot_l = dot(s.N, L);

    // Before the facing test, and it has to survive it: this is light that
    // entered the other side, so a surface turned away from the light is where
    // the term does its work rather than where it stops.
    if (scatters) {
        lobes.diffusible =
            subsurface_transmission(s, L, raw_n_dot_l) * radiance * back_visibility;
    }

    float n_dot_l = max(raw_n_dot_l, 0.0);
    if (n_dot_l <= 0.0) {
        return lobes;
    }
    vec3 H = normalize(L + s.V);
    float n_dot_h = max(dot(s.N, H), 0.0);
    float v_dot_h = max(dot(s.V, H), 0.0);

    vec3 F = fresnel_schlick(v_dot_h, s.f0);
    vec3 specular = base_specular(s, L, H, n_dot_l, n_dot_h) * F * s.energy;

    // Diffuse keeps the energy not reflected (1 - F) and not metallic.
    vec3 kd = (vec3(1.0) - F) * (1.0 - s.metallic);

    vec3 color = specular;
    vec3 diffusible = vec3(0.0);
    if (scatters) {
        // The wrap *replaces* the cosine rather than scaling it — that is the
        // whole mechanism — so this term deliberately misses the `n_dot_l` the
        // returns below apply to everything else. Multiplying by both would
        // undo the wrap at the one place it matters, the band just inside the
        // terminator where `n_dot_l` is near zero.
        diffusible = kd * s.albedo * wrapped_diffuse(s, raw_n_dot_l) / PI;
    } else {
        color += kd * s.albedo / PI;
    }

    // Sheen sits beside the base lobes rather than over them: it is a separate
    // set of fibres catching light, not a film. The base is scaled down for it
    // once, in `read_surface`, rather than per light here.
    if ((s.flags & MATERIAL_SHEEN) != 0u) {
        color += s.sheen_color
               * distribution_charlie(n_dot_h, s.sheen_a)
               * visibility_neubelt(s.n_dot_v, n_dot_l);
    }

    if ((s.flags & MATERIAL_CLEARCOAT) == 0u) {
        lobes.direct += color * radiance * n_dot_l * visibility;
        lobes.diffusible += diffusible * radiance * visibility;
        return lobes;
    }

    // The coat is a film *over* everything above, so it both adds a reflection
    // and takes one away: what the base layer receives is what got *through* the
    // coat, which is the `1 - Fc`. Its own lobe is then lit by its own normal —
    // a coat normal map that disagrees with the base's is the entire reason the
    // two n·l terms are kept apart rather than folded into the common factor.
    //
    // A polyurethane film: 4% at normal incidence, as every clear coat is.
    float Fc = fresnel_schlick(v_dot_h, 0.04) * s.coat;
    float coat_n_dot_l = max(dot(s.coat_N, L), 0.0);
    float coat_n_dot_h = max(dot(s.coat_N, H), 0.0);
    float coat_spec = distribution_ggx(coat_n_dot_h, s.coat_a)
                    * visibility_kelemen(v_dot_h) * Fc;

    lobes.direct += (color * (1.0 - Fc) * n_dot_l + coat_spec * coat_n_dot_l)
                  * radiance * visibility;
    // Under the film like every other diffuse term. The transmitted half added
    // above is not: it left the medium through the *other* face, which has its
    // own coat if it has one at all, and attenuating it here would take a
    // reflection off this side out of light that never touched it.
    lobes.diffusible += diffusible * (1.0 - Fc) * radiance * visibility;
    return lobes;
}

#endif
