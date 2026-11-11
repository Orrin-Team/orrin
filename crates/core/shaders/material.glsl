// The material record, once. `shading.glsl`, `prepass.frag` and the masked
// `shadow.frag` read the same buffer of these, each through a descriptor set of
// its own, so the row is declared here and the binding is left to the includer.
//
// Mirrors GpuMaterial in forward.rs, field for field. std430 packs this exactly
// like the Rust #[repr(C)] struct because every field is 16 bytes, and it takes
// the array stride from the struct: a pass that declared only the fields it
// reads would step through the table at the wrong pitch and read every material
// past the first out of the middle of its neighbour. One declaration is what
// makes growing the row safe.

#ifndef ORRIN_MATERIAL_GLSL
#define ORRIN_MATERIAL_GLSL

struct GpuMaterial {
    vec4 base_color;   // rgb = albedo, a = opacity
    vec4 emissive;     // rgb = emissive
    vec4 params;       // x = metallic, y = roughness, z = reflectance, w = ior
    uvec4 tex_indices; // x=albedo, y=normal, z=metal-rough, w=emissive
    vec4 clearcoat;    // x = strength, y = perceptual roughness
    vec4 sheen;        // rgb = colour, a = perceptual roughness
    vec4 anisotropy;   // x = strength, y = cos(rotation), z = sin(rotation)
    vec4 transmission; // x = transmission, y = thickness, z = attenuation distance
    vec4 attenuation;  // rgb = what the volume absorbs over `transmission.z`
    vec4 subsurface;   // rgb = scattering tint, a = forward-scatter power
    vec4 subsurface_radius; // rgb = per-channel mean free path, metres
    vec4 parallax;     // x = height field depth in metres, y = min steps, z = max steps
    vec4 alpha;        // x = the alpha a MASKED fragment must reach to survive
    uvec4 tex_indices_ext; // x=clearcoat, y=clearcoat normal, z=sheen, w=anisotropy
    uvec4 tex_flags;       // x = transmission map, y = feature flags, z = subsurface, w = height
    uvec4 surface;         // x = the first word of this material's parameters in SurfaceParams
};

// Feature bits in `tex_flags.y`, mirroring `material_flags` in forward.rs.
//
// `v_material` is one value for a whole draw, so every test against these is a
// coherent branch: a draw either takes a lobe or does not, and a material that
// never asked for one pays a single comparison rather than the lobe's cost.
#ifdef ORRIN_FLAT_SURFACE
// The flat forward pipelines' build: a material drawn with one uses none of the
// optional lobes, so every test of their bits folds to false and the compiler
// removes the lobes outright. Their presence alone costs every material a
// quarter of the forward pass — see `ForwardPipelines` in forward.rs.
const uint MATERIAL_CLEARCOAT    = 0u;
const uint MATERIAL_SHEEN        = 0u;
const uint MATERIAL_ANISOTROPY   = 0u;
const uint MATERIAL_TRANSMISSION = 0u;
const uint MATERIAL_SUBSURFACE   = 0u;
const uint MATERIAL_PARALLAX     = 0u;
#else
const uint MATERIAL_CLEARCOAT    = 1u << 0;
const uint MATERIAL_SHEEN        = 1u << 1;
const uint MATERIAL_ANISOTROPY   = 1u << 2;
const uint MATERIAL_TRANSMISSION = 1u << 3;
const uint MATERIAL_SUBSURFACE   = 1u << 4;
const uint MATERIAL_PARALLAX     = 1u << 5;
#endif
const uint MATERIAL_MASKED       = 1u << 6;

#endif
