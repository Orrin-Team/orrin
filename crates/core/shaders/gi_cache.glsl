// The world radiance cache's shapes, shared by the passes that fill it and
// read it: a probe ray as the trace leaves it, a cell of the table, and the
// key a hit is filed under. `gfx/vulkan/gi.rs` describes the cache and holds
// the constants here in lock-step.

#ifndef ORRIN_GI_CACHE_GLSL
#define ORRIN_GI_CACHE_GLSL

const uint GI_NO_CELL = 0xffffffffu;

// What a probe ray met.
const uint GI_MISS = 0u;
const uint GI_FRONT = 1u;
const uint GI_BACK = 2u;

// One probe ray, as the trace leaves it for the update and the cache.
struct GiRay {
    // xyz = where it landed, w = how far it went. Unused for a miss.
    vec4 hit;
    // x = the cache cell it was filed under, or `GI_NO_CELL`; y = what it met.
    uvec4 meta;
};

// One cell of the table: the key's checksum, zero for an empty cell; the
// frame it was last used; the lowest-numbered ray that used it this frame,
// which is the one the cache shades it from.
struct GiCell {
    uint checksum;
    uint used;
    uint representative;
    uint unused;
};

// How many cells a key may look along from where it hashes to before it
// gives up and the ray goes uncached.
const uint GI_CACHE_PROBES = 8u;

// A cell is the finest size out to twice this many metres from the camera,
// and doubles with every doubling of the distance after: about as wide as
// what the probes there can resolve.
const float GI_CACHE_NEAR = 16.0;
const int GI_CACHE_LODS = 8;

// PCG, one step: a well-mixed word from any word.
uint gi_hash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// The key a hit at `P` on a surface facing `N` is filed under, `reach` from
// the probe that found it, as two words: the cell's position at its level of
// detail, the level, and which of six ways the surface faces — so the two
// sides of a thin wall are two cells, and light on one is not read on the
// other. A ray shorter than its cell is filed at the finest level whatever its
// distance, since a cell wider than the ray reaches past what the ray could
// see — round a corner, or through the wall it stopped at (Solari).
uvec2 gi_cell_key(vec3 P, vec3 N, float reach, vec3 camera, float finest) {
    float far = max(distance(P, camera), GI_CACHE_NEAR) / GI_CACHE_NEAR;
    int lod = clamp(int(floor(log2(far))), 0, GI_CACHE_LODS - 1);
    float cell = finest * exp2(float(lod));
    if (reach < cell) {
        lod = 0;
        cell = finest;
    }
    ivec3 q = ivec3(floor(P / cell));
    vec3 a = abs(N);
    uint axis = a.x >= a.y && a.x >= a.z ? 0u : (a.y >= a.z ? 1u : 2u);
    uint face = axis * 2u + (N[axis] < 0.0 ? 1u : 0u);
    uint lo = (uint(q.x) & 0x7ffffu) | ((uint(q.y) & 0x1fffu) << 19u);
    uint hi = ((uint(q.y) >> 13u) & 0x3fu) | ((uint(q.z) & 0x7ffffu) << 6u) | (uint(lod) << 25u)
            | (face << 28u);
    return uvec2(lo, hi);
}

// Where a key starts looking in a table of `mask + 1` cells.
uint gi_cell_home(uvec2 key, uint mask) {
    return gi_hash(key.x ^ gi_hash(key.y)) & mask;
}

// A second hash of the key, stored in its cell to tell it from another key
// that hashed to the same place. Never zero, which marks an empty cell.
uint gi_cell_checksum(uvec2 key) {
    return max(gi_hash(key.y ^ gi_hash(key.x ^ 0x9e3779b9u)), 1u);
}

#endif
