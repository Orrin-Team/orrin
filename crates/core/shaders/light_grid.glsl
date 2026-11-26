// Which punctual lights reach a point in the world: the lookup half of
// `gfx/light_grid.rs`, which bins them. A ray's hit loops over its cell's
// lists as a fragment loops over its cluster's.
//
// `world_light_cell` here and `cell_of` there are a lock-step pair, as are the
// two grid constants: that module's test holds the binning conservative against
// exactly this lookup.
//
// The includer says where the three bindings are:
//
//     #define ORRIN_LIGHT_GRID_SET
//     #define ORRIN_LIGHT_GRID_BINDING
//     #define ORRIN_LIGHT_GRID_CELLS_BINDING
//     #define ORRIN_LIGHT_GRID_LIGHTS_BINDING

#ifndef ORRIN_LIGHT_GRID_GLSL
#define ORRIN_LIGHT_GRID_GLSL

const float LIGHT_GRID_CELLS = 16.0;
const int LIGHT_GRID_CASCADES = 8;

layout(set = ORRIN_LIGHT_GRID_SET, binding = ORRIN_LIGHT_GRID_BINDING) uniform LightGrid {
    // xyz = the cascade's minimum corner, w = the edge of one of its cells.
    vec4 cascades[LIGHT_GRID_CASCADES];
} light_grid;

// Per cell: x = where its list starts, y = its point count in the low sixteen
// bits and its spot count in the high. The points come first.
layout(set = ORRIN_LIGHT_GRID_SET, binding = ORRIN_LIGHT_GRID_CELLS_BINDING, std430)
readonly buffer LightGridCells {
    uvec2 light_grid_cells[];
};

layout(set = ORRIN_LIGHT_GRID_SET, binding = ORRIN_LIGHT_GRID_LIGHTS_BINDING, std430)
readonly buffer LightGridLights {
    uint light_grid_lights[];
};

// The lists of the cell `position` reads, in the finest cascade that contains
// it: x = the first point light's entry, y = how many points, z = how many
// spots, which follow them. No lights past the last cascade.
uvec3 world_light_cell(vec3 position) {
    for (int level = 0; level < LIGHT_GRID_CASCADES; ++level) {
        vec4 cascade = light_grid.cascades[level];
        vec3 local = floor((position - cascade.xyz) / cascade.w);
        if (all(greaterThanEqual(local, vec3(0.0)))
            && all(lessThan(local, vec3(LIGHT_GRID_CELLS)))) {
            float cell = (local.z * LIGHT_GRID_CELLS + local.y) * LIGHT_GRID_CELLS + local.x
                       + float(level) * LIGHT_GRID_CELLS * LIGHT_GRID_CELLS * LIGHT_GRID_CELLS;
            uvec2 range = light_grid_cells[uint(cell)];
            return uvec3(range.x, range.y & 0xffffu, range.y >> 16u);
        }
    }
    return uvec3(0u);
}

#endif
