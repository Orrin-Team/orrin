// Storing a running mean in a half float, which the probes' atlases and the
// final gather's accumulation both do.

#ifndef ORRIN_HALF_FLOAT_GLSL
#define ORRIN_HALF_FLOAT_GLSL

// The half float nearest `v`, as a float the store converts exactly. A store
// to a half-float image may truncate (RDNA packs with round-toward-zero), and
// a running mean truncated every frame settles below its true mean by about
// as many half units in the last place as it averages frames: three parts in
// a hundred at a 32-frame history, and seven at 256.
//
// Not rounded up or down by chance, though a mean of N frames is not moved at
// all by a frame within N/4096 to N/2048 of it: a mean is that long only
// where its frames disagree by far more than that, and in Sponza's aisle a
// chance rounding moved no region of the probes or the gather by more than
// two parts in a hundred.
vec4 half_nearest(vec4 v) {
    ivec4 e;
    frexp(v, e);
    vec4 unit = exp2(vec4(max(e, ivec4(-13)) - 11));
    return clamp(roundEven(v / unit) * unit, vec4(-65504.0), vec4(65504.0));
}

#endif
