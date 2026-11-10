// ACES filmic tonemap approximation (Narkowicz 2015), operating on
// linear radiance. Shared by the frame's tonemap and the material preview's,
// so a material is judged through the curve it will be seen through.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
