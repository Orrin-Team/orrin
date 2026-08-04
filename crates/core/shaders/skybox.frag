#version 460

layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 f_color;

// Separated for the reason forward.frag documents: Metal allows far fewer
// sampler states per stage than sampled images, so samplers are shared rather
// than combined into the binding.
layout(set = 0, binding = 0) uniform textureCube u_environment;
layout(set = 0, binding = 1) uniform sampler u_environment_sampler;

layout(push_constant) uniform Push {
    mat4 inv_view_rot_proj;
    vec4 params; // x = intensity
} push;

void main() {
    vec3 dir = normalize(v_dir);
    vec3 sky = textureLod(samplerCube(u_environment, u_environment_sampler), dir, 0.0).rgb;
    f_color = vec4(sky * push.params.x, 1.0);
}
