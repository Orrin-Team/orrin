#version 460

layout(location = 0) in vec3 v_view_normal;
layout(location = 0) out vec4 f_normal;

void main() {
    f_normal = vec4(normalize(v_view_normal) * 0.5 + 0.5, 1.0);
}
