#version 330 core

layout (location = 0) in vec3 in_position;

uniform mat4 u_view;   // já sem translação
uniform mat4 u_proj;

out vec3 v_dir;

void main() {
    v_dir = in_position;
    // truque clássico: gl_Position.z = w garante depth = 1.0 no clip space,
    // permitindo que a skybox apareça SEMPRE no fundo.
    vec4 pos = u_proj * u_view * vec4(in_position, 1.0);
    gl_Position = pos.xyww;
}
