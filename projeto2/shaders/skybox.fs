#version 330 core

in vec3 v_dir;

uniform samplerCube u_cubemap;

out vec4 frag_color;

void main() {
    frag_color = texture(u_cubemap, v_dir);
}
