#version 330 core

in vec3 v_dir;

uniform samplerCube u_cubemap;
uniform float u_brightness;

out vec4 frag_color;

void main() {
    vec4 texel = texture(u_cubemap, v_dir);
    frag_color = vec4(texel.rgb * u_brightness, texel.a);
}
