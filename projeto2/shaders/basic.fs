#version 330 core

in vec2 v_uv;

uniform sampler2D u_tex;
uniform vec3 u_kd;          // cor base (multiplicativa)
uniform int  u_wireframe;   // 1 = override em cor sólida

out vec4 frag_color;

void main() {
    if (u_wireframe == 1) {
        // amarelo neon — fica destacado contra grama/lago/madeira
        frag_color = vec4(0.95, 0.95, 0.10, 1.0);
        return;
    }
    vec4 texel = texture(u_tex, v_uv);
    frag_color = vec4(texel.rgb * u_kd, 1.0);
}
