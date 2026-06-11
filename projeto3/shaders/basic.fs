#version 330 core

in vec2 v_uv;
in vec3 v_world_pos;
in vec3 v_normal;

uniform sampler2D u_tex;
uniform int  u_wireframe;

uniform int u_environment; // 0 = externo, 1 = interno
uniform vec3 u_base_color;
uniform float u_material_diffuse;
uniform float u_material_specular;
uniform float u_shininess;

uniform vec3 u_view_pos;

uniform int u_ambient_enabled;
uniform float u_ambient_strength;
uniform float u_diffuse_gain;
uniform float u_specular_gain;

uniform int u_sun_enabled;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;

uniform int u_boat_light_enabled;
uniform vec3 u_boat_light_pos;
uniform vec3 u_boat_light_color;
uniform float u_boat_light_intensity;

uniform int u_lightbulb_enabled;
uniform vec3 u_lightbulb_pos;
uniform vec3 u_lightbulb_color;
uniform float u_lightbulb_intensity;

uniform int u_flashlight_enabled;
uniform vec3 u_flashlight_pos;
uniform vec3 u_flashlight_dir;
uniform vec3 u_flashlight_color;
uniform float u_flashlight_intensity;
uniform float u_flashlight_inner_cutoff;
uniform float u_flashlight_outer_cutoff;

out vec4 frag_color;

vec3 apply_directional_light(
    vec3 base,
    vec3 normal,
    vec3 view_dir,
    vec3 direction,
    vec3 color,
    float intensity
) {
    vec3 light_dir = normalize(-direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), u_shininess);

    vec3 diffuse = base * u_material_diffuse * u_diffuse_gain * diff;
    vec3 specular = vec3(u_material_specular * u_specular_gain * spec);
    return (diffuse + specular) * color * intensity;
}

vec3 apply_point_light(
    vec3 base,
    vec3 normal,
    vec3 view_dir,
    vec3 light_pos,
    vec3 color,
    float intensity
) {
    vec3 to_light = light_pos - v_world_pos;
    float distance_to_light = length(to_light);
    vec3 light_dir = normalize(to_light);

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), u_shininess);

    float attenuation = 1.0 / (1.0 + 0.09 * distance_to_light + 0.032 * distance_to_light * distance_to_light);
    vec3 diffuse = base * u_material_diffuse * u_diffuse_gain * diff;
    vec3 specular = vec3(u_material_specular * u_specular_gain * spec);
    return (diffuse + specular) * color * intensity * attenuation;
}

vec3 apply_spot_light(
    vec3 base,
    vec3 normal,
    vec3 view_dir,
    vec3 light_pos,
    vec3 spot_dir,
    vec3 color,
    float intensity,
    float inner_cutoff,
    float outer_cutoff
) {
    vec3 to_light = light_pos - v_world_pos;
    float distance_to_light = length(to_light);
    vec3 light_dir = normalize(to_light);
    vec3 from_light_dir = normalize(v_world_pos - light_pos);

    float theta = dot(from_light_dir, normalize(spot_dir));
    float cone = clamp((theta - outer_cutoff) / max(inner_cutoff - outer_cutoff, 0.001), 0.0, 1.0);

    float diff = max(dot(normal, light_dir), 0.0);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), u_shininess);

    float attenuation = 1.0 / (1.0 + 0.09 * distance_to_light + 0.032 * distance_to_light * distance_to_light);
    vec3 diffuse = base * u_material_diffuse * u_diffuse_gain * diff;
    vec3 specular = vec3(u_material_specular * u_specular_gain * spec);
    return (diffuse + specular) * color * intensity * attenuation * cone;
}

void main() {
    if (u_wireframe == 1) {
        frag_color = vec4(0.95, 0.95, 0.10, 1.0);
        return;
    }

    vec3 base = texture(u_tex, v_uv).rgb * u_base_color;
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(u_view_pos - v_world_pos);

    vec3 color = vec3(0.0);
    if (u_ambient_enabled == 1) {
        color += base * u_ambient_strength;
    }

    if (u_environment == 0) {
        if (u_sun_enabled == 1) {
            color += apply_directional_light(
                base, normal, view_dir,
                u_sun_direction, u_sun_color, u_sun_intensity
            );
        }
        if (u_boat_light_enabled == 1) {
            color += apply_point_light(
                base, normal, view_dir,
                u_boat_light_pos, u_boat_light_color, u_boat_light_intensity
            );
        }
    } else {
        if (u_lightbulb_enabled == 1) {
            color += apply_point_light(
                base, normal, view_dir,
                u_lightbulb_pos, u_lightbulb_color, u_lightbulb_intensity
            );
        }
        if (u_flashlight_enabled == 1) {
            color += apply_spot_light(
                base, normal, view_dir,
                u_flashlight_pos, u_flashlight_dir,
                u_flashlight_color, u_flashlight_intensity,
                u_flashlight_inner_cutoff, u_flashlight_outer_cutoff
            );
        }
    }

    frag_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
