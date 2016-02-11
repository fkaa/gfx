#version 150 core
#

#define MAX_TILES_W ((800 + 32 - 1) / 32)
#define MAX_TILES_H ((600 + 32 - 1) / 32)

#define MAX_LIGHTS 100 

uniform isamplerBuffer u_LightIndexTex;

struct LightInfo {
    vec3 pos;
    float radius;
    vec4 color;
};

uniform u_LightData {
    LightInfo lights[MAX_LIGHTS];
};

uniform u_LightGrid {
    // .x = offset
    // .y = count
    ivec2 tiles[475];
};

in vec3 v_Pos;
in vec3 v_Normal;
in vec3 v_Color;

out vec4 o_Color;

void main() {
    ivec2 tile_idx = ivec2(gl_FragCoord) / ivec2(32);
    ivec2 tile = tiles[tile_idx.x + tile_idx.y * MAX_TILES_W];

    vec3 ambient = v_Color * 0.1;
    vec3 result = vec3(0.0);

    for (int idx = 0; idx < tile.y; ++idx) {
        int id = texelFetch(u_LightIndexTex, tile.x + idx).x;
        LightInfo light = lights[id];

        vec3 light_dir = light.pos - v_Pos;
        float light_dist = length(light_dir);
        light_dir = normalize(light_dir);

        float inner = 0.0;
        float ndot = max(dot(v_Normal, light_dir), 0.0);

        float attenuation = max(1.0 - max(0.0, (light_dist - inner) / (light.radius-inner)), 0.0);
        result += ndot * v_Color;
    }

    //o_Color = vec4(result + ambient, 1.0);
    o_Color = vec4(vec3(float(tile.y)), 1.0);
    //o_Color = vec4(vec3(float(tile_idx.x + tile_idx.y * MAX_TILES_W) / 475.0), 1.0);
}
