#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord; // Local UV (0-1 within a tile)
layout (location = 2) in vec2 aAtlasOffset; // Atlas tile coordinates (e.g., [2.0, 1.0])

uniform mat4 projection;
uniform mat4 view;

out vec2 v_TexCoord;
out vec2 v_AtlasOffset;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0);
    v_TexCoord = aTexCoord;
    v_AtlasOffset = aAtlasOffset;
}
