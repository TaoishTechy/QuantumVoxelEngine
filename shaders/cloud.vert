#version 330 core

// Per-quad attributes
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

// Per-instance attributes
layout (location = 2) in vec3 aInstancePosition; // (x, y, radius)
layout (location = 3) in vec4 aInstanceColor;    // (r, g, b, a)

uniform mat4 projection;
uniform mat4 view;

out vec2 TexCoord;
out vec4 ParticleColor;

void main()
{
    // Calculate the screen-space position of the particle
    vec4 world_pos = vec4(aInstancePosition.xy, 0.0, 1.0);
    vec4 view_pos = view * world_pos;

    // Billboard the quad to always face the camera
    view_pos.xy += aPos * aInstancePosition.z; // aInstancePosition.z stores radius

    gl_Position = projection * view_pos;

    TexCoord = aTexCoord;
    ParticleColor = aInstanceColor;
}
