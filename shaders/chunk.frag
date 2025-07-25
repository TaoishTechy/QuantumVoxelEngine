#version 330 core

in vec2 TexCoord;
in float TexIndex;

out vec4 FragColor;

uniform sampler2DArray textureAtlas;

void main()
{
    FragColor = texture(textureAtlas, vec3(TexCoord, TexIndex));
}
