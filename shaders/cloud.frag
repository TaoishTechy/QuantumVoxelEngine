#version 330 core

in vec2 TexCoord;
in vec4 ParticleColor;

out vec4 FragColor;

uniform sampler2D particleTexture; // A simple soft circle texture

void main()
{
    // A soft particle effect using a texture
    float alpha = texture(particleTexture, TexCoord).r * ParticleColor.a;
    FragColor = vec4(ParticleColor.rgb, alpha);
}
