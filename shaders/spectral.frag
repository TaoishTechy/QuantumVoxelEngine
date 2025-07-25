#version 330 core

out vec4 FragColor;
uniform float wavelength_nm; // Input wavelength in nanometers

// Physically-inspired conversion from wavelength to RGB
// Based on polynomial approximations of CIE 1931 color matching functions.
vec3 wavelengthToRGB(float l){ // l in nm
    float t = clamp((l - 380.0) / (700.0 - 380.0), 0.0, 1.0);
    float r = 1.6419 * pow(t, 3.0) - 2.6578 * pow(t, 2.0) + 0.9776 * t + 0.0640;
    float g = -0.4712 * pow(t, 3.0) + 1.9071 * pow(t, 2.0) + 0.0654 * t + 0.0582;
    float b = 0.0736 * pow(t, 3.0) - 0.2280 * pow(t, 2.0) + 1.2803 * t - 0.0503;
    return clamp(vec3(r, g, b), 0.0, 1.0);
}

void main()
{
    vec3 color = wavelengthToRGB(wavelength_nm);
    FragColor = vec4(color, 1.0);
}
