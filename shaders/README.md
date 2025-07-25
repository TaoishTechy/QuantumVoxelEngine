## 🎛 Shader Usage Instructions (`/shaders/` directory)

### 1. File Structure and Placement

- Place all shader files (`.vert`, `.frag`) in the `/shaders/` directory.
- Maintain consistent naming across your source and codebase.

### 2. Writing Shaders

#### Vertex Shader Example (`chunk.vert`):
```glsl
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 projection;
uniform mat4 view;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
}
```

#### Fragment Shader Example (`chunk.frag`):
```glsl
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red
}
```

### 3. Referencing Shaders in Code

```python
import gpu_backend
shader = gpu_backend.Shader("shaders/chunk.vert", "shaders/chunk.frag")
shader.use()
```

- Ensure relative paths are correct
- Files must be readable and not blocked

### 4. Debugging Shader Loading

- Compilation errors printed to terminal
- Use `#version 330 core` at the top of each shader
- Match uniforms/attributes between GLSL and Python

### 5. Hot-reloading

- Edit shaders live but restart to reload them

### 6. Best Practices

- Use clean, well-commented shaders
- Document all uniforms and inputs
- Use consistent naming conventions

### 7. Common Troubleshooting

| Issue | Cause |
|-------|-------|
| Black screen | Compilation failed or unset uniforms |
| Artifacts | Attribute mismatch or uninitialized variables |
| Warnings | Check GLSL version and syntax |

---
