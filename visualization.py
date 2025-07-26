# visualization.py (GPU Accelerated)
# Renders quantum effects using an OpenGL backend for high performance.

import pygame
import numpy as np
import math
import random
from OpenGL.GL import *
import gpu_backend # Import the corrected backend
import ctypes

class QuantumVisualizationEngine:
    """
    High-performance renderer for quantum effects using OpenGL VBOs.
    Now includes proper resource cleanup.
    """

    def __init__(self, screen):
        self.width, self.height = screen.get_size()
        self.effects = []

        # --- GPU Setup ---
        self.quad_mesh = gpu_backend.QuadMesh()
        self.cloud_shader = gpu_backend.Shader("shaders/cloud.vert", "shaders/cloud.frag")

        # Create a single large VBO for all particle instances
        self.max_particles = 20000
        # Particle format: x, y, radius, r, g, b, a (7 floats)
        self.instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_particles * 7 * 4, None, GL_DYNAMIC_DRAW)

        # Configure instanced vertex attributes
        glBindVertexArray(self.quad_mesh.vao)
        stride = 7 * 4 # 7 floats * 4 bytes/float
        # Attribute 2: Position (x, y, radius)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1) # This attribute is per-instance
        # Attribute 3: Color (r, g, b, a)
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1) # This attribute is per-instance
        glBindVertexArray(0)

        # Create a soft particle texture
        self.particle_texture = self.create_soft_particle_texture(64)

    def create_soft_particle_texture(self, size: int) -> int:
        """Creates a simple texture of a soft, blurred circle."""
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # Create a radial gradient
        data = np.zeros((size, size), dtype=np.uint8)
        center = size / 2.0
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
                intensity = 255 * max(0, 1.0 - dist / center)
                data[y, x] = int(intensity)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture

    def update(self, dt: float, entity_iterable):
        """Update all visual effects."""
        # This is where transient effects would be created, updated, and timed out
        pass

    def draw(self, projection_matrix, view_matrix):
        """Draws all quantum effects using a single instanced draw call."""
        instance_data = []

        # Example: Collect data for all visible particles
        # In a real game, this would come from an effects system or entity list
        for i in range(10): # Draw 10 example quantum objects
            pos = [random.uniform(100, self.width - 100), random.uniform(100, self.height - 100)]
            radius = random.uniform(10, 50)
            color = [0.0, 1.0, 1.0, random.uniform(0.1, 0.5)]
            instance_data.extend([pos[0], pos[1], radius, *color])

        if not instance_data:
            return

        # --- Render Pass ---
        glEnable(GL_BLEND)
        glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE) # Additive blending for a glow effect

        self.cloud_shader.use()
        glUniformMatrix4fv(self.cloud_shader.get_uniform_location("projection"), 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(self.cloud_shader.get_uniform_location("view"), 1, GL_FALSE, view_matrix)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.particle_texture)
        glUniform1i(self.cloud_shader.get_uniform_location("particleTexture"), 0)

        # Update the VBO with the new instance data
        instance_data_np = np.array(instance_data, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data_np.nbytes, instance_data_np)

        # Perform the instanced draw call
        glBindVertexArray(self.quad_mesh.vao)
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, len(instance_data) // 7)
        glBindVertexArray(0)

        glDisable(GL_BLEND)

    def destroy(self):
        """
        FIX: Frees all GPU resources held by the engine.
        This must be called on shutdown to prevent VRAM leaks.
        """
        if self.particle_texture:
            glDeleteTextures(1, [self.particle_texture])
        if self.instance_vbo:
            glDeleteBuffers(1, [self.instance_vbo])

        self.quad_mesh.destroy()
        self.cloud_shader.destroy()

        self.particle_texture, self.instance_vbo = None, None
