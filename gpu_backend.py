# gpu_backend.py
# Contains OpenGL context management, shader loading utilities,
# and VBO/VAO wrapper classes for efficient GPU rendering.

import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import ctypes

class Shader:
    """A wrapper for an OpenGL shader program that handles resource cleanup."""
    def __init__(self, vertex_path: str, fragment_path: str):
        self.program = None # Initialize to None
        try:
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()
            with open(fragment_path, 'r') as f:
                fragment_src = f.read()

            # FIX: Raise a runtime error on compilation failure to prevent silent errors.
            vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            self.program = shaders.compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            # Provide a clear, actionable error message.
            raise RuntimeError(f"Shader compilation failed for '{vertex_path}' and '{fragment_path}': {e}")

    def use(self):
        """Activates the shader program."""
        if self.program:
            glUseProgram(self.program)

    def get_uniform_location(self, name: str) -> int:
        """Gets the location of a uniform variable in the shader."""
        if self.program:
            return glGetUniformLocation(self.program, name)
        return -1

    def destroy(self):
        """
        FIX: Frees the GPU memory associated with the shader program.
        This must be called to prevent VRAM leaks.
        """
        if self.program:
            glDeleteProgram(self.program)
            self.program = None

class QuadMesh:
    """
    A VBO/VAO wrapper for a simple 2D quad, used for instanced rendering.
    Now includes proper resource cleanup and robust attribute pointers.
    """
    def __init__(self):
        # Vertex data: position (x, y), texture coordinate (u, v)
        verts = np.array([
            -0.5, -0.5, 0, 0,
             0.5, -0.5, 1, 0,
             0.5,  0.5, 1, 1,
            -0.5,  0.5, 0, 1
        ], dtype=np.float32)

        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        self.ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # FIX: Calculate stride and offsets dynamically from the array's properties.
        # This makes the code robust against changes to the vertex format.
        stride = 4 * verts.itemsize  # 4 floats per vertex
        pos_offset = ctypes.c_void_p(0)
        tex_offset = ctypes.c_void_p(2 * verts.itemsize) # Offset is 2 floats from the start

        # Attribute 0: Vertex positions (2 floats)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, pos_offset)
        glEnableVertexAttribArray(0)
        # Attribute 1: Texture coordinates (2 floats)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, tex_offset)
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def draw(self):
        """Binds the VAO and draws the indexed quad."""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def destroy(self):
        """
        FIX: Frees the GPU memory associated with the mesh's VAO and VBOs.
        This must be called to prevent VRAM leaks.
        """
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.ibo:
            glDeleteBuffers(1, [self.ibo])
        self.vao, self.vbo, self.ibo = None, None, None
