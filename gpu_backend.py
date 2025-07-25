# gpu_backend.py
# Contains OpenGL context management, shader loading utilities,
# and VBO/VAO wrapper classes for efficient GPU rendering.

import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np

class Shader:
    """A wrapper for an OpenGL shader program."""
    def __init__(self, vertex_path, fragment_path):
        self.program = self.compile_shader(vertex_path, fragment_path)

    def compile_shader(self, vertex_path, fragment_path):
        """Loads and compiles a vertex and fragment shader."""
        with open(vertex_path, 'r') as f:
            vertex_src = f.read()
        with open(fragment_path, 'r') as f:
            fragment_src = f.read()

        try:
            vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            program = shaders.compileProgram(vertex_shader, fragment_shader)
            return program
        except Exception as e:
            print("ERROR: Shader compilation failed!")
            print(e)
            return None

    def use(self):
        """Activates the shader program."""
        if self.program:
            glUseProgram(self.program)

    def get_uniform_location(self, name):
        """Gets the location of a uniform variable in the shader."""
        return glGetUniformLocation(self.program, name)

class QuadMesh:
    """A VBO/VAO wrapper for a simple 2D quad, used for instanced rendering."""
    def __init__(self):
        verts = np.array([
            # x,  y,   u, v
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

        # Vertex positions
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Texture coordinates
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def draw(self):
        """Binds the VAO and draws the indexed quad."""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
