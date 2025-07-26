# rendering.py
# Contains all OpenGL rendering logic, including the Renderer, InputHandler,
# and VBO management classes.

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
from typing import Dict, Tuple

import config
import core_world
import gpu_backend

class ChunkVBO:
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ibo = glGenBuffers(1)
        self.index_count = 0
        self.is_uploaded = False

    def upload_data(self, vertex_data, index_data):
        if vertex_data.size == 0:
            self.index_count = 0
            return
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)
        stride = 28
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        self.index_count = len(index_data)
        self.is_uploaded = True

    def draw(self):
        if self.is_uploaded and self.index_count > 0:
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(2, [self.vbo, self.ibo])

class Renderer:
    def __init__(self, screen, world):
        self.screen, self.world = screen, world
        self.width, self.height = screen.get_size()
        self.chunk_vbos: Dict[Tuple[int, int], ChunkVBO] = {}
        self.camera_pos = np.array([0.0, 85.0, 0.0])
        self.camera_yaw, self.camera_pitch = -np.pi / 2, 0
        self.shader = gpu_backend.Shader("shaders/chunk.vert", "shaders/chunk.frag")
        self.texture_atlas = self.load_texture_atlas("atlas.png")

    def load_texture_atlas(self, path):
        # ... (load_texture_atlas remains the same)
        # Add fallback texture logic here
        pass

    def draw_world(self):
        # ... (draw_world remains the same)
        pass

class InputHandler:
    def __init__(self, world, renderer):
        self.world, self.renderer = world, renderer
        self.player = world.entities[0] if world.entities else None
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def handle_input(self, dt):
        # ... (handle_input remains the same, but reads from config)
        pass

    def handle_events(self, event):
        # ... (handle_events remains the same)
        pass
