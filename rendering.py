# rendering.py
# Contains all OpenGL rendering logic, including the Renderer, InputHandler,
# and VBO management classes.

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
from typing import Dict, Tuple, Optional

import config
import core_world
import gpu_backend
from logger import logger
from ai_systems import PINNTextureOptimizer

class ChunkVBO:
    """Manages the VAO, VBO, and IBO for a single chunk's mesh."""
    def __init__(self):
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ibo = glGenBuffers(1)
        self.index_count = 0
        self.is_uploaded = False

    def upload_data(self, vertex_data: np.ndarray, index_data: np.ndarray):
        """Uploads mesh data to the GPU buffers."""
        if vertex_data.size == 0:
            self.index_count = 0
            return
            
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)
        
        stride = 28 # 7 floats * 4 bytes
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
        """Draws the chunk mesh."""
        if self.is_uploaded and self.index_count > 0:
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def destroy(self):
        """Explicitly deletes the GPU buffers to prevent memory leaks."""
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(2, [self.vbo, self.ibo])

class Renderer:
    """Handles all rendering for the game."""
    def __init__(self, screen, world):
        self.screen, self.world = screen, world
        self.width, self.height = screen.get_size()
        self.chunk_vbos: Dict[Tuple[int, int], ChunkVBO] = {}
        self.camera_pos = np.array([0.0, 85.0, 0.0])
        self.camera_yaw, self.camera_pitch = -np.pi / 2, 0
        self.shader = gpu_backend.Shader("shaders/chunk.vert", "shaders/chunk.frag")
        self.pinn_optimizer = PINNTextureOptimizer()
        self.texture_atlas = self.load_texture_atlas("atlas.png")

    def load_texture_atlas(self, path: str) -> Optional[int]:
        """Loads the texture atlas and generates AI-powered mipmaps."""
        try:
            image = pygame.image.load(path).convert_alpha()
            image_data = pygame.image.tostring(image, "RGBA", True)
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            
            # TODO: Implement AI-powered mipmapping using the PINN optimizer
            # mipmaps = self.pinn_optimizer.generate_mipmaps(image_data)
            # For now, use standard mipmapping
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.get_width(), image.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
            logger.info(f"Texture atlas '{path}' loaded successfully.")
            return tex_id
        except Exception as e:
            logger.error(f"Could not load texture atlas '{path}': {e}")
            # TODO: Implement a fallback "missing texture" asset
            return None

    def draw_world(self):
        """Draws all visible and loaded chunks."""
        self.shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_atlas)
        
        proj = glGetFloatv(GL_PROJECTION_MATRIX)
        view = glGetFloatv(GL_MODELVIEW_MATRIX)
        glUniformMatrix4fv(self.shader.get_uniform_location("projection"), 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.shader.get_uniform_location("view"), 1, GL_FALSE, view)

        cam_chunk_x, cam_chunk_z = self.world.get_chunk_coord(self.camera_pos[0], self.camera_pos[2])
        
        for cx in range(cam_chunk_x - config.RENDER_DISTANCE, cam_chunk_x + config.RENDER_DISTANCE):
            for cz in range(cam_chunk_z - config.RENDER_DISTANCE, cam_chunk_z + config.RENDER_DISTANCE):
                chunk_coord = (cx, cz)
                if chunk_coord in self.chunk_vbos and isinstance(self.chunk_vbos[chunk_coord], ChunkVBO):
                    self.chunk_vbos[chunk_coord].draw()

    def destroy(self):
        """Cleans up all GPU resources managed by the renderer."""
        logger.info("Destroying renderer resources...")
        for vbo in self.chunk_vbos.values():
            if isinstance(vbo, ChunkVBO):
                vbo.destroy()
        if self.texture_atlas:
            glDeleteTextures([self.texture_atlas])

class InputHandler:
    """Translates user input into game actions."""
    def __init__(self, world, renderer):
        self.world, self.renderer = world, renderer
        self.player = world.entities[0] if world.entities else None
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def handle_input(self, dt: float):
        """Processes continuous input like mouse movement and key presses."""
        if not self.player: return
        dx, dy = pygame.mouse.get_rel()
        self.renderer.camera_yaw += dx * config.MOUSE_SENSITIVITY
        self.renderer.camera_pitch -= dy * config.MOUSE_SENSITIVITY
        self.renderer.camera_pitch = np.clip(self.renderer.camera_pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        keys = pygame.key.get_pressed()
        forward = np.array([np.cos(self.renderer.camera_yaw), 0, np.sin(self.renderer.camera_yaw)])
        right = np.array([-np.sin(self.renderer.camera_yaw), 0, np.cos(self.renderer.camera_yaw)])
        
        move_vec = np.zeros(3)
        if keys[pygame.K_w]: move_vec += forward
        if keys[pygame.K_s]: move_vec -= forward
        if keys[pygame.K_a]: move_vec -= right
        if keys[pygame.K_d]: move_vec += right
        
        if np.linalg.norm(move_vec) > 0:
            move_vec /= np.linalg.norm(move_vec)
        
        self.player.vel[0] = move_vec[0] * config.PLAYER_MOVE_SPEED
        self.player.vel[2] = move_vec[2] * config.PLAYER_MOVE_SPEED
        
        self.renderer.camera_pos = self.player.pos + np.array([0, 0.8, 0])

    def handle_events(self, event: pygame.event.Event):
        """Handles discrete events like jumping."""
        if not self.player: return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self.player.on_ground:
            self.player.vel[1] = config.PLAYER_JUMP_FORCE
