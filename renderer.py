# renderer.py
# The main rendering engine for the Quantum Voxel Engine.
# Handles all OpenGL calls, shader management, and mesh generation/rendering for the world.

import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import ctypes
import random

# --- Engine Imports ---
import config
from settings import settings
import core_world
import gpu_backend

# =============================================================================
# ARTIST & CONTENT DESIGNER SECTION
# To add a new block style, add an entry to this dictionary.
# =============================================================================

BLOCK_FACE_STYLES = {
    # Default style for unknown blocks
    "default": {
        "color": [1.0, 0.0, 1.0], # Magenta
        "pattern": None
    },
    core_world.BlockType.STONE: {
        "color": [0.5, 0.5, 0.5],
        "pattern": "random_noise",
        "pattern_intensity": 0.05
    },
    core_world.BlockType.DIRT: {
        "color": [0.6, 0.4, 0.2],
        "pattern": None
    },
    core_world.BlockType.GRASS: {
        # FIX: Added a new pattern for the top face of grass blocks.
        "top": {"color": [0.2, 0.8, 0.2], "pattern": "grass_top"},
        "bottom": {"color": [0.6, 0.4, 0.2], "pattern": None}, # Dirt color
        "side": {"color": [0.6, 0.4, 0.2], "pattern": None}    # Dirt color
    },
    core_world.BlockType.WOOD: {
        "color": [0.7, 0.5, 0.3],
        "pattern": None
    },
    core_world.BlockType.LAVA: {
        "color": [1.0, 0.5, 0.0],
        "pattern": None
    },
    core_world.BlockType.WATER: {
        "color": [0.2, 0.5, 1.0],
        "pattern": None
    },
    core_world.BlockType.OBSIDIAN: {
        "color": [0.1, 0.1, 0.2],
        "pattern": None
    },
    core_world.BlockType.SAND: {
        "color": [0.9, 0.9, 0.7],
        "pattern": "random_noise",
        "pattern_intensity": 0.08
    },
    # --- New Block Styles ---
    core_world.BlockType.LEAVES: {
        "color": [0.1, 0.5, 0.1],
        "pattern": None
    },
    core_world.BlockType.BRICK: {
        "color": [0.7, 0.25, 0.15],
        "pattern": "bricks"
    },
    core_world.BlockType.METAL: {
        "color": [0.8, 0.8, 0.85],
        "pattern": "stripes",
        "pattern_intensity": 0.1
    },
    core_world.BlockType.ICE: {
        "color": [0.8, 0.95, 1.0],
        "pattern": None
    },
    core_world.BlockType.GLOWSTONE: {
        "color": [1.0, 0.9, 0.6],
        "pattern": None
    },
    core_world.BlockType.SANDSTONE: {
        "top": {"color": [0.85, 0.8, 0.6], "pattern": None},
        "bottom": {"color": [0.85, 0.8, 0.6], "pattern": None},
        "side": {"color": [0.9, 0.85, 0.65], "pattern": "random_noise", "pattern_intensity": 0.05}
    }
}

# =============================================================================
# END OF ARTIST & CONTENT DESIGNER SECTION
# =============================================================================


def create_projection_matrix(fov, aspect_ratio, near_plane, far_plane):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
    matrix[2, 3] = -1.0
    matrix[3, 2] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
    return matrix

def create_view_matrix(position, yaw, pitch):
    yaw_rad, pitch_rad = np.radians(yaw), np.radians(pitch)
    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)

    xaxis = np.array([cos_yaw, 0, -sin_yaw])
    yaxis = np.array([sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch])
    zaxis = np.array([sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw])

    rotation_matrix = np.array([
        [xaxis[0], yaxis[0], zaxis[0], 0],
        [xaxis[1], yaxis[1], zaxis[1], 0],
        [xaxis[2], yaxis[2], zaxis[2], 0],
        [0, 0, 0, 1]], dtype=np.float32)

    translation_matrix = np.identity(4, dtype=np.float32)
    translation_matrix[3, 0:3] = -position
    return translation_matrix @ rotation_matrix

class ChunkMesh:
    def __init__(self, vertices):
        self.vertex_count = int(len(vertices) / 6)
        if self.vertex_count == 0:
            self.vao, self.vbo = None, None
            return

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def draw(self):
        if self.vao:
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        if self.vbo: glDeleteBuffers(1, [self.vbo])
        if self.vao: glDeleteVertexArrays(1, [self.vao])
        self.vao, self.vbo = None, None

class Renderer:
    def __init__(self, screen, world):
        self.screen, self.world = screen, world
        self.width, self.height = screen.get_size()
        self.camera_orientation_provider = lambda: (0, 0)
        self.player_position_provider = lambda: [0, 0, 0]
        self.chunk_meshes, self.wireframe_mode = {}, False

        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos; layout (location = 1) in vec3 aColor;
        out vec3 FragColor;
        uniform mat4 projection; uniform mat4 view; uniform mat4 model;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            FragColor = aColor;
        }"""
        fragment_shader_source = """
        #version 330 core
        in vec3 FragColor; out vec4 FinalColor;
        void main() { FinalColor = vec4(FragColor, 1.0); }"""
        self.shader = self.compile_shader_from_source(vertex_shader_source, fragment_shader_source)

        glClearColor(0.5, 0.8, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE)

        self.projection_matrix = create_projection_matrix(
            settings.get("graphics.fov", 75), self.width / self.height, 0.1, 1000.0)
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.projection_matrix)

    def toggle_wireframe(self): self.wireframe_mode = not self.wireframe_mode
    def compile_shader_from_source(self, vert_src, frag_src):
        try:
            return shaders.compileProgram(
                shaders.compileShader(vert_src, GL_VERTEX_SHADER),
                shaders.compileShader(frag_src, GL_FRAGMENT_SHADER))
        except Exception as e: raise RuntimeError(f"Shader compilation failed: {e}")
    def set_camera_orientation_provider(self, p): self.camera_orientation_provider = p
    def set_player_position_provider(self, p): self.player_position_provider = p

    def update(self, dt):
        player_pos = self.player_position_provider()
        pcx, pcz = self.world.get_chunk_coord(player_pos[0], player_pos[2])

        for x in range(pcx - config.RENDER_DISTANCE, pcx + config.RENDER_DISTANCE + 1):
            for z in range(pcz - config.RENDER_DISTANCE, pcz + config.RENDER_DISTANCE + 1):
                coord = (x, z)
                chunk = self.world.get_or_create_chunk(x, z)
                if coord not in self.chunk_meshes or (chunk and chunk.is_dirty):
                    self._build_chunk_mesh(coord, chunk)

        for coord in list(self.chunk_meshes.keys()):
            if abs(coord[0] - pcx) > config.RENDER_DISTANCE + 1 or abs(coord[1] - pcz) > config.RENDER_DISTANCE + 1:
                self.chunk_meshes.pop(coord).destroy()

    def _get_face_color(self, style, face_name, x, y, z):
        """Gets the base color for a face and applies any procedural patterns."""
        face_style = style.get(face_name, style)
        color = np.array(face_style.get("color", [1,0,1]), dtype=np.float32)
        pattern = face_style.get("pattern")

        if pattern == "random_noise":
            intensity = face_style.get("pattern_intensity", 0.1)
            noise = (random.random() - 0.5) * intensity
            return np.clip(color + noise, 0, 1)
        if pattern == "stripes":
            intensity = face_style.get("pattern_intensity", 0.1)
            if (x + y + z) % 2 == 0:
                return np.clip(color - intensity, 0, 1)
        if pattern == "bricks":
            intensity = 0.15
            # Simple procedural brick pattern based on world coordinates
            if (y % 4 == 0) or ((x + (y // 2)) % 6 == 0 and z % 2 == 0):
                 return np.clip(color - intensity, 0, 1) # Mortar color
        # FIX: Added the grass_top pattern to randomly select from three shades of green.
        if pattern == "grass_top":
            green1 = np.array([0.2, 0.8, 0.2])
            green2 = np.array([0.25, 0.75, 0.25])
            green3 = np.array([0.15, 0.85, 0.15])
            return random.choice([green1, green2, green3])
        return color

    def _build_chunk_mesh(self, coord, chunk):
        vertices = []
        for x, y, z in np.ndindex(chunk.blocks.shape):
            block_type = chunk.blocks[x, y, z]
            if block_type == core_world.BlockType.AIR: continue

            style = BLOCK_FACE_STYLES.get(block_type, BLOCK_FACE_STYLES["default"])

            # Top (+Y)
            if y == 255 or chunk.blocks[x, y + 1, z] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "top", x, y, z)
                v = [[x,y+1,z],[x+1,y+1,z],[x+1,y+1,z+1],[x,y+1,z+1]]
                vertices.extend([*v[0],*c, *v[2],*c, *v[1],*c, *v[0],*c, *v[3],*c, *v[2],*c])
            # Bottom (-Y)
            if y == 0 or chunk.blocks[x, y - 1, z] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "bottom", x, y, z)
                v = [[x,y,z],[x+1,y,z],[x+1,y,z+1],[x,y,z+1]]
                vertices.extend([*v[0],*c, *v[1],*c, *v[2],*c, *v[0],*c, *v[2],*c, *v[3],*c])
            # Right (+X)
            if x == 15 or chunk.blocks[x + 1, y, z] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "side", x, y, z)
                v = [[x+1,y,z],[x+1,y,z+1],[x+1,y+1,z+1],[x+1,y+1,z]]
                vertices.extend([*v[0],*c, *v[1],*c, *v[2],*c, *v[0],*c, *v[2],*c, *v[3],*c])
            # Left (-X)
            if x == 0 or chunk.blocks[x - 1, y, z] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "side", x, y, z)
                v = [[x,y,z],[x,y,z+1],[x,y+1,z+1],[x,y+1,z]]
                vertices.extend([*v[0],*c, *v[2],*c, *v[1],*c, *v[0],*c, *v[3],*c, *v[2],*c])
            # Front (+Z)
            if z == 15 or chunk.blocks[x, y, z + 1] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "side", x, y, z)
                v = [[x,y,z+1],[x+1,y,z+1],[x+1,y+1,z+1],[x,y+1,z+1]]
                vertices.extend([*v[0],*c, *v[1],*c, *v[2],*c, *v[0],*c, *v[2],*c, *v[3],*c])
            # Back (-Z)
            if z == 0 or chunk.blocks[x, y, z - 1] == core_world.BlockType.AIR:
                c = self._get_face_color(style, "side", x, y, z)
                v = [[x,y,z],[x+1,y,z],[x+1,y+1,z],[x,y+1,z]]
                vertices.extend([*v[0],*c, *v[2],*c, *v[1],*c, *v[0],*c, *v[3],*c, *v[2],*c])

        if coord in self.chunk_meshes: self.chunk_meshes[coord].destroy()
        self.chunk_meshes[coord] = ChunkMesh(vertices)
        chunk.is_dirty = False

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self.wireframe_mode else GL_FILL)
        glUseProgram(self.shader)

        pos, (yaw, pitch) = self.player_position_provider(), self.camera_orientation_provider()
        view_matrix = create_view_matrix(pos, yaw, pitch)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view_matrix)

        for coord, mesh in self.chunk_meshes.items():
            model_matrix = np.identity(4, dtype=np.float32)
            model_matrix[3, 0] = coord[0] * 16
            model_matrix[3, 2] = coord[1] * 16
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model_matrix)
            mesh.draw()

        if self.wireframe_mode: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def destroy(self):
        for mesh in self.chunk_meshes.values(): mesh.destroy()
        self.chunk_meshes.clear()
        if self.shader: glDeleteProgram(self.shader)
        print("Renderer destroyed.")
