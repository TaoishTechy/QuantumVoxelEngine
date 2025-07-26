# renderer.py
# The main rendering engine for the Quantum Voxel Engine.
# Handles all OpenGL calls, shader management, and mesh generation/rendering for the world.

import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import ctypes

# --- Engine Imports ---
import config
from settings import settings
import core_world
import gpu_backend

# --- Helper Functions for 3D Math ---

def create_projection_matrix(fov, aspect_ratio, near_plane, far_plane):
    """Creates a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
    matrix[2, 3] = -1.0
    matrix[3, 2] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
    return matrix

def create_view_matrix(position, yaw, pitch):
    """Creates a view matrix for a first-person camera."""
    # Rotation
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    xaxis = np.array([cos_yaw, 0, -sin_yaw])
    yaxis = np.array([sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch])
    zaxis = np.array([sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw])

    rotation_matrix = np.array([
        [xaxis[0], yaxis[0], zaxis[0], 0],
        [xaxis[1], yaxis[1], zaxis[1], 0],
        [xaxis[2], yaxis[2], zaxis[2], 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # Translation
    translation_matrix = np.identity(4, dtype=np.float32)
    translation_matrix[3, 0] = -position[0]
    translation_matrix[3, 1] = -position[1]
    translation_matrix[3, 2] = -position[2]

    return translation_matrix @ rotation_matrix

class ChunkMesh:
    """A class to hold the VAO and VBO for a single chunk's mesh."""
    def __init__(self, vertices):
        self.vertex_count = int(len(vertices) / 6) # 6 floats per vertex
        if self.vertex_count == 0:
            self.vao = None
            self.vbo = None
            return

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

        # Vertex format: x, y, z, r, g, b (6 floats -> 24 bytes)
        # Position (Attribute 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Color (Attribute 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def draw(self):
        if self.vao:
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self):
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        self.vao, self.vbo = None, None

class Renderer:
    """
    The main rendering engine.
    """
    def __init__(self, screen, world):
        self.screen = screen
        self.world = world
        self.width, self.height = screen.get_size()

        self.camera_orientation_provider = lambda: (0, 0)
        self.player_position_provider = lambda: [0, 0, 0]

        self.chunk_meshes = {}
        self.wireframe_mode = False

        # --- Shaders ---
        vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;

        out vec3 FragColor;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            FragColor = aColor;
        }
        """
        fragment_shader_source = """
        #version 330 core
        in vec3 FragColor;
        out vec4 FinalColor;

        void main()
        {
            FinalColor = vec4(FragColor, 1.0);
        }
        """
        self.shader = self.compile_shader_from_source(vertex_shader_source, fragment_shader_source)

        # --- OpenGL Setup ---
        glClearColor(0.5, 0.8, 1.0, 1.0) # Sky blue
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # --- Matrices ---
        self.projection_matrix = create_projection_matrix(
            settings.get("graphics.fov", 75),
            self.width / self.height,
            0.1, 1000.0
        )
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, self.projection_matrix)

    def toggle_wireframe(self):
        """Toggles wireframe rendering mode."""
        self.wireframe_mode = not self.wireframe_mode

    def compile_shader_from_source(self, vertex_src, fragment_src):
        """Compiles a shader program from source strings."""
        try:
            vert = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            frag = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            program = shaders.compileProgram(vert, frag)
            return program
        except Exception as e:
            raise RuntimeError(f"Shader compilation from source failed: {e}")

    def set_camera_orientation_provider(self, provider):
        self.camera_orientation_provider = provider

    def set_player_position_provider(self, provider):
        self.player_position_provider = provider

    def update(self, dt):
        """Handles loading and unloading chunk meshes based on player position."""
        player_pos = self.player_position_provider()
        player_chunk_coord = self.world.get_chunk_coord(player_pos[0], player_pos[2])

        render_dist = config.RENDER_DISTANCE
        for x in range(player_chunk_coord[0] - render_dist, player_chunk_coord[0] + render_dist + 1):
            for z in range(player_chunk_coord[1] - render_dist, player_chunk_coord[1] + render_dist + 1):
                coord = (x, z)
                chunk = self.world.get_or_create_chunk(x, z)
                if coord not in self.chunk_meshes and chunk:
                    self._build_chunk_mesh(coord, chunk)
                elif chunk and chunk.is_dirty:
                    self._build_chunk_mesh(coord, chunk)

        loaded_coords = list(self.chunk_meshes.keys())
        for coord in loaded_coords:
            dist_x = abs(coord[0] - player_chunk_coord[0])
            dist_z = abs(coord[1] - player_chunk_coord[1])
            if dist_x > render_dist + 1 or dist_z > render_dist + 1:
                if coord in self.chunk_meshes:
                    self.chunk_meshes[coord].destroy()
                    del self.chunk_meshes[coord]

    def _build_chunk_mesh(self, coord, chunk):
        """Generates a mesh for a single chunk using a simple face-culling approach."""
        vertices = []

        colors = {
            core_world.BlockType.GRASS: [0.2, 0.8, 0.2],
            core_world.BlockType.DIRT: [0.6, 0.4, 0.2],
            core_world.BlockType.STONE: [0.5, 0.5, 0.5],
            core_world.BlockType.WOOD: [0.7, 0.5, 0.3],
            core_world.BlockType.LAVA: [1.0, 0.5, 0.0],
            core_world.BlockType.WATER: [0.2, 0.5, 1.0],
            core_world.BlockType.OBSIDIAN: [0.1, 0.1, 0.2],
            core_world.BlockType.SAND: [0.9, 0.9, 0.7],
        }

        for x_local in range(chunk.CHUNK_SIZE_X):
            for y_local in range(chunk.CHUNK_HEIGHT):
                for z_local in range(chunk.CHUNK_SIZE_Z):
                    block_type = chunk.blocks[x_local, y_local, z_local]
                    if block_type == core_world.BlockType.AIR:
                        continue

                    color = colors.get(block_type, [1, 0, 1])

                    # Check each face for visibility
                    # Top (+Y)
                    if y_local == chunk.CHUNK_HEIGHT - 1 or chunk.blocks[x_local, y_local + 1, z_local] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local, y_local + 1, z_local], [x_local + 1, y_local + 1, z_local], [x_local + 1, y_local + 1, z_local + 1], [x_local, y_local + 1, z_local + 1]
                        vertices.extend([*v1, *color, *v3, *color, *v2, *color])
                        vertices.extend([*v1, *color, *v4, *color, *v3, *color])
                    # Bottom (-Y)
                    if y_local == 0 or chunk.blocks[x_local, y_local - 1, z_local] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local, y_local, z_local], [x_local + 1, y_local, z_local], [x_local + 1, y_local, z_local + 1], [x_local, y_local, z_local + 1]
                        vertices.extend([*v1, *color, *v2, *color, *v3, *color])
                        vertices.extend([*v1, *color, *v3, *color, *v4, *color])
                    # Right (+X)
                    if x_local == chunk.CHUNK_SIZE_X - 1 or chunk.blocks[x_local + 1, y_local, z_local] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local + 1, y_local, z_local], [x_local + 1, y_local, z_local + 1], [x_local + 1, y_local + 1, z_local + 1], [x_local + 1, y_local + 1, z_local]
                        # FIX: Reversed winding order to match other faces.
                        vertices.extend([*v1, *color, *v3, *color, *v2, *color])
                        vertices.extend([*v1, *color, *v4, *color, *v3, *color])
                    # Left (-X)
                    if x_local == 0 or chunk.blocks[x_local - 1, y_local, z_local] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local, y_local, z_local], [x_local, y_local, z_local + 1], [x_local, y_local + 1, z_local + 1], [x_local, y_local + 1, z_local]
                        vertices.extend([*v1, *color, *v2, *color, *v3, *color])
                        vertices.extend([*v1, *color, *v3, *color, *v4, *color])
                    # Front (+Z)
                    if z_local == chunk.CHUNK_SIZE_Z - 1 or chunk.blocks[x_local, y_local, z_local + 1] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local, y_local, z_local + 1], [x_local + 1, y_local, z_local + 1], [x_local + 1, y_local + 1, z_local + 1], [x_local, y_local + 1, z_local + 1]
                        # FIX: Reversed winding order to match other faces.
                        vertices.extend([*v1, *color, *v3, *color, *v2, *color])
                        vertices.extend([*v1, *color, *v4, *color, *v3, *color])
                    # Back (-Z)
                    if z_local == 0 or chunk.blocks[x_local, y_local, z_local - 1] == core_world.BlockType.AIR:
                        v1, v2, v3, v4 = [x_local, y_local, z_local], [x_local + 1, y_local, z_local], [x_local + 1, y_local + 1, z_local], [x_local, y_local + 1, z_local]
                        vertices.extend([*v1, *color, *v2, *color, *v3, *color])
                        vertices.extend([*v1, *color, *v3, *color, *v4, *color])

        if coord in self.chunk_meshes:
            self.chunk_meshes[coord].destroy()
        self.chunk_meshes[coord] = ChunkMesh(vertices)
        chunk.is_dirty = False

    def draw(self):
        """Clears the screen and draws the world."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(self.shader)

        pos = self.player_position_provider()
        yaw, pitch = self.camera_orientation_provider()
        view_matrix = create_view_matrix(pos, yaw, pitch)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view_matrix)

        for coord, mesh in self.chunk_meshes.items():
            model_matrix = np.identity(4, dtype=np.float32)
            model_matrix[3, 0] = coord[0] * core_world.Chunk.CHUNK_SIZE_X
            model_matrix[3, 2] = coord[1] * core_world.Chunk.CHUNK_SIZE_Z
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model_matrix)
            mesh.draw()

        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def destroy(self):
        """Cleans up all GPU resources."""
        for mesh in self.chunk_meshes.values():
            mesh.destroy()
        self.chunk_meshes.clear()
        if self.shader:
            glDeleteProgram(self.shader)
        print("Renderer destroyed.")
