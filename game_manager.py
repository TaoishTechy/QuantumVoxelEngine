# game_manager.py
# Main entry point for the game. Initializes all systems, runs the game loop,
# and orchestrates the interactions between different modules.

import sys
import os
import random
import pygame
import numpy as np
import concurrent.futures
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
import time
import queue
import ctypes
import threading
import logging
from typing import Dict, Tuple, List, Any, Optional
from collections import OrderedDict
from logging.handlers import QueueHandler, QueueListener

# --- Centralized Configuration ---
class Config:
    SCREEN_WIDTH: int = 1280
    SCREEN_HEIGHT: int = 720
    FOV: int = 70
    NEAR_PLANE: float = 0.1
    FAR_PLANE: float = 500.0
    SKY_COLOR: Tuple[float, float, float, float] = (0.5, 0.7, 1.0, 1.0)
    MESH_TIME_BUDGET_MS: float = 8.0
    MAX_MESH_UPLOADS_PER_FRAME: int = 5
    MAX_OUTSTANDING_MESH_TASKS: int = 8
    RENDER_DISTANCE: int = 5
    PLAYER_MOVE_SPEED: float = 50.0
    MOUSE_SENSITIVITY: float = 0.15
    TEXTURE_MAP: Dict[int, Tuple[int, int]] = {
        0: (0, 0), # Stone
        1: (1, 0), # Wood
    }
    DEFAULT_TEXTURE_COORDS: Tuple[int, int] = (3, 0)
    UI_THEME: Dict[str, Tuple[int, int, int]] = {
        "background": (20, 30, 50),
        "foreground": (230, 240, 255),
    }

# --- Logger Setup ---
def setup_logger() -> logging.Logger:
    log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(log_queue)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    listener = QueueListener(log_queue, handler)
    listener.start()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)
    return root_logger
logger = setup_logger()

# --- Accessibility Utilities ---
def get_luminance(rgb: Tuple[int, int, int]) -> float:
    srgb = [val / 255.0 for val in rgb]
    linear_rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * linear_rgb[0] + 0.7152 * linear_rgb[1] + 0.0722 * linear_rgb[2]

def get_contrast_ratio(lum1: float, lum2: float) -> float:
    lighter, darker = max(lum1, lum2), min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)

class ThemeManager:
    WCAG_AA_NORMAL_TEXT = 4.5
    def __init__(self, theme_palette: dict):
        self.palette = theme_palette
        self.validate_palette()

    def validate_palette(self) -> None:
        try:
            bg_lum = get_luminance(self.palette['background'])
            fg_lum = get_luminance(self.palette['foreground'])
            ratio = get_contrast_ratio(bg_lum, fg_lum)
            if ratio < self.WCAG_AA_NORMAL_TEXT:
                logger.warning(f"Theme contrast ratio is {ratio:.2f}, below WCAG AA standard of {self.WCAG_AA_NORMAL_TEXT}")
        except KeyError as e:
            logger.error(f"Theme palette is missing a required key: {e}")

# --- GPU Backend ---
class Shader:
    """A wrapper for an OpenGL shader program with robust error checking."""
    def __init__(self, vertex_path: str, fragment_path: str):
        self.program = self.compile_shader(vertex_path, fragment_path)

    def compile_shader(self, vertex_path: str, fragment_path: str) -> Optional[int]:
        """Loads and compiles a vertex and fragment shader."""
        try:
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()
            with open(fragment_path, 'r') as f:
                fragment_src = f.read()

            vertex_shader = shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
            program = shaders.compileProgram(vertex_shader, fragment_shader)
            return program
        except shaders.ShaderCompilationError as e:
            logger.critical(f"Shader compilation failed:\n{e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"An unexpected error occurred during shader compilation: {e}")
            sys.exit(1)

    def use(self):
        """Activates the shader program."""
        if self.program:
            glUseProgram(self.program)

    def get_uniform_location(self, name: str) -> int:
        """Gets the location of a uniform variable in the shader."""
        if self.program:
            return glGetUniformLocation(self.program, name)
        return -1

# --- Module Imports (Assumed to exist in the same directory) ---
import core_world
import procedural_generation

# --- Meshing Worker ---
def worker_build_mesh(chunk_coord: Tuple[int, int], world: 'core_world.WorldState') -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    wx, wz = chunk_coord
    chunk = world.get_or_create_chunk(wx, wz)
    vertices, indices, vertex_count = [], [], 0

    for x in range(core_world.Chunk.CHUNK_SIZE_X):
        for y in range(core_world.Chunk.CHUNK_HEIGHT):
            for z in range(core_world.Chunk.CHUNK_SIZE_Z):
                block_type = chunk.get_block(x, y, z)
                if block_type == core_world.BlockType.AIR: continue
                world_x, world_y, world_z = wx * 16 + x, y, wz * 16 + z
                tex_coords = Config.TEXTURE_MAP.get(block_type, Config.DEFAULT_TEXTURE_COORDS)

                # Top Face (+Y)
                if world.get_block(world_x, y + 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y + 1, world_z + 1), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y + 1, world_z), (world_x, y + 1, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Bottom Face (-Y)
                if world.get_block(world_x, y - 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z), (world_x + 1, y, world_z), (world_x + 1, y, world_z + 1), (world_x, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Right Face (+X)
                if world.get_block(world_x + 1, y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, world_z), (world_x + 1, y + 1, world_z), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Left Face (-X)
                if world.get_block(world_x - 1, y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z + 1), (world_x, y + 1, world_z + 1), (world_x, y + 1, world_z), (world_x, y, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Front Face (+Z)
                if world.get_block(world_x, y, world_z + 1).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z + 1), (world_x, y + 1, world_z + 1), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Back Face (-Z)
                if world.get_block(world_x, y, world_z - 1).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, world_z), (world_x + 1, y + 1, world_z), (world_x, y + 1, world_z), (world_x, y, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4

    return chunk_coord, np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_face_data(verts, tex_coords, start_index):
    uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]
    vertex_data = [(*verts[i], *uvs[i], *tex_coords) for i in range(4)]
    index_data = [start_index, start_index + 1, start_index + 2, start_index, start_index + 2, start_index + 3]
    return vertex_data, index_data

# --- Rendering and Input Classes ---
class ChunkVBO:
    def __init__(self):
        self.vao, self.vbo, self.ibo = glGenVertexArrays(1), glGenBuffers(1), glGenBuffers(1)
        self.index_count, self.is_uploaded = 0, False

    def upload_data(self, vertex_data, index_data):
        if vertex_data.size == 0: self.index_count = 0; return
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo); glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)
        stride = 28
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        self.index_count, self.is_uploaded = len(index_data), True

    def draw(self):
        if self.is_uploaded and self.index_count > 0:
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao]); glDeleteBuffers(2, [self.vbo, self.ibo])

class Renderer:
    def __init__(self, screen, world):
        self.screen, self.world = screen, world
        self.width, self.height = screen.get_size()
        self.chunk_vbos: Dict[Tuple[int, int], Any] = {}
        self.camera_pos = np.array([0.0, 100.0, 0.0])
        self.camera_yaw, self.camera_pitch = 0, 0
        self.shader = Shader("shaders/chunk.vert", "shaders/chunk.frag")
        self.texture_atlas = self.load_texture_atlas("atlas.png")

    def load_texture_atlas(self, path):
        try:
            image = pygame.image.load(path).convert_alpha()
            image_data = pygame.image.tostring(image, "RGBA", True)
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.get_width(), image.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
            return tex_id
        except Exception as e:
            logger.error(f"Could not load texture atlas '{path}': {e}")
            return None

    def draw_world(self):
        self.shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_atlas)
        proj = glGetFloatv(GL_PROJECTION_MATRIX); view = glGetFloatv(GL_MODELVIEW_MATRIX)
        glUniformMatrix4fv(self.shader.get_uniform_location("projection"), 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.shader.get_uniform_location("view"), 1, GL_FALSE, view)
        cam_chunk_x, cam_chunk_z = self.world.get_chunk_coord(self.camera_pos[0], self.camera_pos[2])
        for cx in range(cam_chunk_x - Config.RENDER_DISTANCE, cam_chunk_x + Config.RENDER_DISTANCE):
            for cz in range(cam_chunk_z - Config.RENDER_DISTANCE, cam_chunk_z + Config.RENDER_DISTANCE):
                if (cx, cz) in self.chunk_vbos and isinstance(self.chunk_vbos[(cx, cz)], ChunkVBO):
                    self.chunk_vbos[(cx, cz)].draw()

class InputHandler:
    def __init__(self, renderer):
        self.renderer = renderer
        pygame.mouse.set_visible(False); pygame.event.set_grab(True)

    def handle_input(self, dt):
        dx, dy = pygame.mouse.get_rel()
        self.renderer.camera_yaw += dx * Config.MOUSE_SENSITIVITY
        self.renderer.camera_pitch -= dy * Config.MOUSE_SENSITIVITY
        self.renderer.camera_pitch = np.clip(self.renderer.camera_pitch, -90, 90)

        keys = pygame.key.get_pressed()

        yaw_rad = np.radians(self.renderer.camera_yaw)
        pitch_rad = np.radians(self.renderer.camera_pitch)

        forward = np.array([np.cos(yaw_rad) * np.cos(pitch_rad), np.sin(pitch_rad), np.sin(yaw_rad) * np.cos(pitch_rad)])
        right = np.array([-np.sin(yaw_rad), 0, np.cos(yaw_rad)])

        move_vec = np.zeros(3)
        if keys[pygame.K_w]: move_vec -= right
        if keys[pygame.K_s]: move_vec += right
        if keys[pygame.K_a]: move_vec += forward
        if keys[pygame.K_d]: move_vec -= forward
        if keys[pygame.K_SPACE]: move_vec[1] += 1
        if keys[pygame.K_LSHIFT]: move_vec[1] -= 1

        if np.linalg.norm(move_vec) > 0:
            move_vec /= np.linalg.norm(move_vec)

        self.renderer.camera_pos += move_vec * Config.PLAYER_MOVE_SPEED * dt

    def handle_events(self, event):
        pass

class GameManager:
    def __init__(self):
        logger.info("--- LAUNCHING GOD TIER QUANTUM VOXEL ENGINE ---")
        pygame.init()
        self.width, self.height = Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Quantum Voxel Engine (Spectator Mode)")
        self.clock = pygame.time.Clock()
        self.running = True

        self.mesh_executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix='MeshWorker')
        self.mesh_priority_queue = queue.PriorityQueue()
        self.pending_uploads = []
        self.pending_uploads_lock = threading.Lock()
        self.active_mesh_tasks = set()
        self.active_mesh_tasks_lock = threading.Lock()

        self.world = self.setup_world()
        self.renderer = Renderer(self.screen, self.world)
        self.input_handler = InputHandler(self.renderer)
        self.theme_manager = ThemeManager(Config.UI_THEME)
        self.setup_opengl()
        logger.info("GameManager initialized successfully.")

    def setup_opengl(self):
        glClearColor(*[c/255.0 for c in Config.UI_THEME['background']], 1.0)
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE)

    def setup_world(self) -> core_world.WorldState:
        world = core_world.WorldState()
        world.generator = procedural_generation.QuantumProceduralGenerator(seed=1337)
        return world

    def update_chunks(self):
        cam_pos = self.renderer.camera_pos
        cam_chunk_x, cam_chunk_z = self.world.get_chunk_coord(cam_pos[0], cam_pos[2])
        for cx in range(cam_chunk_x - Config.RENDER_DISTANCE, cam_chunk_x + Config.RENDER_DISTANCE):
            for cz in range(cam_chunk_z - Config.RENDER_DISTANCE, cam_chunk_z + Config.RENDER_DISTANCE):
                chunk_coord = (cx, cz)
                if chunk_coord not in self.renderer.chunk_vbos:
                    dist_sq = (cx - cam_chunk_x)**2 + (cz - cam_chunk_z)**2
                    self.mesh_priority_queue.put((dist_sq, chunk_coord))
                    self.renderer.chunk_vbos[chunk_coord] = "pending"

    def process_mesh_tasks(self):
        with self.active_mesh_tasks_lock:
            can_start = Config.MAX_OUTSTANDING_MESH_TASKS - len(self.active_mesh_tasks)
            for _ in range(can_start):
                if self.mesh_priority_queue.empty(): break
                _, chunk_coord = self.mesh_priority_queue.get()
                if chunk_coord in self.active_mesh_tasks: continue
                self.active_mesh_tasks.add(chunk_coord)
                future = self.mesh_executor.submit(worker_build_mesh, chunk_coord, self.world)
                future.add_done_callback(lambda f, c=chunk_coord: self.on_mesh_completed(f, c))

    def on_mesh_completed(self, future, chunk_coord):
        try:
            _, vertex_data, index_data = future.result()
            with self.pending_uploads_lock:
                self.pending_uploads.append((chunk_coord, vertex_data, index_data))
        except Exception as e:
            logger.error(f"Mesh build for {chunk_coord} failed: {e}")
        finally:
            with self.active_mesh_tasks_lock:
                if chunk_coord in self.active_mesh_tasks:
                    self.active_mesh_tasks.remove(chunk_coord)

    def process_uploads(self):
        start_time = time.perf_counter()
        uploads_this_frame = 0
        with self.pending_uploads_lock:
            while self.pending_uploads and uploads_this_frame < Config.MAX_MESH_UPLOADS_PER_FRAME:
                if (time.perf_counter() - start_time) * 1000 > Config.MESH_TIME_BUDGET_MS:
                    logger.warning("Mesh upload budget exceeded.")
                    break
                chunk_coord, vertex_data, index_data = self.pending_uploads.pop(0)
                vbo = ChunkVBO()
                vbo.upload_data(vertex_data, index_data)
                self.renderer.chunk_vbos[chunk_coord] = vbo
                uploads_this_frame += 1

    def run(self):
        logger.info("Starting main game loop...")
        while self.running:
            try:
                dt = self.clock.tick(60) / 1000.0
                self.handle_events()
                self.update_chunks()
                self.process_mesh_tasks()
                self.process_uploads()
                self.input_handler.handle_input(dt)
                self.draw_scene()
            except Exception as e:
                logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
                self.running = False
        self.shutdown()

    def draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(Config.FOV, self.width / self.height, Config.NEAR_PLANE, Config.FAR_PLANE)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        cam = self.renderer
        glRotatef(cam.camera_pitch, 1, 0, 0)
        glRotatef(cam.camera_yaw, 0, 1, 0)
        glTranslatef(-cam.camera_pos[0], -cam.camera_pos[1], -cam.camera_pos[2])
        self.renderer.draw_world()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            self.input_handler.handle_events(event)

    def shutdown(self):
        logger.info("Shutting down...")
        self.mesh_executor.shutdown(wait=True)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameManager()
    game.run()
