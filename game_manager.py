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
from OpenGL.GLU import *
import time
import queue
import ctypes

# Add the script's directory to the Python path to ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core and advanced modules
import core_world
import procedural_generation
import simulation_core
import visualization
import gpu_backend

# --- Constants ---
MESH_TIME_BUDGET_MS = 8.0 # Increased budget for smoother mesh loading
MAX_MESH_UPLOADS_PER_FRAME = 4 # Allow more uploads per frame
MAX_OUTSTANDING_MESH_TASKS = 8 # Limit concurrent mesh builds

# --- Thread-Safe Worker Function for VBO Mesh Building ---
def worker_build_mesh(chunk_coord, world, texture_map):
    """
    (THREAD-SAFE) Constructs the mesh for a single chunk.
    This function is designed to be run in a separate thread.
    """
    wx, wz = chunk_coord
    chunk = world.get_or_create_chunk(wx, wz)

    vertices = []
    indices = []
    vertex_count = 0

    for x in range(chunk.CHUNK_SIZE_X):
        for y in range(chunk.CHUNK_HEIGHT):
            for z in range(chunk.CHUNK_SIZE_Z):
                block_type = chunk.get_block(x, y, z)
                if block_type == core_world.BlockType.AIR: continue

                world_x, world_y, world_z = wx * chunk.CHUNK_SIZE_X + x, y, wz * chunk.CHUNK_SIZE_Z + z

                tex_coords = texture_map.get(block_type, (3,0)) # Default to dirt

                # --- FIX: Use world coordinates for all vertices and correct winding order ---
                # Top Face (+Y)
                if world.get_block(world_x, world_y + 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y + 1, world_z), (world_x + 1, y + 1, world_z), (world_x + 1, y + 1, world_z + 1), (world_x, y + 1, world_z + 1)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # Bottom Face (-Y)
                if world.get_block(world_x, world_y - 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z), (world_x + 1, y, world_z), (world_x + 1, y, world_z + 1), (world_x, y, world_z + 1)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # Right Face (+X)
                if world.get_block(world_x + 1, world_y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, z), (world_x + 1, y + 1, z), (world_x + 1, y + 1, z + 1), (world_x + 1, y, z + 1)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # Left Face (-X)
                if world.get_block(world_x - 1, world_y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, z + 1), (world_x, y + 1, z + 1), (world_x, y + 1, z), (world_x, y, z)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # Front Face (+Z)
                if world.get_block(world_x, world_y, world_z + 1).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, z + 1), (world_x + 1, y + 1, z + 1), (world_x, y + 1, z + 1), (world_x, y, z + 1)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # Back Face (-Z)
                if world.get_block(world_x, world_y, world_z - 1).type == core_world.BlockType.AIR:
                    v = [(world_x, y, z), (world_x, y + 1, z), (world_x + 1, y + 1, z), (world_x + 1, y, z)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4

    return chunk_coord, np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_face_data(verts, tex_coords, start_index):
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # Vertex format: [x, y, z, u, v, atlas_x, atlas_y]
    vertex_data = [(*verts[i], *uvs[i], *tex_coords) for i in range(4)]
    # Indices for two triangles forming a quad
    index_data = [start_index, start_index + 1, start_index + 2, start_index + 2, start_index + 3, start_index]
    return vertex_data, index_data

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

class Renderer:
    def __init__(self, screen, world, game_manager):
        self.screen, self.world, self.game_manager = screen, world, game_manager
        self.width, self.height = screen.get_size()
        self.chunk_vbos = {}
        self.camera_pos = np.array([0.0, 85.0, 0.0])
        self.camera_yaw, self.camera_pitch = -np.pi / 2, 0
        self.fov, self.near_plane, self.far_plane = 70, 0.1, 500.0
        self.sky_color = (0.5, 0.7, 1.0, 1.0)
        self.shader = gpu_backend.Shader("shaders/chunk.vert", "shaders/chunk.frag")
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
            print(f"ERROR: Could not load texture atlas '{path}': {e}")
            return None

    def draw_world(self):
        self.shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_atlas)

        proj = glGetFloatv(GL_PROJECTION_MATRIX)
        view = glGetFloatv(GL_MODELVIEW_MATRIX)
        glUniformMatrix4fv(self.shader.get_uniform_location("projection"), 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.shader.get_uniform_location("view"), 1, GL_FALSE, view)

        cam_chunk_x, cam_chunk_z = self.world.get_chunk_coord(self.camera_pos[0], self.camera_pos[2])
        render_distance = 4

        for cx in range(cam_chunk_x - render_distance, cam_chunk_x + render_distance):
            for cz in range(cam_chunk_z - render_distance, cam_chunk_z + render_distance):
                chunk_coord = (cx, cz)
                if chunk_coord in self.chunk_vbos and isinstance(self.chunk_vbos[chunk_coord], ChunkVBO):
                    self.chunk_vbos[chunk_coord].draw()

class InputHandler:
    def __init__(self, world, renderer):
        self.world, self.renderer = world, renderer
        self.player = world.entities[0] if world.entities else None
        self.move_speed, self.jump_force, self.mouse_sensitivity = 5.0, 5.0, 0.005
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def handle_input(self, dt):
        if not self.player: return
        dx, dy = pygame.mouse.get_rel()
        self.renderer.camera_yaw += dx * self.mouse_sensitivity
        self.renderer.camera_pitch -= dy * self.mouse_sensitivity
        self.renderer.camera_pitch = np.clip(self.renderer.camera_pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)

        keys = pygame.key.get_pressed()
        forward = np.array([np.cos(self.renderer.camera_yaw), 0, np.sin(self.renderer.camera_yaw)])
        right = np.array([-np.sin(self.renderer.camera_yaw), 0, np.cos(self.renderer.camera_yaw)])

        move_vec = np.zeros(3)
        if keys[pygame.K_w]: move_vec += forward
        if keys[pygame.K_s]: move_vec -= forward
        if keys[pygame.K_a]: move_vec -= right
        if keys[pygame.K_d]: move_vec += right

        if np.linalg.norm(move_vec) > 0: move_vec /= np.linalg.norm(move_vec)

        self.player.vel[0], self.player.vel[2] = move_vec[0] * self.move_speed, move_vec[2] * self.move_speed
        self.renderer.camera_pos = self.player.pos + np.array([0, 0.8, 0])

    def handle_events(self, event):
        if not self.player: return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and self.player.on_ground:
            self.player.vel[1] = self.jump_force

class GameManager:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1280, 720
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("God Tier Quantum Voxel Game (VBO Optimized)")
        self.clock = pygame.time.Clock()
        self.running = True

        self.mesh_executor = concurrent.futures.ThreadPoolExecutor()
        self.mesh_priority_queue = queue.PriorityQueue()
        self.pending_uploads = []

        self.world = self.setup_world()
        self.renderer = Renderer(self.screen, self.world, self)
        self.input_handler = InputHandler(self.world, self.renderer)
        self.setup_opengl()

    def setup_opengl(self):
        glClearColor(*self.renderer.sky_color)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    def setup_world(self):
        world = core_world.WorldState()
        world.generator = procedural_generation.QuantumProceduralGenerator(seed=1337)

        # Pre-generate spawn chunks
        for x in range(-2, 3):
            for z in range(-2, 3):
                world.get_or_create_chunk(x, z)

        # Find a safe spawn height
        ground_height = 0
        for y in range(core_world.Chunk.CHUNK_HEIGHT - 1, -1, -1):
            if world.get_block(0, y, 0).type != core_world.BlockType.AIR:
                ground_height = y
                break

        spawn_pos = [0, ground_height + 5.0, 0] # Start higher up
        player = core_world.PhysicsObject(pos=spawn_pos, mass=60, size=(0.8, 1.8, 0.8))
        world.add_entity(player)
        return world

    def update_chunks(self):
        cam_pos = self.renderer.camera_pos
        cam_chunk_x, cam_chunk_z = self.world.get_chunk_coord(cam_pos[0], cam_pos[2])
        render_distance = 4

        for cx in range(cam_chunk_x - render_distance, cam_chunk_x + render_distance):
            for cz in range(cam_chunk_z - render_distance, cam_chunk_z + render_distance):
                chunk_coord = (cx, cz)
                if chunk_coord not in self.renderer.chunk_vbos:
                    dist_sq = (cx - cam_chunk_x)**2 + (cz - cam_chunk_z)**2
                    self.mesh_priority_queue.put((dist_sq, chunk_coord))
                    self.renderer.chunk_vbos[chunk_coord] = "pending"

    def process_mesh_tasks(self):
        if len(self.mesh_executor._threads) < MAX_OUTSTANDING_MESH_TASKS and not self.mesh_priority_queue.empty():
            _, chunk_coord = self.mesh_priority_queue.get()
            texture_map = {
                core_world.BlockType.STONE: (0,0),
                core_world.BlockType.WOOD: (1,0),
            }
            future = self.mesh_executor.submit(worker_build_mesh, chunk_coord, self.world, texture_map)
            future.add_done_callback(self.on_mesh_completed)

    def on_mesh_completed(self, future):
        try:
            chunk_coord, vertex_data, index_data = future.result()
            self.pending_uploads.append((chunk_coord, vertex_data, index_data))
        except Exception as e:
            print(f"ERROR: Mesh build failed: {e}")

    def process_uploads(self):
        start_time = time.perf_counter()
        uploads_this_frame = 0
        while self.pending_uploads and uploads_this_frame < MAX_MESH_UPLOADS_PER_FRAME:
            if (time.perf_counter() - start_time) * 1000 > MESH_TIME_BUDGET_MS:
                break

            chunk_coord, vertex_data, index_data = self.pending_uploads.pop(0)
            vbo = ChunkVBO()
            vbo.upload_data(vertex_data, index_data)
            self.renderer.chunk_vbos[chunk_coord] = vbo
            uploads_this_frame += 1

    def run(self):
        frame_count = 0
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            frame_count += 1
            self.handle_events()
            self.update_chunks()
            self.process_mesh_tasks()
            self.process_uploads()
            self.input_handler.handle_input(dt)
            self.world.step_simulation(dt)
            self.draw_scene()

            if frame_count % 120 == 0:
                player = self.input_handler.player
                print(f"FPS: {self.clock.get_fps():.1f} | Pos: ({player.pos[0]:.1f}, {player.pos[1]:.1f}, {player.pos[2]:.1f}) | "
                      f"Chunks: {len(self.renderer.chunk_vbos)} | Pending: {len(self.pending_uploads)}")

    def draw_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.renderer.fov, self.width / self.height, self.renderer.near_plane, self.renderer.far_plane)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam = self.renderer
        glRotatef(np.degrees(cam.camera_pitch), 1, 0, 0)
        glRotatef(np.degrees(cam.camera_yaw), 0, 1, 0)
        glTranslatef(-cam.camera_pos[0], -cam.camera_pos[1], -cam.camera_pos[2])

        self.renderer.draw_world()
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): self.running = False
            self.input_handler.handle_events(event)

    def shutdown(self):
        self.mesh_executor.shutdown(wait=True)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameManager()
    game.run()
