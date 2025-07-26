# game_manager.py
# Main entry point for the game. Initializes all systems, runs the game loop,
# and orchestrates the interactions between different modules.

import sys
import os
import pygame
import concurrent.futures
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import queue
import threading
from typing import Dict, Any

from logger import logger
import config
from accessibility_utils import ThemeManager
import core_world
import procedural_generation
import simulation_core
from rendering import Renderer, InputHandler, ChunkVBO
from meshing import worker_build_mesh

class GameManager:
    def __init__(self):
        logger.info("--- LAUNCHING GOD TIER QUANTUM VOXEL ENGINE ---")
        pygame.init()
        self.width, self.height = config.SCREEN_WIDTH, config.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Quantum Voxel Engine (Refactored)")
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
        self.input_handler = InputHandler(self.world, self.renderer)
        self.theme_manager = ThemeManager(config.UI_THEME)
        self.setup_opengl()
        logger.info("GameManager initialized successfully.")

    def setup_opengl(self):
        glClearColor(*[c/255.0 for c in config.UI_THEME['background']], 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    def setup_world(self) -> core_world.WorldState:
        world = core_world.WorldState()
        world.generator = procedural_generation.QuantumProceduralGenerator(seed=1337)
        player = core_world.PhysicsObject(pos=[0, 80, 0], mass=60, size=(0.8, 1.8, 0.8))
        world.add_entity(player)
        return world

    def update_chunks(self):
        # ... (update_chunks remains the same)
        pass

    def process_mesh_tasks(self):
        with self.active_mesh_tasks_lock:
            can_start = config.MAX_OUTSTANDING_MESH_TASKS - len(self.active_mesh_tasks)
            for _ in range(can_start):
                if self.mesh_priority_queue.empty(): break
                _, chunk_coord = self.mesh_priority_queue.get()
                if chunk_coord in self.active_mesh_tasks: continue
                
                self.active_mesh_tasks.add(chunk_coord)
                future = self.mesh_executor.submit(worker_build_mesh, chunk_coord, self.world)
                future.add_done_callback(lambda f, c=chunk_coord: self.on_mesh_completed(f, c))

    def on_mesh_completed(self, future: concurrent.futures.Future, chunk_coord: tuple):
        try:
            _, vertex_data, index_data = future.result()
            with self.pending_uploads_lock:
                self.pending_uploads.append((chunk_coord, vertex_data, index_data))
        except Exception as e:
            logger.error(f"Mesh build for {chunk_coord} failed: {e}")
        finally:
            with self.active_mesh_tasks_lock:
                self.active_mesh_tasks.remove(chunk_coord)

    def process_uploads(self):
        start_time = time.perf_counter()
        uploads_this_frame = 0
        with self.pending_uploads_lock:
            while self.pending_uploads and uploads_this_frame < config.MAX_MESH_UPLOADS_PER_FRAME:
                if (time.perf_counter() - start_time) * 1000 > config.MESH_TIME_BUDGET_MS:
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
                self.world.step_simulation(dt)
                self.draw_scene()
            except Exception as e:
                logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
                self.running = False
        self.shutdown()

    def draw_scene(self):
        # ... (draw_scene remains the same)
        pass

    def handle_events(self):
        # ... (handle_events remains the same)
        pass

    def shutdown(self):
        logger.info("Shutting down...")
        self.mesh_executor.shutdown(wait=True)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameManager()
    game.run()
