# game_manager.py
# The central orchestrator for the Quantum Voxel Engine.
# Initializes all subsystems, runs the main game loop, and handles shutdown.

import pygame
import sys
import numpy as np

# --- Core Engine Imports ---
from logger import logger
from settings import settings
import config

# --- Subsystem Imports ---
import core_world
import procedural_generation
from input_handler import InputHandler
import renderer

class GameManager:
    """
    The main class that initializes and runs all game components.
    """
    def __init__(self):
        """
        Initializes Pygame, loads settings, and sets up all engine subsystems.
        """
        logger.info("--- LAUNCHING GOD TIER QUANTUM VOXEL ENGINE ---")

        # Initialize Pygame
        pygame.init()

        # Load settings and configure the display
        self.screen_width = settings.get("screen.width", 1280)
        self.screen_height = settings.get("screen.height", 720)
        self.max_fps = settings.get("graphics.max_fps", 144)

        # Set up the OpenGL display
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("Quantum Voxel Engine")

        # Initialize core game systems
        self.world = core_world.WorldState()
        self.world.generator = procedural_generation.QuantumProceduralGenerator(seed=42)

        # Create the player entity and add it to the world
        self.player = core_world.PhysicsObject(pos=[8, 80, 8], size=[0.8, 1.8, 0.8])
        self.world.add_entity(self.player)

        # Initialize the InputHandler with the player object.
        self.input_handler = InputHandler(self.player)

        # Initialize the rendering engine
        self.renderer = renderer.Renderer(self.screen, self.world)
        self.renderer.set_camera_orientation_provider(self.input_handler.get_camera_orientation)
        self.renderer.set_player_position_provider(lambda: self.player.pos)

        # Initialize game clock
        self.clock = pygame.time.Clock()
        self.running = False

        # Add spectator mode, enabled by default to prevent falling on load.
        self.spectator_mode = True
        logger.info("Spectator mode enabled by default. Press 'V' to toggle physics.")

    def run(self):
        """
        Starts and runs the main game loop.
        """
        self.running = True
        while self.running:
            # Calculate delta time for frame-independent physics and movement
            dt = self.clock.tick(self.max_fps) / 1000.0

            # Process events, update game state, and render the scene
            self.handle_events()
            self.update(dt)
            self.draw()

        # Once the loop exits, shut down the engine
        self.shutdown()

    def handle_events(self):
        """
        Processes all pending Pygame events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                # Add toggles for spectator mode and wireframe rendering.
                if event.key == pygame.K_v:
                    self.spectator_mode = not self.spectator_mode
                    mode_text = "SPECTATOR" if self.spectator_mode else "PHYSICS"
                    logger.info(f"Switched to {mode_text} mode.")
                    # When switching to physics, reset velocity to prevent carrying over momentum.
                    if not self.spectator_mode:
                        self.player.vel = np.zeros(3, dtype=float)

                if event.key == pygame.K_r:
                    self.renderer.toggle_wireframe()
                    logger.info(f"Wireframe mode {'enabled' if self.renderer.wireframe_mode else 'disabled'}.")

            # Pass events to the input handler for discrete actions (e.g., ESC key)
            self.input_handler.handle_events(event)

    def update(self, dt: float):
        """
        Updates all game logic for the current frame.
        """
        # Handle continuous input (keyboard and mouse movement)
        self.input_handler.handle_input(dt)

        # Only step the physics simulation if not in spectator mode.
        if not self.spectator_mode:
            self.world.step_simulation(dt)

        # Update the renderer (e.g., for world streaming)
        self.renderer.update(dt)

        # Update the window title with FPS and player position
        fps = self.clock.get_fps()
        pos = self.player.pos
        mode = "Spectator" if self.spectator_mode else "Physics"
        pygame.display.set_caption(
            f"QVE | FPS: {fps:.1f} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | Mode: {mode} [V]"
        )

    def draw(self):
        """
        Renders the entire scene to the screen.
        """
        self.renderer.draw()
        pygame.display.flip()

    def shutdown(self):
        """
        Cleans up all resources and exits the game.
        """
        logger.info("--- SHUTTING DOWN ENGINE ---")
        self.renderer.destroy()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    try:
        game = GameManager()
        game.run()
    except Exception as e:
        # Use the centralized logger to report the crash
        logger.critical("Engine crashed with a fatal error:", exc_info=True)
        # Still print to stderr for visibility if logging fails
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)
