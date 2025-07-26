# input_handler.py
# Manages user input for player movement and camera control.

import pygame
import numpy as np
from settings import settings # Import the global settings object
from core_world import PhysicsObject # Assuming the player is a PhysicsObject

class InputHandler:
    """
    Handles keyboard and mouse input to control the player/camera.
    Separates input logic from rendering and game logic.
    """
    def __init__(self, player: PhysicsObject):
        """
        Initializes the input handler.

        Args:
            player (PhysicsObject): The player entity to be controlled.
        """
        self.player = player

        # Load sensitivity and speed from the central settings object.
        self.mouse_sensitivity = settings.get("player.mouse_sensitivity", 0.1)
        # FIX: Increased movement speed as requested.
        self.move_speed = settings.get("player.move_speed", 25.0)

        # Camera orientation angles
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0

        # Grab the mouse and hide the cursor for a first-person experience
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def handle_input(self, dt: float):
        """
        Processes all continuous input (mouse movement and key presses) each frame.
        This should be called once per game loop update.

        Args:
            dt (float): The delta time since the last frame.
        """
        # --- Mouse Look ---
        dx, dy = pygame.mouse.get_rel()

        # Inverted the horizontal mouse input to correct the left/right turning.
        self.camera_yaw -= dx * self.mouse_sensitivity
        self.camera_pitch -= dy * self.mouse_sensitivity

        # Clamp the pitch to prevent the camera from flipping over
        self.camera_pitch = np.clip(self.camera_pitch, -89.9, 89.9)

        # --- Keyboard Movement ---
        keys = pygame.key.get_pressed()

        # Recalculated vectors for a standard right-handed, Y-up coordinate system
        # to ensure movement is always relative to the camera's direction.
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)

        cos_pitch = np.cos(pitch_rad)

        # The full 3D forward vector, pointing where the camera looks. Used for spectator/fly mode.
        forward = np.array([
            -np.sin(yaw_rad) * cos_pitch,
            np.sin(pitch_rad),
            -np.cos(yaw_rad) * cos_pitch
        ])

        # FIX: The right vector is derived from the cross product of the forward vector and the world up vector.
        # This order (forward x up) correctly yields a vector pointing to the right.
        right = np.cross(forward, np.array([0, 1, 0]))
        if np.linalg.norm(right) > 0:
            right = right / np.linalg.norm(right)

        move_vec = np.zeros(3, dtype=float)

        # Key bindings now use the corrected vectors for intuitive movement.
        # W/S for forward/backward movement along the camera's line of sight.
        if keys[pygame.K_w]:
            move_vec += forward
        if keys[pygame.K_s]:
            move_vec -= forward
        # A/D for strafing left/right.
        if keys[pygame.K_a]:
            move_vec -= right
        if keys[pygame.K_d]:
            move_vec += right

        # Vertical movement for flying/spectator mode is applied directly to the world Y-axis.
        if keys[pygame.K_SPACE]:
            move_vec[1] += 1
        if keys[pygame.K_LSHIFT]:
            move_vec[1] -= 1

        # Normalize the movement vector to ensure consistent speed in all directions.
        if move_vec.any(): # Check if any element is non-zero
            norm = np.linalg.norm(move_vec)
            if norm > 0:
                move_vec /= norm

        # Apply movement to the player's position.
        self.player.pos += move_vec * self.move_speed * dt

    def handle_events(self, event: pygame.event.Event):
        """
        Handles discrete events like key presses (e.g., for toggles or single actions).
        This should be called in the event loop.

        Args:
            event (pygame.event.Event): The event to process.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Toggle mouse grab and visibility
                current_grab = pygame.event.get_grab()
                pygame.event.set_grab(not current_grab)
                pygame.mouse.set_visible(current_grab)

    def get_camera_orientation(self) -> tuple[float, float]:
        """Returns the current camera orientation angles."""
        return self.camera_yaw, self.camera_pitch
