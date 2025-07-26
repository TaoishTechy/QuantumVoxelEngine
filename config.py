# config.py
# Centralized configuration for all game constants and tunables.

from typing import Dict, Tuple

# Screen and Rendering
SCREEN_WIDTH: int = 1280
SCREEN_HEIGHT: int = 720
FOV: int = 70
NEAR_PLANE: float = 0.1
FAR_PLANE: float = 500.0
SKY_COLOR: Tuple[float, float, float, float] = (0.5, 0.7, 1.0, 1.0)

# Performance
MESH_TIME_BUDGET_MS: float = 4.0
MAX_MESH_UPLOADS_PER_FRAME: int = 2
MAX_OUTSTANDING_MESH_TASKS: int = 8
RENDER_DISTANCE: int = 4 # In chunks

# Physics
GRAVITY_MULTIPLIER: float = 1.0
TERMINAL_VELOCITY: float = -50.0

# Player
PLAYER_MOVE_SPEED: float = 5.0
PLAYER_JUMP_FORCE: float = 5.0
MOUSE_SENSITIVITY: float = 0.005

# Textures
TEXTURE_MAP: Dict[int, Tuple[int, int]] = {
    0: (0, 0), # Stone
    1: (1, 0), # Wood
    # 2: (2, 0), # Grass
    # 3: (3, 0), # Dirt
}
DEFAULT_TEXTURE_COORDS: Tuple[int, int] = (3, 0) # Dirt

# Accessibility & UI
UI_THEME: Dict[str, Tuple[int, int, int]] = {
    "background": (20, 30, 50),
    "foreground": (230, 240, 255),
    "accent": (60, 100, 180)
}
