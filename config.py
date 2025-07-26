# config.py
# Centralized, non-user-editable constants for game balance and engine configuration.
# For user-editable settings, use settings.json.

# --- Rendering ---
RENDER_DISTANCE = 8  # Chunk radius to load and render around the player

# --- Physics ---
GRAVITY = -9.81
TERMINAL_VELOCITY = -50.0

# --- World Generation ---
# These could be moved to settings.json if you want to make them user-configurable
BIOME_SCALE = 0.02
TERRAIN_BASE_HEIGHT = 64
HILL_AMPLITUDE = 20
MOUNTAIN_AMPLITUDE = 40
FRACTAL_OCTAVES = 6

# --- Quantum Mechanics ---
QUANTUM_ENTROPY_THRESHOLD = 5000  # Chunks with entropy below this are compressed on eviction
QUANTUM_WAVE_FUNCTION_SPREAD = 2.5 # Influences how far the player model predicts movement
QUANTUM_PERTURBATION_CHANCE = 0.05 # Chance per frame for a quantum object to jitter
QUANTUM_PERTURBATION_MAGNITUDE = 0.2 # How much a quantum object jitters

# --- World Systems ---
STORM_PROBABILITY_PER_TICK = 0.0001 # Lowered for more reasonable frequency
STORM_SUPERPOSITION_DURATION = 45.0 # Seconds the storm can remain in superposition
