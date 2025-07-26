# core_world.py
# Contains: WorldState, Chunk, Block, PhysicsObject, QuantumObject, PhysicsEngine
# This script is responsible for all physics, world state, and core quantum logic.
# It is designed to be completely independent of rendering and user input for CPU optimization.

import numpy as np
import random
from collections import defaultdict
import config # Import centralized constants

# --- Constants and Enumerations ---
class BlockType:
    """Defines the types of blocks available in the world. Names are now descriptive."""
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WOOD = 4
    WATER = 5
    QUANTUM_ORE = 6

class Material:
    """Defines physical properties for materials."""
    STONE = 0
    WOOD = 1
    METAL = 2
    QUANTUM = 3

# --- Core Data Structures ---

class Block:
    """
    Represents a single voxel in the world with state.
    NOTE: This class should only be instantiated when block-specific state (like health)
    is required. For simple block type checking, use the integer IDs from BlockType.
    """
    def __init__(self, block_type=BlockType.AIR):
        self.type = block_type
        self.health = 100
        # Quantum properties can be added here
        self.quantum_state = 0.0

class Chunk:
    """A 16x256x16 segment of the world."""
    CHUNK_SIZE_X = 16
    CHUNK_SIZE_Z = 16
    CHUNK_HEIGHT = 256

    def __init__(self):
        # Initialize a 3D NumPy array for blocks, storing only integer IDs for efficiency.
        self.blocks = np.full(
            (self.CHUNK_SIZE_X, self.CHUNK_HEIGHT, self.CHUNK_SIZE_Z),
            fill_value=BlockType.AIR,
            dtype=np.uint8 # Use a memory-efficient integer type
        )
        self.is_dirty = True # Flag for mesh regeneration

    def get_block(self, x: int, y: int, z: int) -> int:
        """Gets a block ID at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            return self.blocks[x, y, z]
        return BlockType.AIR

    def set_block(self, x: int, y: int, z: int, block_type: int) -> bool:
        """Sets a block at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            if self.blocks[x, y, z] != block_type:
                self.blocks[x, y, z] = block_type
                self.is_dirty = True
            return True
        return False

    def calculate_entropy(self) -> float:
        """
        Calculates the 'entropy' of a chunk based on the variety of block types.
        A simple measure is the number of unique block types present.
        """
        unique_blocks, counts = np.unique(self.blocks, return_counts=True)
        # Simple entropy: more types of blocks or more changes = higher entropy
        # A more advanced version could use Shannon entropy: -sum(p*log(p))
        return len(unique_blocks)

class PhysicsObject:
    """Represents a dynamic entity in the world (e.g., player, item, particle)."""
    def __init__(self, pos, mass=1.0, size=(1, 1, 1)):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0
        self.size = np.array(size, dtype=float)
        self.on_ground = False

class QuantumObject(PhysicsObject):
    """An entity with quantum properties."""
    def __init__(self, pos, mass=0.1, size=(0.5, 0.5, 0.5)):
        super().__init__(pos, mass, size)
        self.is_quantum = True
        self.is_observed = False
        self.probability_cloud_radius = 5.0 # Example property
        self.ghost_positions = []

# --- Main World and Physics Engine ---

class WorldState:
    """Manages all chunks, entities, and the overall state of the world."""
    def __init__(self):
        self.chunks = {}  # Dictionary to store chunks by their (x, z) coordinate
        self.entities = [] # Global list of all entities
        self.physics_engine = PhysicsEngine(self)
        self.generator = None # Will be set by GameManager

    def get_chunk_coord(self, world_x: float, world_z: float) -> tuple[int, int]:
        """Converts world coordinates to chunk coordinates."""
        return (int(world_x // Chunk.CHUNK_SIZE_X), int(world_z // Chunk.CHUNK_SIZE_Z))

    def get_or_create_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Retrieves a chunk, creating it if it doesn't exist."""
        coord = (chunk_x, chunk_z)
        if coord not in self.chunks:
            if self.generator:
                self.chunks[coord] = self.generate_chunk_from_data(chunk_x, chunk_z)
            else:
                # Fallback if no generator is attached
                self.chunks[coord] = Chunk()
        return self.chunks[coord]

    def generate_chunk_from_data(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Creates a chunk object from procedurally generated data."""
        chunk = Chunk()
        chunk_data = self.generator.generate_chunk(chunk_x, chunk_z)
        chunk.blocks = chunk_data
        return chunk

    def get_block_type(self, world_x: float, world_y: float, world_z: float) -> int:
        """
        Gets a block's type ID at absolute world coordinates.
        FIX: This is the high-performance version that returns an integer, not a Block object.
        """
        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        # No need to create a chunk just to check a block, assume AIR if it doesn't exist.
        if (chunk_x, chunk_z) in self.chunks:
            chunk = self.chunks[(chunk_x, chunk_z)]
            local_x = int(world_x % Chunk.CHUNK_SIZE_X)
            local_y = int(world_y)
            local_z = int(world_z % Chunk.CHUNK_SIZE_Z)
            return chunk.get_block(local_x, local_y, local_z)
        return BlockType.AIR

    def set_block(self, world_x: float, world_y: float, world_z: float, block_type: int) -> bool:
        """Sets a block at absolute world coordinates."""
        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        if chunk:
            local_x = int(world_x % Chunk.CHUNK_SIZE_X)
            local_y = int(world_y)
            local_z = int(world_z % Chunk.CHUNK_SIZE_Z)
            return chunk.set_block(local_x, local_y, local_z, block_type)
        return False

    def add_entity(self, entity: PhysicsObject):
        """Adds a new entity to the world."""
        self.entities.append(entity)

    def step_simulation(self, dt: float):
        """Advances the entire world state by one time step."""
        self.physics_engine.update(dt)

class PhysicsEngine:
    """Handles all physics calculations for the world."""
    # FIX: Physics constants are now sourced from the config file.
    GRAVITY = np.array([0, config.GRAVITY, 0])
    TERMINAL_VELOCITY = config.TERMINAL_VELOCITY

    def __init__(self, world_state: WorldState):
        self.world = world_state

    def update(self, dt: float):
        """Updates all entities."""
        for entity in self.world.entities:
            self.apply_gravity(entity)
            self.update_entity_position(entity, dt)
            self.handle_collisions(entity)
            if isinstance(entity, QuantumObject) and not entity.is_observed:
                self.apply_quantum_effects(entity, dt)

    def apply_gravity(self, entity: PhysicsObject):
        """Applies gravity to a single entity."""
        if not entity.on_ground:
            entity.force += self.GRAVITY * entity.mass

    def update_entity_position(self, entity: PhysicsObject, dt: float):
        """Updates position and velocity based on forces."""
        if entity.inv_mass == 0: return
        acceleration = entity.force * entity.inv_mass
        entity.vel += acceleration * dt
        if entity.vel[1] < self.TERMINAL_VELOCITY:
            entity.vel[1] = self.TERMINAL_VELOCITY
        entity.pos += entity.vel * dt
        entity.force = np.zeros(3)

    def handle_collisions(self, entity: PhysicsObject):
        """Simple Axis-Aligned Bounding Box (AABB) collision with voxel grid."""
        entity.on_ground = False
        min_c = entity.pos - entity.size / 2
        max_c = entity.pos + entity.size / 2
        min_ix, min_iy, min_iz = map(int, np.floor(min_c))
        max_ix, max_iy, max_iz = map(int, np.ceil(max_c))

        for x in range(min_ix, max_ix + 1):
            for y in range(min_iy, max_iy + 1):
                for z in range(min_iz, max_iz + 1):
                    # FIX: Use the high-performance get_block_type method
                    block_type = self.world.get_block_type(x, y, z)
                    if block_type != BlockType.AIR:
                        self.resolve_collision(entity, (x, y, z))

    def resolve_collision(self, entity: PhysicsObject, block_pos: tuple[int, int, int]):
        """Resolves an AABB collision by pushing the entity out."""
        block_min = np.array(block_pos, dtype=float)
        block_max = block_min + 1.0
        entity_min = entity.pos - entity.size / 2
        entity_max = entity.pos + entity.size / 2

        overlap_x = min(entity_max[0], block_max[0]) - max(entity_min[0], block_min[0])
        overlap_y = min(entity_max[1], block_max[1]) - max(entity_min[1], block_min[1])
        overlap_z = min(entity_max[2], block_max[2]) - max(entity_min[2], block_min[2])

        if overlap_x < 0 or overlap_y < 0 or overlap_z < 0: return

        overlaps = [overlap_x, overlap_y, overlap_z]
        min_axis = np.argmin(overlaps)

        if min_axis == 0: # X-axis collision
            direction = np.sign(entity.pos[0] - (block_pos[0] + 0.5))
            entity.pos[0] += direction * overlap_x
            entity.vel[0] = 0
        elif min_axis == 1: # Y-axis collision
            # FIX: Correct on_ground logic.
            was_falling = entity.vel[1] <= 0
            direction = np.sign(entity.pos[1] - (block_pos[1] + 0.5))
            entity.pos[1] += direction * overlap_y
            entity.vel[1] = 0
            # Only set on_ground if we were falling and got pushed up.
            if was_falling and direction > 0:
                entity.on_ground = True
        elif min_axis == 2: # Z-axis collision
            direction = np.sign(entity.pos[2] - (block_pos[2] + 0.5))
            entity.pos[2] += direction * overlap_z
            entity.vel[2] = 0

    def apply_quantum_effects(self, entity: QuantumObject, dt: float):
        """Simulates quantum behavior for an unobserved quantum object."""
        # Use a more controlled random perturbation
        if random.random() < config.QUANTUM_PERTURBATION_CHANCE:
             perturbation = (np.random.rand(3) - 0.5) * config.QUANTUM_PERTURBATION_MAGNITUDE
             entity.pos += perturbation
