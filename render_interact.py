# core_world.py
# Contains: WorldState, Chunk, Block, PhysicsObject, QuantumObject, PhysicsEngine
# This script is responsible for all physics, world state, and core quantum logic.
# It is designed to be completely independent of rendering and user input for CPU optimization.

import numpy as np
import random
from collections import defaultdict

# --- Constants and Enumerations ---
class BlockType:
    """Defines the types of blocks available in the world."""
    AIR = 0
    STONE = 1
    WOOD = 2
    WATER = 3
    QUANTUM_ORE = 4

class Material:
    """Defines physical properties for materials."""
    STONE = 0
    WOOD = 1
    METAL = 2
    QUANTUM = 3

# --- Core Data Structures ---

class Block:
    """Represents a single voxel in the world."""
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
        # Initialize a 3D NumPy array for blocks
        self.blocks = np.full(
            (self.CHUNK_SIZE_X, self.CHUNK_HEIGHT, self.CHUNK_SIZE_Z),
            fill_value=BlockType.AIR,
            dtype=int
        )

    def get_block(self, x, y, z):
        """Gets a block at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            return self.blocks[x, y, z]
        return None

    def set_block(self, x, y, z, block_type):
        """Sets a block at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            self.blocks[x, y, z] = block_type
            return True
        return False

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

    def get_chunk_coord(self, world_x, world_z):
        """Converts world coordinates to chunk coordinates."""
        return (int(world_x // Chunk.CHUNK_SIZE_X), int(world_z // Chunk.CHUNK_SIZE_Z))

    def get_or_create_chunk(self, chunk_x, chunk_z):
        """Retrieves a chunk, creating it if it doesn't exist."""
        if (chunk_x, chunk_z) not in self.chunks:
            if self.generator:
                self.chunks[(chunk_x, chunk_z)] = self.generate_chunk_from_data(chunk_x, chunk_z)
            else:
                # Fallback if no generator is attached
                self.chunks[(chunk_x, chunk_z)] = Chunk()
        return self.chunks[(chunk_x, chunk_z)]

    def generate_chunk_from_data(self, chunk_x, chunk_z):
        """Creates a chunk object from procedurally generated data."""
        chunk = Chunk()
        chunk_data = self.generator.generate_chunk(chunk_x, chunk_z, Chunk.CHUNK_SIZE_X)
        chunk.blocks = chunk_data
        return chunk

    def get_block(self, world_x, world_y, world_z):
        """Gets a block at absolute world coordinates."""
        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        if chunk:
            local_x = int(world_x % Chunk.CHUNK_SIZE_X)
            local_y = int(world_y)
            local_z = int(world_z % Chunk.CHUNK_SIZE_Z)
            block_type = chunk.get_block(local_x, local_y, local_z)
            if block_type is not None:
                return Block(block_type)
        return Block(BlockType.AIR) # Return air if out of bounds or chunk doesn't exist

    def set_block(self, world_x, world_y, world_z, block_type):
        """Sets a block at absolute world coordinates."""
        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)
        if chunk:
            local_x = int(world_x % Chunk.CHUNK_SIZE_X)
            local_y = int(world_y)
            local_z = int(world_z % Chunk.CHUNK_SIZE_Z)
            return chunk.set_block(local_x, local_y, local_z, block_type)
        return False

    def add_entity(self, entity):
        """Adds a new entity to the world."""
        self.entities.append(entity)

    def step_simulation(self, dt):
        """Advances the entire world state by one time step."""
        self.physics_engine.update(dt)

class PhysicsEngine:
    """Handles all physics calculations for the world."""
    GRAVITY = np.array([0, -9.81, 0])
    TERMINAL_VELOCITY = -50.0

    def __init__(self, world_state):
        self.world = world_state

    def update(self, dt):
        """Updates all entities."""
        for entity in self.world.entities:
            self.apply_gravity(entity)
            self.update_entity_position(entity, dt)
            self.handle_collisions(entity)
            if isinstance(entity, QuantumObject) and not entity.is_observed:
                self.apply_quantum_effects(entity, dt)

    def apply_gravity(self, entity):
        """Applies gravity to a single entity."""
        if not entity.on_ground:
            entity.force += self.GRAVITY * entity.mass

    def update_entity_position(self, entity, dt):
        """Updates position and velocity based on forces."""
        if entity.inv_mass == 0: return
        acceleration = entity.force * entity.inv_mass
        entity.vel += acceleration * dt
        if entity.vel[1] < self.TERMINAL_VELOCITY:
            entity.vel[1] = self.TERMINAL_VELOCITY
        entity.pos += entity.vel * dt
        entity.force = np.zeros(3)

    def handle_collisions(self, entity):
        """Simple Axis-Aligned Bounding Box (AABB) collision with voxel grid."""
        entity.on_ground = False
        min_c = entity.pos - entity.size / 2
        max_c = entity.pos + entity.size / 2
        min_ix, min_iy, min_iz = map(int, np.floor(min_c))
        max_ix, max_iy, max_iz = map(int, np.ceil(max_c))

        for x in range(min_ix, max_ix):
            for y in range(min_iy, max_iy):
                for z in range(min_iz, max_iz):
                    block = self.world.get_block(x, y, z)
                    if block and block.type != BlockType.AIR:
                        self.resolve_collision(entity, (x, y, z))

    def resolve_collision(self, entity, block_pos):
        """Resolves an AABB collision by pushing the entity out."""
        block_min = np.array(block_pos)
        block_max = block_min + 1.0
        entity_min = entity.pos - entity.size / 2
        entity_max = entity.pos + entity.size / 2

        overlap_x = min(entity_max[0], block_max[0]) - max(entity_min[0], block_min[0])
        overlap_y = min(entity_max[1], block_max[1]) - max(entity_min[1], block_min[1])
        overlap_z = min(entity_max[2], block_max[2]) - max(entity_min[2], block_min[2])

        if overlap_x < 0 or overlap_y < 0 or overlap_z < 0: return

        overlaps = [overlap_x, overlap_y, overlap_z]
        min_axis = np.argmin(overlaps)

        if min_axis == 0:
            direction = np.sign(entity.pos[0] - (block_pos[0] + 0.5))
            entity.pos[0] += direction * overlap_x
            entity.vel[0] = 0
        elif min_axis == 1:
            direction = np.sign(entity.pos[1] - (block_pos[1] + 0.5))
            entity.pos[1] += direction * overlap_y
            if direction > 0: # Corrected: Pushed up from below
                entity.on_ground = True
            entity.vel[1] = 0
        elif min_axis == 2:
            direction = np.sign(entity.pos[2] - (block_pos[2] + 0.5))
            entity.pos[2] += direction * overlap_z
            entity.vel[2] = 0

    def apply_quantum_effects(self, entity, dt):
        """Simulates quantum behavior for an unobserved quantum object."""
        if random.random() < 0.05:
             perturbation = (np.random.rand(3) - 0.5) * 0.2
             entity.pos += perturbation
