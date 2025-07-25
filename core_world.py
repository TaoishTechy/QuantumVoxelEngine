# core_world.py
# This script is responsible for all physics, world state, and core quantum logic.

import numpy as np
from typing import Tuple, Optional, Dict
from settings import settings

class BlockType:
    AIR = 0
    STONE = 1
    WOOD = 2
    # You can add GRASS, DIRT, etc. here and map them in game_manager.py
    # GRASS = 3
    # DIRT = 4

class Block:
    """Represents a single voxel in the world."""
    def __init__(self, block_type: int = BlockType.AIR):
        self.type = block_type

class Chunk:
    """A 16x256x16 segment of the world."""
    CHUNK_SIZE_X = 16
    CHUNK_SIZE_Z = 16
    CHUNK_HEIGHT = 256

    def __init__(self):
        self.blocks = np.full(
            (self.CHUNK_SIZE_X, self.CHUNK_HEIGHT, self.CHUNK_SIZE_Z),
            fill_value=BlockType.AIR,
            dtype=np.uint8 # Use a memory-efficient integer type for block IDs
        )

    def get_block(self, x: int, y: int, z: int) -> int:
        """Gets a block type ID at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            return self.blocks[x, y, z]
        return BlockType.AIR

    def set_block(self, x: int, y: int, z: int, block_type: int) -> bool:
        """Sets a block at local chunk coordinates."""
        if 0 <= x < self.CHUNK_SIZE_X and 0 <= y < self.CHUNK_HEIGHT and 0 <= z < self.CHUNK_SIZE_Z:
            self.blocks[x, y, z] = block_type
            return True
        return False

class PhysicsObject:
    """Represents a dynamic entity in the world (e.g., player, item, particle)."""
    def __init__(self, pos: list, mass: float = 1.0, size: Tuple[float, float, float] = (1, 1, 1)):
        self.pos = np.array(pos, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.force = np.zeros(3, dtype=np.float32)
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0
        self.size = np.array(size, dtype=np.float32)
        self.on_ground = False

class QuantumObject(PhysicsObject):
    """An entity with quantum properties."""
    def __init__(self, pos: list, mass: float = 0.1, size: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        super().__init__(pos, mass, size)
        self.is_quantum = True
        self.is_observed = False
        self.probability_cloud_radius = 5.0

class WorldState:
    """Manages all chunks, entities, and the overall state of the world."""
    def __init__(self):
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.entities: list = []
        self.physics_engine = PhysicsEngine(self)
        self.generator: Optional['procedural_generation.QuantumProceduralGenerator'] = None

    def get_chunk_coord(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Converts world coordinates to chunk coordinates."""
        return (int(world_x // Chunk.CHUNK_SIZE_X), int(world_z // Chunk.CHUNK_SIZE_Z))

    def get_or_create_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """Retrieves a chunk, creating it if it doesn't exist."""
        coord = (chunk_x, chunk_z)
        if coord not in self.chunks:
            if self.generator:
                chunk = Chunk()
                chunk_data = self.generator.generate_chunk(chunk_x, chunk_z)
                chunk.blocks = chunk_data
                self.chunks[coord] = chunk
            else:
                self.chunks[coord] = Chunk() # Fallback to an empty chunk
        return self.chunks[coord]

    def get_block(self, world_x: float, world_y: float, world_z: float) -> Block:
        """Gets a block at absolute world coordinates."""
        if not (0 <= world_y < Chunk.CHUNK_HEIGHT):
            return Block(BlockType.AIR)

        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)

        local_x = int(world_x % Chunk.CHUNK_SIZE_X)
        local_z = int(world_z % Chunk.CHUNK_SIZE_Z)

        return Block(chunk.get_block(local_x, int(world_y), local_z))

    def set_block(self, world_x: float, world_y: float, world_z: float, block_type: int) -> bool:
        """Sets a block at absolute world coordinates."""
        if not (0 <= world_y < Chunk.CHUNK_HEIGHT):
            return False

        chunk_x, chunk_z = self.get_chunk_coord(world_x, world_z)
        chunk = self.get_or_create_chunk(chunk_x, chunk_z)

        local_x = int(world_x % Chunk.CHUNK_SIZE_X)
        local_z = int(world_z % Chunk.CHUNK_SIZE_Z)

        return chunk.set_block(local_x, int(world_y), local_z, block_type)

    def add_entity(self, entity: PhysicsObject):
        """Adds a new entity to the world."""
        self.entities.append(entity)

    def step_simulation(self, dt: float):
        """Advances the entire world state by one time step."""
        self.physics_engine.update(self.entities, dt)

class PhysicsEngine:
    """Handles all physics calculations for the world."""
    GRAVITY = np.array([0, -9.81, 0]) * settings.get('physics.gravity_multiplier', 1.0)
    TERMINAL_VELOCITY = -50.0

    def __init__(self, world_state: 'WorldState'):
        self.world = world_state

    def update(self, entities: list, dt: float):
        """Updates all entities."""
        for entity in entities:
            if not entity.on_ground:
                entity.force += self.GRAVITY * entity.mass

            # Apply forces and update velocity
            if entity.inv_mass > 0:
                acceleration = entity.force * entity.inv_mass
                entity.vel += acceleration * dt
                if entity.vel[1] < self.TERMINAL_VELOCITY:
                    entity.vel[1] = self.TERMINAL_VELOCITY
                entity.pos += entity.vel * dt

            entity.force = np.zeros(3) # Reset force for next frame
            self.handle_collisions(entity)

    def handle_collisions(self, entity: PhysicsObject):
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

    def resolve_collision(self, entity: PhysicsObject, block_pos: Tuple[int, int, int]):
        """Resolves an AABB collision by pushing the entity out."""
        block_min = np.array(block_pos)
        block_max = block_min + 1.0

        entity_min = entity.pos - entity.size / 2
        entity_max = entity.pos + entity.size / 2

        overlap_x = min(entity_max[0], block_max[0]) - max(entity_min[0], block_min[0])
        overlap_y = min(entity_max[1], block_max[1]) - max(entity_min[1], block_min[1])
        overlap_z = min(entity_max[2], block_max[2]) - max(entity_min[2], block_min[2])

        if overlap_x < 0 or overlap_y < 0 or overlap_z < 0: return

        min_axis = np.argmin([overlap_x, overlap_y, overlap_z])

        if min_axis == 0: # X-axis
            direction = np.sign(entity.pos[0] - (block_pos[0] + 0.5))
            entity.pos[0] += direction * overlap_x
            entity.vel[0] = 0
        elif min_axis == 1: # Y-axis
            direction = np.sign(entity.pos[1] - (block_pos[1] + 0.5))
            entity.pos[1] += direction * overlap_y
            if direction > 0:
                entity.on_ground = True
            entity.vel[1] = 0
        elif min_axis == 2: # Z-axis
            direction = np.sign(entity.pos[2] - (block_pos[2] + 0.5))
            entity.pos[2] += direction * overlap_z
            entity.vel[2] = 0
