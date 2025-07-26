# tests/test_core_world.py
# Unit tests for the core world and physics systems.

import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_world import WorldState, PhysicsObject, BlockType
from procedural_generation import QuantumProceduralGenerator
from settings import settings

@pytest.fixture
def world() -> WorldState:
    """Pytest fixture to create a fresh WorldState for each test."""
    world_instance = WorldState()
    # Attach a generator for tests that require chunk creation
    world_instance.generator = QuantumProceduralGenerator(seed=123)
    return world_instance

def test_world_state_initialization(world: WorldState):
    """Tests that the WorldState initializes correctly."""
    assert isinstance(world.chunks, dict)
    assert len(world.chunks) == 0
    assert isinstance(world.entities, list)
    assert len(world.entities) == 0
    assert world.physics_engine is not None

def test_add_entity(world: WorldState):
    """Tests that entities can be added to the world."""
    assert len(world.entities) == 0
    player = PhysicsObject(pos=[0, 100, 0], mass=60)
    world.add_entity(player)
    assert len(world.entities) == 1
    assert world.entities[0] is player

def test_player_gravity(world: WorldState):
    """Tests that gravity is applied correctly to a non-static entity."""
    player = PhysicsObject(pos=[0, 100, 0], mass=60)
    world.add_entity(player)
    
    # Player starts with zero force
    assert player.force[1] == 0
    
    # Simulate one physics step
    world.step_simulation(0.016)
    
    # After one step, the downward force from gravity should have been applied
    expected_gravity_force = world.physics_engine.GRAVITY[1] * player.mass
    assert player.force[1] == pytest.approx(expected_gravity_force)

def test_get_chunk_coord(world: WorldState):
    """Tests the conversion of world coordinates to chunk coordinates."""
    assert world.get_chunk_coord(0, 0) == (0, 0)
    assert world.get_chunk_coord(15, 15) == (0, 0)
    assert world.get_chunk_coord(16, 0) == (1, 0)
    assert world.get_chunk_coord(-1, -1) == (-1, -1)
    assert world.get_chunk_coord(-17, 31) == (-2, 1)

def test_get_or_create_chunk(world: WorldState):
    """Tests that chunks are created and cached correctly."""
    assert len(world.chunks) == 0
    
    # First access should create the chunk
    chunk1 = world.get_or_create_chunk(0, 0)
    assert len(world.chunks) == 1
    assert chunk1 is not None
    
    # Second access should return the same, cached chunk instance
    chunk2 = world.get_or_create_chunk(0, 0)
    assert len(world.chunks) == 1
    assert chunk1 is chunk2

def test_set_and_get_block(world: WorldState):
    """Tests that blocks can be set and retrieved from the world."""
    # Initially, the block should be AIR
    assert world.get_block(10, 50, 10).type == BlockType.AIR
    
    # Set a block
    world.set_block(10, 50, 10, BlockType.STONE)
    
    # Verify the block was set
    assert world.get_block(10, 50, 10).type == BlockType.STONE
    
    # Set it back to AIR
    world.set_block(10, 50, 10, BlockType.AIR)
    assert world.get_block(10, 50, 10).type == BlockType.AIR

def test_collision_and_on_ground_flag(world: WorldState):
    """Tests that a falling entity correctly collides and sets the on_ground flag."""
    # Place a stone block at y=50
    world.set_block(0, 50, 0, BlockType.STONE)
    
    # Create a player just above the block
    player = PhysicsObject(pos=[0, 52, 0], mass=60, size=(0.8, 1.8, 0.8))
    world.add_entity(player)
    
    assert not player.on_ground
    
    # Simulate for a few seconds to let the player fall
    for _ in range(100):
        world.step_simulation(0.016)
    
    # Player should now be on the ground and have a very small downward velocity
    assert player.on_ground
    assert abs(player.vel[1]) < 0.1
    # Player's bottom should be at y=51 (top of the block)
    player_bottom_y = player.pos[1] - player.size[1] / 2
    assert player_bottom_y == pytest.approx(51.0, abs=0.1)
