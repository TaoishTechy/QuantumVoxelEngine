# procedural_generation.py
# Implements advanced world generation using quantum-inspired algorithms.

import numpy as np
import random
from typing import Tuple, Dict, Optional, Any

import core_world # Import corrected core_world
import config # Import centralized constants

class QuantumProceduralGenerator:
    """
    Generates deterministic voxel chunks using quantum-inspired algorithms.
    This version uses vectorized operations for significantly improved performance.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initializes the QuantumProceduralGenerator.
        """
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.biome_cache: Dict[Tuple[int, int], str] = {}
        # Perlin noise permutation table, initialized once
        self._p = np.arange(256, dtype=int)
        np.random.default_rng(self.seed).shuffle(self._p)
        self._p = np.stack([self._p, self._p]).flatten()

    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Fade function as defined by Ken Perlin."""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def _lerp(self, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return a + x * (b - a)

    def _grad(self, hash_val: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates the dot product between a gradient vector and distance vector."""
        h = hash_val & 15
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, 0))
        return np.where(h & 1 == 0, u, -u) + np.where(h & 2 == 0, v, -v)

    def _perlin_noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized Perlin noise implementation."""
        xi, yi = x.astype(int), y.astype(int)
        xf, yf = x - xi, y - yi

        u, v = self._fade(xf), self._fade(yf)

        p = self._p
        xi &= 255
        yi &= 255

        g00 = self._grad(p[p[xi] + yi], xf, yf)
        g10 = self._grad(p[p[xi + 1] + yi], xf - 1, yf)
        g01 = self._grad(p[p[xi] + yi + 1], xf, yf - 1)
        g11 = self._grad(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)

        x1 = self._lerp(g00, g10, u)
        x2 = self._lerp(g01, g11, u)
        return self._lerp(x1, x2, v)

    def _fractal_noise(self, x: np.ndarray, z: np.ndarray, octaves: int, persistence: float = 0.5) -> np.ndarray:
        """Creates fractal noise by layering multiple octaves of Perlin noise."""
        total = np.zeros_like(x, dtype=float)
        frequency = 1.0
        amplitude = 1.0
        for i in range(octaves):
            total += amplitude * self._perlin_noise(x * frequency, z * frequency)
            frequency *= 2
            amplitude *= persistence
        return total

    def get_biome(self, chunk_x: int, chunk_z: int) -> str:
        """Determines the biome for a chunk using a large-scale noise map."""
        if (chunk_x, chunk_z) in self.biome_cache:
            return self.biome_cache[(chunk_x, chunk_z)]

        biome_noise = self._perlin_noise(
            np.array([[chunk_x * config.BIOME_SCALE]]),
            np.array([[chunk_z * config.BIOME_SCALE]])
        )[0,0]

        if biome_noise > 0.3: biome = 'mountains'
        elif biome_noise > 0.0: biome = 'hills'
        elif biome_noise > -0.2: biome = 'forest'
        else: biome = 'plains'

        self.biome_cache[(chunk_x, chunk_z)] = biome
        return biome

    def generate_chunk(self, chunk_x: int, chunk_z: int) -> np.ndarray:
        """Generates the block data for a single chunk using vectorized operations."""
        chunk_data = np.zeros(
            (core_world.Chunk.CHUNK_SIZE_X, core_world.Chunk.CHUNK_HEIGHT, core_world.Chunk.CHUNK_SIZE_Z),
            dtype=np.uint8
        )

        biome = self.get_biome(chunk_x, chunk_z)

        x_coords = np.arange(chunk_x * 16, (chunk_x + 1) * 16)
        z_coords = np.arange(chunk_z * 16, (chunk_z + 1) * 16)
        xx, zz = np.meshgrid(x_coords, z_coords)

        base_noise = self._fractal_noise(xx * 0.01, zz * 0.01, octaves=4)

        if biome == 'plains': heightmap = config.TERRAIN_BASE_HEIGHT + base_noise * 5
        elif biome == 'hills': heightmap = config.TERRAIN_BASE_HEIGHT + 6 + base_noise * config.HILL_AMPLITUDE
        elif biome == 'forest': heightmap = config.TERRAIN_BASE_HEIGHT + 4 + base_noise * 10
        else: # mountains
            mountain_noise = self._fractal_noise(xx * 0.005, zz * 0.005, octaves=config.FRACTAL_OCTAVES)
            heightmap = config.TERRAIN_BASE_HEIGHT + 16 + base_noise * 15 + mountain_noise * config.MOUNTAIN_AMPLITUDE

        # Ensure heightmap does not exceed chunk boundaries before using it
        heightmap = np.clip(heightmap, 0, core_world.Chunk.CHUNK_HEIGHT - 1).astype(int)

        # Create a 3D Y-coordinate grid to compare against the heightmap
        y_coords = np.arange(core_world.Chunk.CHUNK_HEIGHT).reshape(1, -1, 1)

        # Use broadcasting to create boolean masks for each block type
        heightmap_3d = heightmap.T.reshape(16, 1, 16)

        stone_mask = y_coords < (heightmap_3d - 4)
        dirt_mask = (y_coords >= (heightmap_3d - 4)) & (y_coords < heightmap_3d)
        grass_mask = y_coords == (heightmap_3d - 1)

        # Apply masks to the chunk data array in one shot
        chunk_data[stone_mask] = core_world.BlockType.STONE
        chunk_data[dirt_mask] = core_world.BlockType.DIRT
        chunk_data[grass_mask] = core_world.BlockType.GRASS

        # TODO: Add ore generation and cave systems using 3D noise and masks.

        return chunk_data
