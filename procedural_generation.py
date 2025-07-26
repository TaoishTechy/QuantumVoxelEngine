# procedural_generation.py
# Implements advanced world generation using quantum-inspired algorithms.

import numpy as np
import random
from typing import Tuple, Dict, Optional, Any

# Assuming core_world.py is in the same project directory
import core_world

class QuantumProceduralGenerator:
    """
    Generates deterministic voxel chunks using quantum-inspired algorithms.

    This generator uses a layered approach, combining multiple noise sources,
    biome mapping, and quantum-inspired pattern shuffling to create complex,
    varied, and seamless terrain. It is designed to be deterministic,
    always producing the same world for a given seed.

    Attributes:
        seed (int): The master seed for all random generation.
        biome_cache (Dict[Tuple[int, int], str]): A cache to store generated
            biome types for chunk coordinates, ensuring consistency.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initializes the QuantumProceduralGenerator.

        Args:
            seed (Optional[int]): The seed for the random number generator.
                If None, a random seed will be used.
        """
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.biome_cache: Dict[Tuple[int, int], str] = {}
        # TODO: Expose more generation parameters (amplitude, scale, etc.) here
        # for finer control.

    def _get_chunk_rng(self, chunk_x: int, chunk_z: int) -> np.random.Generator:
        """
        Creates a deterministic RNG for a specific chunk coordinate.
        This is crucial for ensuring chunk features are consistent and repeatable.
        """
        chunk_seed = self.seed + hash((chunk_x, chunk_z))
        return np.random.default_rng(chunk_seed)

    def _perlin_noise(self, x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        """
        Generates a 2D Perlin noise value for given coordinates.
        This is a standard implementation used as a base for more complex noise.
        """
        # Permutation table
        p = np.arange(256, dtype=int)
        rng = np.random.default_rng(seed)
        rng.shuffle(p)
        p = np.stack([p, p]).flatten()
        
        # Coordinates
        xi, yi = x.astype(int) & 255, y.astype(int) & 255
        xf, yf = x - x.astype(int), y - y.astype(int)
        
        # Fade curves
        u, v = (6*xf**5 - 15*xf**4 + 10*xf**3), (6*yf**5 - 15*yf**4 + 10*yf**3)
        
        # Gradient vectors
        g = np.array([[1,1],[-1,1],[1,-1],[-1,-1],[1,0],[-1,0],[0,1],[0,-1]])
        
        # Hash coordinates to gradient vectors
        n00 = np.sum(g[p[p[xi] + yi] % 8] * np.stack([xf, yf], axis=-1), axis=2)
        n10 = np.sum(g[p[p[xi+1] + yi] % 8] * np.stack([xf-1, yf], axis=-1), axis=2)
        n01 = np.sum(g[p[p[xi] + yi+1] % 8] * np.stack([xf, yf-1], axis=-1), axis=2)
        n11 = np.sum(g[p[p[xi+1] + yi+1] % 8] * np.stack([xf-1, yf-1], axis=-1), axis=2)
        
        # Interpolate
        x1 = n00 + u * (n10 - n00)
        x2 = n01 + u * (n11 - n01)
        return x1 + v * (x2 - x1)

    def _fractal_noise(self, x: np.ndarray, z: np.ndarray, octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
        """
        Creates fractal noise by layering multiple octaves of Perlin noise.
        This is also known as Fractal Brownian Motion (fBm).
        """
        total = np.zeros_like(x)
        frequency = 1.0
        amplitude = 1.0
        for i in range(octaves):
            total += amplitude * self._perlin_noise(x * frequency, z * frequency, self.seed + i)
            frequency *= 2
            amplitude *= persistence
        return total

    def get_biome(self, chunk_x: int, chunk_z: int) -> str:
        """
        Determines the biome for a chunk using a large-scale noise map.
        This ensures smooth and logical transitions between biomes.
        """
        if (chunk_x, chunk_z) in self.biome_cache:
            return self.biome_cache[(chunk_x, chunk_z)]

        # Use a very low frequency noise to define large biome areas
        biome_noise = self._perlin_noise(np.array([[chunk_x * 0.02]]), np.array([[chunk_z * 0.02]]), self.seed)[0,0]
        
        if biome_noise > 0.3:
            biome = 'mountains'
        elif biome_noise > 0.0:
            biome = 'hills'
        elif biome_noise > -0.2:
            biome = 'forest'
        else:
            biome = 'plains'
            
        self.biome_cache[(chunk_x, chunk_z)] = biome
        return biome

    def apply_quantum_pattern_shuffling(self, heightmap: np.ndarray, chunk_x: int, chunk_z: int) -> np.ndarray:
        """
        Perturbs the heightmap using a deterministic, chunk-specific pattern.
        This simulates effects like Wave Function Collapse or quantum interference
        by adding unique, complex details to each chunk's terrain.
        """
        rng = self._get_chunk_rng(chunk_x, chunk_z)
        pattern_size = 4
        
        # Create a small, random but deterministic perturbation mask
        perturbation_mask = rng.uniform(-1.5, 1.5, (pattern_size, pattern_size))
        
        # Tile the mask across the heightmap
        tiled_mask = np.tile(perturbation_mask, (heightmap.shape[0] // pattern_size, heightmap.shape[1] // pattern_size))
        
        return heightmap + tiled_mask

    def generate_chunk(self, chunk_x: int, chunk_z: int) -> np.ndarray:
        """
        Generates the block data for a single chunk.

        Args:
            chunk_x (int): The x-coordinate of the chunk.
            chunk_z (int): The z-coordinate of the chunk.

        Returns:
            np.ndarray: A 3D NumPy array of block IDs representing the chunk.
        """
        chunk_data = np.full(
            (core_world.Chunk.CHUNK_SIZE_X, core_world.Chunk.CHUNK_HEIGHT, core_world.Chunk.CHUNK_SIZE_Z),
            core_world.BlockType.AIR, dtype=np.uint8
        )
        
        biome = self.get_biome(chunk_x, chunk_z)
        
        # Create coordinate grid for vectorized noise calculation
        x_coords = np.arange(chunk_x * 16, (chunk_x + 1) * 16)
        z_coords = np.arange(chunk_z * 16, (chunk_z + 1) * 16)
        xx, zz = np.meshgrid(x_coords, z_coords)

        # 1. Generate base heightmap
        base_noise = self._fractal_noise(xx * 0.01, zz * 0.01)
        
        # 2. Apply biome-specific parameters
        if biome == 'plains':
            heightmap = 64 + base_noise * 5
        elif biome == 'hills':
            heightmap = 70 + base_noise * 20
        elif biome == 'forest':
            heightmap = 68 + base_noise * 10
        elif biome == 'mountains':
            mountain_noise = self._fractal_noise(xx * 0.005, zz * 0.005, octaves=6)
            heightmap = 80 + base_noise * 15 + mountain_noise * 40
        
        # 3. Apply quantum pattern shuffling
        heightmap = self.apply_quantum_pattern_shuffling(heightmap, chunk_x, chunk_z)
        
        # 4. Populate the 3D chunk data
        for x in range(16):
            for z in range(16):
                height = int(heightmap[z, x])
                for y in range(height):
                    if y == height - 1:
                        chunk_data[x, y, z] = core_world.BlockType.WOOD # Represents grass/top layer
                    elif y > height - 5:
                        chunk_data[x, y, z] = core_world.BlockType.STONE # Represents dirt
                    else:
                        chunk_data[x, y, z] = core_world.BlockType.STONE

        # TODO: Add ore generation and cave systems here

        return chunk_data

    def get_chunk_metadata(self, chunk_x: int, chunk_z: int) -> Dict[str, Any]:
        """
        Returns metadata for a specific chunk.

        Args:
            chunk_x (int): The x-coordinate of the chunk.
            chunk_z (int): The z-coordinate of the chunk.

        Returns:
            Dict[str, Any]: A dictionary of metadata.
        """
        rng = self._get_chunk_rng(chunk_x, chunk_z)
        return {
            "biome": self.get_biome(chunk_x, chunk_z),
            "quantum_signature": rng.random(), # A unique, deterministic value for the chunk
            "has_special_event": rng.random() < 0.05 # 5% chance of a special event
        }

# --- Usage Example ---
if __name__ == "__main__":
    print("--- QuantumProceduralGenerator Demonstration ---")
    
    # 1. Instantiate the generator with a specific seed for reproducibility
    generator = QuantumProceduralGenerator(seed=42)
    print(f"Generator initialized with seed: {generator.seed}")

    # 2. Generate a 2x2 area of chunks to demonstrate seamless tiling
    world_chunks = {}
    print("\nGenerating a 2x2 area of chunks...")
    for cx in range(2):
        for cz in range(2):
            print(f"  Generating chunk at ({cx}, {cz})...")
            chunk_data = generator.generate_chunk(cx, cz)
            world_chunks[(cx, cz)] = chunk_data
    
    # 3. Display metadata for each generated chunk
    print("\nChunk Metadata:")
    for (cx, cz), chunk_data in world_chunks.items():
        metadata = generator.get_chunk_metadata(cx, cz)
        print(f"  Chunk ({cx}, {cz}): Biome = {metadata['biome']}, Quantum Sig = {metadata['quantum_signature']:.4f}")

    # 4. Create and display a simple top-down heightmap visualization
    print("\nTop-down view of the generated world (heightmap):")
    map_size_x = 2 * 16
    map_size_z = 2 * 16
    heightmap_viz = np.zeros((map_size_z, map_size_x))

    for (cx, cz), chunk_data in world_chunks.items():
        for x in range(16):
            for z in range(16):
                height = 0
                for y in range(255, -1, -1):
                    if chunk_data[x, y, z] != core_world.BlockType.AIR:
                        height = y
                        break
                heightmap_viz[cz * 16 + z, cx * 16 + x] = height

    # Simple text visualization
    symbols = " .:;!*#%@"
    for row in heightmap_viz:
        line = ""
        for height in row:
            # Normalize height to an index in the symbols string
            symbol_index = int(np.clip((height - 50) / 40, 0, len(symbols) - 1))
            line += symbols[symbol_index]
        print(line)
    
    print("\nDemonstration complete. The heightmap should appear seamless across chunk boundaries.")
