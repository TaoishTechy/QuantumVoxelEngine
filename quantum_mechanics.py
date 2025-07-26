# quantum_mechanics.py
# Implements the predictive player model for world streaming and quantum entropy caching.

import numpy as np
from typing import Dict, Tuple, List, Deque
from collections import deque, OrderedDict
import zlib
import threading

from logger import logger
import config
import core_world

class LOD:
    """Enumeration for Level of Detail states."""
    QUANTUM = 0
    MOLECULAR = 1
    CLASSICAL = 2

class QuantumPlayerModel:
    """
    Models player movement as a wave function to predict future locations
    for the chunk streaming and LOD system.
    """
    def __init__(self):
        self.position_history: Deque[np.ndarray] = deque(maxlen=30)
        self.velocity: np.ndarray = np.zeros(3)

    def update(self, player_pos: np.ndarray, dt: float):
        """Updates the model with the player's current position."""
        if len(self.position_history) > 0:
            current_vel = (player_pos - self.position_history[-1]) / (dt + 1e-6)
            self.velocity = self.velocity * 0.9 + current_vel * 0.1
        self.position_history.append(player_pos.copy())

    def get_chunk_load_probabilities(self, current_chunk: Tuple[int, int]) -> Dict[Tuple[int, int], float]:
        """
        Calculates a probability distribution for chunks to load based on a
        Schrödinger-inspired wave function of the player's likely future positions.
        """
        probabilities: Dict[Tuple[int, int], float] = {}
        if len(self.position_history) < 2:
            return { (cx, cz): 1.0 for cx in range(current_chunk[0] - config.RENDER_DISTANCE, current_chunk[0] + config.RENDER_DISTANCE + 1)
                     for cz in range(current_chunk[1] - config.RENDER_DISTANCE, current_chunk[1] + config.RENDER_DISTANCE + 1) }

        predicted_pos = self.position_history[-1] + self.velocity * 0.75
        predicted_chunk = (int(predicted_pos[0] // 16), int(predicted_pos[2] // 16))

        total_probability = 0
        for x in range(current_chunk[0] - config.RENDER_DISTANCE, current_chunk[0] + config.RENDER_DISTANCE + 1):
            for z in range(current_chunk[1] - config.RENDER_DISTANCE, current_chunk[1] + config.RENDER_DISTANCE + 1):
                dist_sq = (x - predicted_chunk[0])**2 + (z - predicted_chunk[1])**2
                probability = np.exp(-dist_sq / (config.RENDER_DISTANCE * 2.5))
                probabilities[(x, z)] = probability
                total_probability += probability

        if total_probability > 0:
            for coord in probabilities:
                probabilities[coord] /= total_probability

        return probabilities

class QuantumEntropyCache:
    """
    An intelligent, thread-safe memory management system that uses quantum entropy
    to prioritize which chunks to keep in memory, compress, or evict.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.vram_cache: Dict[Tuple[int, int], Dict] = {} # Stores active VBOs
        self.ram_cache: Dict[Tuple[int, int], bytes] = {} # Stores compressed voxel data
        self.lock = threading.Lock()

    def add(self, chunk_coord: Tuple[int, int], vbo: 'rendering.ChunkVBO', chunk: 'core_world.Chunk'):
        """Adds a chunk's VBO to the cache, calculating its entropy."""
        with self.lock:
            self.vram_cache[chunk_coord] = {
                'vbo': vbo,
                'entropy': chunk.calculate_entropy(),
                'last_access': time.time()
            }

    def get_chunks_to_evict(self, chunks_in_view: List[Tuple[int, int]],
                            probabilities: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
        """Determines which chunks to evict from VRAM based on entropy and view probability."""
        if len(self.vram_cache) <= self.max_size:
            return []

        with self.lock:
            candidates = [coord for coord in self.vram_cache if coord not in chunks_in_view]
            if not candidates: return []

            # Score candidates: lower is better to evict
            # Low entropy and low view probability = high eviction priority
            def eviction_score(coord):
                prob = probabilities.get(coord, 0)
                entropy = self.vram_cache[coord]['entropy']
                # Invert entropy so high entropy = low score
                return (1.0 - prob) * (1.0 / (1 + entropy))

            candidates.sort(key=eviction_score, reverse=True)
            num_to_evict = len(self.vram_cache) - self.max_size
            return candidates[:num_to_evict]

    def evict(self, chunk_coord: Tuple[int, int], world: 'core_world.WorldState'):
        """Evicts a chunk from VRAM, compressing it to RAM if its entropy is low."""
        with self.lock:
            if chunk_coord not in self.vram_cache: return

            cache_entry = self.vram_cache.pop(chunk_coord)
            vbo = cache_entry['vbo']
            entropy = cache_entry['entropy']

            # Destroy the GPU resources
            vbo.destroy()

            # For low-entropy chunks, compress and move to RAM cache
            if entropy < 5000: # Threshold for what's considered "simple"
                chunk = world.get_or_create_chunk(*chunk_coord)
                compressed_data = zlib.compress(chunk.blocks.tobytes())
                self.ram_cache[chunk_coord] = compressed_data
                logger.info(f"Evicted chunk {chunk_coord} from VRAM and compressed to RAM (entropy: {entropy:.0f}).")
            else:
                logger.info(f"Evicted high-entropy chunk {chunk_coord} from VRAM (entropy: {entropy:.0f}).")
