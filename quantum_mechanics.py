# quantum_mechanics.py
# Implements the predictive player model for world streaming and quantum entropy caching.

import numpy as np
from typing import Dict, Tuple, List
from collections import deque
from logger import logger

class QuantumPlayerModel:
    """
    Models player movement as a wave function to predict future locations
    for the chunk streaming system.
    """
    def __init__(self):
        self.position_history = deque(maxlen=30)
        self.velocity = np.zeros(3)

    def update(self, player_pos: np.ndarray, dt: float):
        """Updates the model with the player's current position."""
        if len(self.position_history) > 0:
            self.velocity = (player_pos - self.position_history[-1]) / dt
        self.position_history.append(player_pos.copy())

    def get_chunk_load_probabilities(self, current_chunk: Tuple[int, int], render_distance: int) -> Dict[Tuple[int, int], float]:
        """
        Calculates a probability distribution for chunks to load.
        Chunks in the direction of player movement are prioritized.
        """
        probabilities = {}
        if len(self.position_history) < 2:
            # If no movement data, load uniformly around player
            for x in range(current_chunk[0] - render_distance, current_chunk[0] + render_distance + 1):
                for z in range(current_chunk[1] - render_distance, current_chunk[1] + render_distance + 1):
                    probabilities[(x, z)] = 1.0
            return probabilities

        # Predict future position
        predicted_pos = self.position_history[-1] + self.velocity * 0.5 # Predict 0.5s ahead
        predicted_chunk = (int(predicted_pos[0] // 16), int(predicted_pos[2] // 16))

        # Create a probability cloud centered on the predicted chunk
        for x in range(current_chunk[0] - render_distance, current_chunk[0] + render_distance + 1):
            for z in range(current_chunk[1] - render_distance, current_chunk[1] + render_distance + 1):
                dist_sq = (x - predicted_chunk[0])**2 + (z - predicted_chunk[1])**2
                # Use Gaussian-like falloff from predicted center
                probability = np.exp(-dist_sq / (render_distance * 2))
                probabilities[(x, z)] = probability
        
        return probabilities

class QuantumEntropyCache:
    """
    An intelligent memory management system that uses quantum entropy
    to prioritize which chunks to keep in memory.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: Dict[Tuple[int, int], Dict] = {} # chunk_coord -> {'entropy': float, 'vbo': ChunkVBO}

    def update_chunk_entropy(self, chunk_coord: Tuple[int, int], chunk: 'core_world.Chunk'):
        """Calculates and updates the entropy for a given chunk."""
        if chunk_coord in self.cache:
            self.cache[chunk_coord]['entropy'] = chunk.calculate_entropy()

    def get_chunks_to_evict(self, chunks_in_view: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Determines which chunks to evict from memory based on entropy and distance."""
        if len(self.cache) <= self.max_size:
            return []

        # Candidate chunks for eviction are those not in the current view
        candidates = [coord for coord in self.cache if coord not in chunks_in_view]
        
        if not candidates:
            return []

        # Sort candidates by entropy (lowest first) to evict simple chunks
        candidates.sort(key=lambda c: self.cache[c].get('entropy', 0))
        
        num_to_evict = len(self.cache) - self.max_size
        return candidates[:num_to_evict]

    def add(self, chunk_coord: Tuple[int, int], vbo: 'rendering.ChunkVBO', chunk: 'core_world.Chunk'):
        """Adds a chunk's VBO to the cache."""
        self.cache[chunk_coord] = {
            'vbo': vbo,
            'entropy': chunk.calculate_entropy()
        }

    def remove(self, chunk_coord: Tuple[int, int]):
        """Removes a chunk from the cache."""
        if chunk_coord in self.cache:
            del self.cache[chunk_coord]
