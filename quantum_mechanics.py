# quantum_mechanics.py
# Contains systems for managing core quantum phenomena like entanglement,
# superposition, and wave function collapse.

import numpy as np
import random

class QuantumEntanglementNetwork:
    """A dynamic system for creating and managing complex entanglement webs."""
    def __init__(self):
        self.entanglement_graph = {}
        self.entanglement_strength = {}

    def add_entanglement_pair(self, obj1, obj2):
        """Entangles two quantum objects."""
        strength = random.uniform(0.5, 1.0)
        self.entanglement_graph.setdefault(obj1.id, []).append(obj2.id)
        self.entanglement_graph.setdefault(obj2.id, []).append(obj1.id)
        self.entanglement_strength[(obj1.id, obj2.id)] = strength
        self.entanglement_strength[(obj2.id, obj1.id)] = strength

    def propagate_measurement_collapse(self, measured_obj, all_objects):
        """When one object is measured, all entangled objects collapse instantly."""
        if measured_obj.id in self.entanglement_graph:
            for partner_id in self.entanglement_graph[measured_obj.id]:
                partner = next((obj for obj in all_objects if obj.id == partner_id), None)
                if partner and not partner.is_observed:
                    partner.is_observed = True
                    partner.pos = measured_obj.pos + np.random.normal(0, 0.1, 3)
                    partner.vel = measured_obj.vel * 0.9

class SuperpositionField:
    """Creates regions where quantum objects can exist in multiple states simultaneously."""
    def __init__(self, center, radius, num_states=3):
        self.center = np.array(center)
        self.radius = radius
        self.num_states = num_states

    def apply_superposition(self, quantum_obj):
        """Places a quantum object in a superposition of states within the field."""
        if not quantum_obj.is_observed and np.linalg.norm(quantum_obj.pos - self.center) <= self.radius:
            quantum_obj.ghost_positions = []
            for i in range(self.num_states):
                angle = (2 * np.pi * i) / self.num_states
                offset = np.array([np.cos(angle), np.sin(angle), 0]) * self.radius * 0.3
                quantum_obj.ghost_positions.append({'position': self.center + offset})

    def collapse_superposition(self, quantum_obj):
        """Collapses the superposition to a single state upon measurement."""
        if hasattr(quantum_obj, 'ghost_positions') and quantum_obj.ghost_positions:
            chosen_state = random.choice(quantum_obj.ghost_positions)
            quantum_obj.pos = chosen_state['position']
            quantum_obj.ghost_positions = []

class QuantumWaveFunctionCollapse:
    """Advanced wave function collapse system for procedural world generation."""
    def __init__(self, grid_size=(64, 64, 64)):
        self.grid_size = grid_size
        self.wave_function = {}
        self.constraints = {}
        self.entropy_map = {}

    def initialize_superposition_grid(self, possible_blocks):
        """Initialize every position in a superposition of all possible blocks."""
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                for z in range(self.grid_size[2]):
                    pos = (x, y, z)
                    self.wave_function[pos] = list(possible_blocks)
                    self.entropy_map[pos] = len(possible_blocks)

    def collapse_lowest_entropy(self):
        """Find the position with the lowest entropy and collapse it."""
        min_entropy = float('inf')
        candidates = []
        for pos, states in self.wave_function.items():
            if 1 < len(states) < min_entropy:
                min_entropy = len(states)
                candidates = [pos]
            elif len(states) == min_entropy:
                candidates.append(pos)

        if not candidates:
            return None

        chosen_pos = random.choice(candidates)
        chosen_state = random.choice(self.wave_function[chosen_pos])
        self.wave_function[chosen_pos] = [chosen_state]
        self.entropy_map[chosen_pos] = 1
        return chosen_pos, chosen_state
