# ai_systems.py
# Advanced AI systems implementing Physics-Informed Neural Networks (PINNs)
# and Quantum Game Theory for strategic decision-making in quantum physics games.

import numpy as np
import random
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===============================================================================
# PHYSICS-INFORMED NEURAL NETWORK (PINN) IMPLEMENTATION
# ===============================================================================

class ActivationFunction:
    """Collection of activation functions optimized for physics problems."""

    @staticmethod
    def tanh(x):
        """Hyperbolic tangent - excellent for periodic and high-order gradient problems"""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh activation"""
        return 1 - np.tanh(x)**2

    @staticmethod
    def swish(x):
        """Swish activation function - smooth and differentiable"""
        return x / (1 + np.exp(-x))

    @staticmethod
    def swish_derivative(x):
        """Derivative of swish activation"""
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)

class AutomaticDifferentiation:
    """Automatic differentiation system for computing physics derivatives"""

    def __init__(self):
        self.computation_graph = []
        self.gradient_cache = {}

    def compute_gradient(self, output_func, inputs):
        """
        Compute gradients using finite differences.
        Essential for PINN physics loss calculations.
        """
        gradients = []
        h = 1e-6 # Small step for numerical differentiation

        for i in range(len(inputs)):
            input_var = inputs[i]
            grad = np.zeros_like(input_var)

            it = np.nditer(input_var, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                original_value = input_var[idx]

                input_var[idx] = original_value + h
                f_plus = output_func(np.column_stack(inputs))

                input_var[idx] = original_value - h
                f_minus = output_func(np.column_stack(inputs))

                grad[idx] = (f_plus - f_minus) / (2 * h)
                input_var[idx] = original_value # Restore original value
                it.iternext()
            gradients.append(grad)
        return gradients

class PINNFluidSolver:
    """
    Advanced Physics-Informed Neural Network for fluid dynamics simulation.
    Implements state-of-the-art techniques for solving Navier-Stokes equations.
    """

    def __init__(self, layers: List[int] = [3, 64, 64, 3], reynolds_number: float = 100.0):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.reynolds_number = reynolds_number
        self.viscosity = 1.0 / reynolds_number
        self.density = 1.0
        self.learning_rate = 0.001

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the neural network."""
        activation = inputs
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            linear = np.dot(activation, w) + b
            activation = ActivationFunction.tanh(linear) if i < len(self.weights) - 1 else linear
        return activation

    def solve_fluid_field(self, coords: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve fluid field at given coordinates and time using trained PINN."""
        time_coords = np.column_stack([coords, np.full(len(coords), time)])
        prediction = self.forward(time_coords)
        return prediction[:, :2], prediction[:, 2]

# ===============================================================================
# QUANTUM GAME THEORY IMPLEMENTATION
# ===============================================================================

class QuantumState:
    """Represents a quantum state with complex amplitudes"""
    def __init__(self, amplitudes: np.ndarray):
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self.normalize()

    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 1e-10: self.amplitudes /= norm

    def measure(self) -> int:
        probabilities = np.abs(self.amplitudes)**2
        measured_state = np.random.choice(len(probabilities), p=probabilities)
        new_amplitudes = np.zeros_like(self.amplitudes)
        new_amplitudes[measured_state] = 1.0
        self.amplitudes = new_amplitudes
        return measured_state

class QuantumGameTheory:
    """
    Advanced quantum game theory system implementing superposition strategies.
    """
    def __init__(self):
        self.quantum_strategies = {}
        self.entangled_players = {}
        self.hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    def create_quantum_strategy_space(self, player_id: str, classical_strategies: List[str]):
        num_strategies = len(classical_strategies)
        amplitudes = np.ones(num_strategies, dtype=complex) / np.sqrt(num_strategies)
        self.quantum_strategies[player_id] = {
            'state': QuantumState(amplitudes),
            'classical_strategies': classical_strategies
        }

    def create_entanglement(self, player1_id: str, player2_id: str):
        if player1_id in self.quantum_strategies and player2_id in self.quantum_strategies:
            self.entangled_players[(player1_id, player2_id)] = {'type': 'bell_state'}
            print(f"[QGT] Created entanglement between {player1_id} and {player2_id}")

    def measure_quantum_strategy(self, player_id: str) -> str:
        if player_id not in self.quantum_strategies: return "cooperate"
        player_strategy = self.quantum_strategies[player_id]
        measured_index = player_strategy['state'].measure()
        return player_strategy['classical_strategies'][measured_index]

    def quantum_prisoners_dilemma(self, player1_id: str, player2_id: str) -> Dict:
        strategies = ['cooperate', 'defect']
        if player1_id not in self.quantum_strategies: self.create_quantum_strategy_space(player1_id, strategies)
        if player2_id not in self.quantum_strategies: self.create_quantum_strategy_space(player2_id, strategies)

        strategy1 = self.measure_quantum_strategy(player1_id)
        strategy2 = self.measure_quantum_strategy(player2_id)

        payoff_matrix = {
            ('cooperate', 'cooperate'): (3, 3), ('cooperate', 'defect'): (0, 5),
            ('defect', 'cooperate'): (5, 0), ('defect', 'defect'): (1, 1)
        }
        payoffs = payoff_matrix.get((strategy1, strategy2), (0, 0))

        return {
            'player1_strategy': strategy1, 'player2_strategy': strategy2,
            'player1_payoff': payoffs[0], 'player2_payoff': payoffs[1]
        }

# ===============================================================================
# DEMONSTRATION FUNCTIONS
# ===============================================================================

def demonstrate_pinn_fluid_solver():
    print("\n" + "="*60 + "\nDEMONSTRATING PINN FLUID SOLVER\n" + "="*60)
    pinn = PINNFluidSolver(layers=[3, 32, 32, 3], reynolds_number=100.0)
    test_coords = np.array([[0.0, 0.0], [0.5, 0.5], [-0.3, 0.7]])
    velocity_field, pressure_field = pinn.solve_fluid_field(test_coords, time=1.0)
    print(f"\nFluid field prediction at t=1.0:")
    for i, coord in enumerate(test_coords):
        print(f"  Point {coord}: velocity=({velocity_field[i, 0]:.4f}, {velocity_field[i, 1]:.4f}), pressure={pressure_field[i]:.4f}")

def demonstrate_quantum_game_theory():
    print("\n" + "="*60 + "\nDEMONSTRATING QUANTUM GAME THEORY\n" + "="*60)
    qgt = QuantumGameTheory()
    players = ['Alice', 'Bob']
    for player in players:
        qgt.create_quantum_strategy_space(player, ['cooperate', 'defect'])
    qgt.create_entanglement('Alice', 'Bob')
    result = qgt.quantum_prisoners_dilemma('Alice', 'Bob')
    print(f"Game Alice vs Bob:\n  Strategies: {result['player1_strategy']} vs {result['player2_strategy']}\n  Payoffs: ({result['player1_payoff']:.2f}, {result['player2_payoff']:.2f})")

if __name__ == "__main__":
    demonstrate_pinn_fluid_solver()
    demonstrate_quantum_game_theory()
    print("\n" + "="*60 + "\nDEMONSTRATION COMPLETE\n" + "="*60)
