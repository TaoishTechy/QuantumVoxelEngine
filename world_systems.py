# world_systems.py
# Manages dynamic, non-player-centric world systems like weather.

import random
from logger import logger
import config # Import centralized constants

class QuantumWeatherSystem:
    """
    A procedural weather system driven by quantum probability.
    FIX: Uses constants from config.py instead of magic numbers.
    """
    def __init__(self):
        self.current_state = "clear"
        self.superposition_storm_active = False
        self.storm_collapse_timer = 0.0

    def update(self, dt: float, player_is_outdoors: bool):
        """Updates the weather state."""
        if self.superposition_storm_active:
            self.storm_collapse_timer -= dt
            # The storm has a higher chance to collapse if the player is outdoors to observe it.
            if self.storm_collapse_timer <= 0 or (player_is_outdoors and random.random() < 0.05):
                self.collapse_storm()

        # Low chance to start a new superposition storm, based on a configurable constant.
        if not self.superposition_storm_active and random.random() < config.STORM_PROBABILITY_PER_TICK:
            self.start_superposition_storm()

    def start_superposition_storm(self):
        """Starts a weather event that is in a superposition of states."""
        logger.info("A superposition storm is forming! The weather is both sunny and rainy.")
        self.superposition_storm_active = True
        self.storm_collapse_timer = config.STORM_SUPERPOSITION_DURATION

    def collapse_storm(self):
        """Collapses the storm into a single, definite weather state."""
        final_state = random.choice(["clear", "rainy", "stormy"])
        logger.info(f"The superposition storm has collapsed into a '{final_state}' state.")
        self.current_state = final_state
        self.superposition_storm_active = False
