# gameplay_systems.py
# Manages interactive gameplay mechanics like inventory and puzzles.

import random
from typing import List, Dict, Any, Tuple

import core_world
from logger import logger

class SchrodingerItem:
    """An item that exists in a superposition of multiple potential states."""
    def __init__(self, item_id: str, possible_states: List[Dict[str, Any]]):
        self.item_id = item_id
        self.possible_states = possible_states
        self.is_collapsed = False
        self.final_state: Dict[str, Any] = {}

    def observe(self) -> Dict[str, Any]:
        """Collapses the item's wave function into a single, definite state."""
        if not self.is_collapsed:
            logger.info(f"Observing Schrödinger's Item '{self.item_id}'. Collapsing state...")
            # In a real game, probabilities could be weighted
            self.final_state = random.choice(self.possible_states)
            self.is_collapsed = True
            logger.info(f"Item collapsed to state: {self.final_state}")
        return self.final_state

class InventoryManager:
    """Manages the player's inventory, including quantum items."""
    def __init__(self):
        self.items: Dict[str, SchrodingerItem] = {}

    def add_item(self, item: SchrodingerItem):
        """Adds a new Schrödinger's Item to the inventory."""
        logger.info(f"Adding item '{item.item_id}' to inventory in a superposed state.")
        self.items[item.item_id] = item

    def get_item_state(self, item_id: str) -> Dict[str, Any]:
        """Gets the state of an item, observing it if necessary."""
        if item_id in self.items:
            return self.items[item_id].observe()
        return {}

class ObserverEffectManager:
    """Manages puzzles and environmental objects that depend on player observation."""
    def __init__(self, world: 'core_world.WorldState'):
        self.world = world
        self.observer_puzzles: List[Dict[str, Any]] = []

    def update(self, player_camera_direction: 'np.ndarray', player_pos: 'np.ndarray'):
        """
        Checks which objects are in the player's view and updates their state.
        """
        # TODO: Implement a proper frustum/visibility check.
        # This is a placeholder for the core logic.
        for puzzle in self.observer_puzzles:
            # Example: A bridge that only exists when looked at.
            is_in_view = self.is_position_in_view(puzzle['position'], player_pos, player_camera_direction)
            
            if is_in_view and not puzzle['is_active']:
                logger.info(f"Observer puzzle at {puzzle['position']} is now in view. Activating.")
                # self.world.set_block(...) to create the bridge
                puzzle['is_active'] = True
            elif not is_in_view and puzzle['is_active']:
                logger.info(f"Observer puzzle at {puzzle['position']} is no longer in view. Deactivating.")
                # self.world.set_block(...) to remove the bridge
                puzzle['is_active'] = False

    def is_position_in_view(self, pos, player_pos, player_dir) -> bool:
        # Simplified check: is the object roughly in front of the player?
        to_object = np.array(pos) - player_pos
        to_object /= np.linalg.norm(to_object)
        return np.dot(player_dir, to_object) > 0.5 # Within about a 60-degree cone
