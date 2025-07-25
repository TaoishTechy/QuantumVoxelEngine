# settings.py
# Loads and provides access to game settings from an external JSON file.

import json
from typing import Dict, Any

class Settings:
    """A singleton class to hold game settings."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance.load_settings()
        return cls._instance

    def load_settings(self) -> None:
        """Loads settings from settings.json."""
        try:
            with open('settings.json', 'r') as f:
                self._settings: Dict[str, Any] = json.load(f)
        except FileNotFoundError:
            print("WARNING: settings.json not found. Using default values.")
            self._settings = {
                "physics": {"gravity_multiplier": 1.0},
                "player": {"jump_force_multiplier": 1.0}
            }

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a setting value, navigating nested keys with dots."""
        keys = key.split('.')
        value = self._settings
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global instance for easy access
settings = Settings()
