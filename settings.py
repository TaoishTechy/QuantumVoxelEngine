# settings.py
# Loads and provides access to game settings from an external JSON file.

import json
from typing import Dict, Any
from logger import logger # Use the centralized logger for warnings

class Settings:
    """
    A singleton class to hold game settings, loaded from settings.json.
    Now with more robust error handling.
    """
    _instance = None
    _settings: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance.load_settings()
        return cls._instance

    def load_settings(self) -> None:
        """
        Loads settings from settings.json.
        FIX: Handles both FileNotFoundError and json.JSONDecodeError.
        """
        try:
            with open('settings.json', 'r') as f:
                self._settings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load or parse settings.json: {e}. Using default values.")
            self._settings = self._get_default_settings()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Provides fallback default settings."""
        return {
            "screen": {
                "width": 1280,
                "height": 720,
                "fullscreen": False
            },
            "player": {
                "mouse_sensitivity": 0.1,
                "jump_force": 5.0
            },
            "graphics": {
                "fov": 75,
                "max_fps": 144
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a setting value, navigating nested keys with dots (e.g., "player.jump_force").
        """
        keys = key.split('.')
        value = self._settings
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Setting '{key}' not found, returning default value: {default}.")
            return default

# Global instance for easy access throughout the application
settings = Settings()
