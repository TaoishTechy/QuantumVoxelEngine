# accessibility_utils.py
# Tools for ensuring the game meets WCAG 2.1 accessibility standards.

import numpy as np
from typing import Tuple
from logger import logger

def get_luminance(rgb: Tuple[int, int, int]) -> float:
    """Calculates the relative luminance of an RGB color (WCAG 1.4.3)."""
    srgb = [val / 255.0 for val in rgb]
    linear_rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * linear_rgb[0] + 0.7152 * linear_rgb[1] + 0.0722 * linear_rgb[2]

def get_contrast_ratio(lum1: float, lum2: float) -> float:
    """Calculates the contrast ratio between two luminance values."""
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)

class ThemeManager:
    """Manages UI themes and ensures they meet accessibility contrast requirements."""
    WCAG_AA_NORMAL_TEXT = 4.5

    def __init__(self, theme_palette: dict):
        self.palette = theme_palette
        self.validate_palette()

    def validate_palette(self) -> None:
        """Checks the default theme palette for contrast issues."""
        bg_lum = get_luminance(self.palette['background'])
        fg_lum = get_luminance(self.palette['foreground'])
        ratio = get_contrast_ratio(bg_lum, fg_lum)
        if ratio < self.WCAG_AA_NORMAL_TEXT:
            logger.warning(f"Default theme has low contrast ratio of {ratio:.2f}. Required: {self.WCAG_AA_NORMAL_TEXT}")
        else:
            logger.info(f"Default theme contrast ratio is OK ({ratio:.2f}).")

    def get_accessible_foreground(self, background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Dynamically selects the best foreground color for a given background."""
        # TODO: Implement ARIA live regions for announcing state changes.
        # TODO: Add icon support to provide non-color cues for UI states.
        bg_lum = get_luminance(background_color)
        best_color = self.palette['foreground']
        max_ratio = 0
        
        for color_name, color_value in self.palette.items():
            if color_name == 'background': continue
            fg_lum = get_luminance(color_value)
            ratio = get_contrast_ratio(bg_lum, fg_lum)
            if ratio > max_ratio:
                max_ratio = ratio
                best_color = color_value
        
        return best_color
