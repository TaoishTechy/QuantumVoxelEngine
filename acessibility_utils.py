# accessibility_utils.py
# Tools for ensuring the game meets WCAG 2.1 accessibility standards.

import numpy as np
from typing import Tuple, Dict
from logger import logger

def get_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Calculates the relative luminance of an RGB color (WCAG 1.4.3).

    Args:
        rgb: A tuple of (r, g, b) values from 0-255.

    Returns:
        The relative luminance (0.0 to 1.0).
    """
    srgb = [val / 255.0 for val in rgb]
    linear_rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return 0.2126 * linear_rgb[0] + 0.7152 * linear_rgb[1] + 0.0722 * linear_rgb[2]

def get_contrast_ratio(lum1: float, lum2: float) -> float:
    """
    Calculates the contrast ratio between two luminance values.

    Args:
        lum1: Luminance of the first color.
        lum2: Luminance of the second color.

    Returns:
        The contrast ratio.
    """
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)

def get_text_shadow_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    Determines an appropriate text shadow/outline color for a given background
    to ensure legibility (addresses text-on-image issues).

    Args:
        background_color: The average RGB color behind the text.

    Returns:
        An RGBA tuple for a semi-transparent black or white shadow.
    """
    bg_lum = get_luminance(background_color)
    # If the background is light, use a dark shadow. If dark, use a light shadow.
    if bg_lum > 0.5:
        return (0, 0, 0, 128)  # Semi-transparent black
    else:
        return (255, 255, 255, 128) # Semi-transparent white

class ThemeManager:
    """Manages UI themes and ensures they meet accessibility contrast requirements."""
    WCAG_AA_NORMAL_TEXT: float = 4.5
    WCAG_AA_LARGE_TEXT: float = 3.0

    def __init__(self, theme_palette: Dict[str, Tuple[int, int, int]]):
        """
        Initializes the ThemeManager with a given color palette.

        Args:
            theme_palette: A dictionary containing color names and RGB values.
                           Must include 'background' and 'foreground' keys.
        """
        self.palette = theme_palette
        self.validate_palette()

    def validate_palette(self) -> None:
        """Checks the default theme palette for contrast issues."""
        try:
            bg_lum = get_luminance(self.palette['background'])
            fg_lum = get_luminance(self.palette['foreground'])
            ratio = get_contrast_ratio(bg_lum, fg_lum)
            
            if ratio < self.WCAG_AA_NORMAL_TEXT:
                logger.warning(f"Default theme has low contrast ratio of {ratio:.2f}. "
                               f"Required for normal text: {self.WCAG_AA_NORMAL_TEXT}")
            else:
                logger.info(f"Default theme contrast ratio is OK ({ratio:.2f}).")
        except KeyError as e:
            logger.error(f"Theme palette is missing a required key: {e}")

    def get_accessible_foreground(self, background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Dynamically selects the best foreground color from the palette for a given background.
        This ensures that text remains readable even on varied background colors.

        Args:
            background_color: The RGB tuple of the background color.

        Returns:
            The RGB tuple of the foreground color with the highest contrast.
        """
        # TODO: Implement ARIA live regions for announcing state changes in the UI.
        # TODO: Add icon support to provide non-color cues for UI states (e.g., success/error icons).
        bg_lum = get_luminance(background_color)
        best_color = self.palette.get('foreground', (255, 255, 255))
        max_ratio = 0

        for color_name, color_value in self.palette.items():
            if color_name == 'background': continue
            
            fg_lum = get_luminance(color_value)
            ratio = get_contrast_ratio(bg_lum, fg_lum)
            
            if ratio > max_ratio:
                max_ratio = ratio
                best_color = color_value
        
        if max_ratio < self.WCAG_AA_NORMAL_TEXT:
            logger.warning(f"No color in the palette provides sufficient contrast against the given background. "
                           f"Max ratio found: {max_ratio:.2f}")

        return best_color
