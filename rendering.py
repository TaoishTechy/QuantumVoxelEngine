# rendering.py
# Contains all OpenGL rendering logic, including the Renderer, InputHandler,
# and VBO management classes.

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
from typing import Dict, Tuple

import config
import core_world
import gpu_backend

class ChunkVBO:
    # ... (ChunkVBO class remains the same)

class Renderer:
    # ... (Renderer class remains the same, but now reads from config.py)

class InputHandler:
    # ... (InputHandler class remains the same, but now reads from config.py)
