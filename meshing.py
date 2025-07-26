# meshing.py
# Contains the thread-safe worker function for building chunk VBO data.

import numpy as np
from typing import Tuple, List, Dict

# --- Module Imports (Assumed to exist) ---
import core_world
from config import Config

def worker_build_mesh(chunk_coord: Tuple[int, int], world: 'core_world.WorldState') -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    """
    (THREAD-SAFE) Constructs the mesh for a single chunk.
    This function is designed to be run in a separate thread.
    """
    wx, wz = chunk_coord
    chunk = world.get_or_create_chunk(wx, wz)
    vertices, indices, vertex_count = [], [], 0
    
    for x in range(core_world.Chunk.CHUNK_SIZE_X):
        for y in range(core_world.Chunk.CHUNK_HEIGHT):
            for z in range(core_world.Chunk.CHUNK_SIZE_Z):
                block_type = chunk.get_block(x, y, z)
                if block_type == core_world.BlockType.AIR: continue
                
                world_x, world_y, world_z = wx * 16 + x, y, wz * 16 + z
                tex_coords = Config.TEXTURE_MAP.get(block_type, Config.DEFAULT_TEXTURE_COORDS)
                
                # Check neighbors and add faces
                # Top Face (+Y)
                if world.get_block(world_x, y + 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y + 1, world_z + 1), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y + 1, world_z), (world_x, y + 1, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Bottom Face (-Y)
                if world.get_block(world_x, y - 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z), (world_x + 1, y, world_z), (world_x + 1, y, world_z + 1), (world_x, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Right Face (+X)
                if world.get_block(world_x + 1, y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, world_z + 1), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y + 1, world_z), (world_x + 1, y, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Left Face (-X)
                if world.get_block(world_x - 1, y, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z), (world_x, y + 1, world_z), (world_x, y + 1, world_z + 1), (world_x, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Front Face (+Z)
                if world.get_block(world_x, y, world_z + 1).type == core_world.BlockType.AIR:
                    v = [(world_x, y, world_z + 1), (world_x, y + 1, world_z + 1), (world_x + 1, y + 1, world_z + 1), (world_x + 1, y, world_z + 1)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4
                # Back Face (-Z)
                if world.get_block(world_x, y, world_z - 1).type == core_world.BlockType.AIR:
                    v = [(world_x + 1, y, world_z), (world_x + 1, y + 1, world_z), (world_x, y + 1, world_z), (world_x, y, world_z)]
                    new_v, new_i = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_v); indices.extend(new_i); vertex_count += 4

    return chunk_coord, np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_face_data(verts: List[Tuple[float, float, float]], tex_coords: Tuple[int, int], start_index: int) -> Tuple[List[float], List[int]]:
    """Creates vertex and index data for a single quad face."""
    uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]
    # Vertex format: [x, y, z, u, v, atlas_x, atlas_y]
    vertex_data = [(*verts[i], *uvs[i], *tex_coords) for i in range(4)]
    # Indices for two triangles forming a quad
    index_data = [start_index, start_index + 1, start_index + 2, start_index, start_index + 2, start_index + 3]
    return vertex_data, index_data
