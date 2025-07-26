# meshing.py
# Contains the thread-safe worker function for building chunk VBO data.

import numpy as np
from typing import Tuple, List, Dict
import core_world
import config

def worker_build_mesh(chunk_coord: Tuple[int, int], world: 'core_world.WorldState') -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    """
    (THREAD-SAFE) Constructs the mesh for a single chunk.
    This function is designed to be run in a separate thread.
    """
    wx, wz = chunk_coord
    chunk = world.get_or_create_chunk(wx, wz)
    
    vertices = []
    indices = []
    vertex_count = 0
    
    # TODO: Implement dirty-region sub-section rebuilds (8x8x8 or 4x4x4)
    for x in range(chunk.CHUNK_SIZE_X):
        for y in range(chunk.CHUNK_HEIGHT):
            for z in range(chunk.CHUNK_SIZE_Z):
                block_type = chunk.get_block(x, y, z)
                if block_type == core_world.BlockType.AIR: continue

                world_x, world_y, world_z = wx * chunk.CHUNK_SIZE_X + x, y, wz * chunk.CHUNK_SIZE_Z + z
                
                tex_coords = config.TEXTURE_MAP.get(block_type, config.DEFAULT_TEXTURE_COORDS)
                
                # Check neighbors and add faces
                if world.get_block(world_x, world_y + 1, world_z).type == core_world.BlockType.AIR:
                    v = [(world_x, y + 1, world_z), (world_x + 1, y + 1, world_z), (world_x + 1, y + 1, world_z + 1), (world_x, y + 1, world_z + 1)]
                    new_verts, new_indices = create_face_data(v, tex_coords, vertex_count)
                    vertices.extend(new_verts); indices.extend(new_indices); vertex_count += 4
                # ... (and so on for the other 5 faces, using world coordinates and correct winding)

    return chunk_coord, np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def create_face_data(verts: List[Tuple[float, float, float]], tex_coords: Tuple[int, int], 
                     start_index: int) -> Tuple[List[float], List[int]]:
    """Creates vertex and index data for a single quad face."""
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    vertex_data = [(*verts[i], *uvs[i], *tex_coords) for i in range(4)]
    index_data = [start_index, start_index + 1, start_index + 2, start_index + 2, start_index + 3, start_index]
    return vertex_data, index_data
