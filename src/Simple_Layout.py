import numpy as np
import pickle

height, width = 200, 500
traversable_mask = np.ones((height, width), dtype=bool)
obstacle_mask = np.zeros((height, width), dtype=bool)
red_boundary_mask = np.zeros((height, width), dtype=bool)
original_rgb = np.zeros((height, width, 3), dtype=np.uint8) + 255  # white background

# Source: entire leftmost wall (all rows at column 0)
boundary_source_points = [(y, 0) for y in range(height)]
entry_point = None  # Not used when the full wall is the source

layout = {
    'traversable_mask': traversable_mask,
    'obstacle_mask': obstacle_mask,
    'white_obstacle_mask': np.zeros_like(traversable_mask),
    'red_boundary_mask': red_boundary_mask,
    'height': height,
    'width': width,
    'entry_point': entry_point,
    'boundary_source_points': boundary_source_points,
    'original_rgb': original_rgb
}

with open('layouts/simple_layout.pkl', 'wb') as f:
    pickle.dump(layout, f)