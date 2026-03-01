"""
Unified grid <-> real coordinate conversion for placement and visualization.

Grid coordinates: discrete indices (with optional halo).
Real coordinates: site/row-aligned positions in placement units (site_width, row_height).
"""
import math


def grid_to_real(grid_x, grid_y, ratio_x, ratio_y):
    nat_x = round(grid_x * ratio_x + ratio_x)
    nat_y = round(grid_y * ratio_y + ratio_y)
    return nat_x, nat_y


def real_to_grid(raw_x, raw_y, ratio_x, ratio_y):
    gx = math.floor(max(0, (raw_x - ratio_x) / ratio_x))
    gy = math.floor(max(0, (raw_y - ratio_y) / ratio_y))
    return gx, gy
