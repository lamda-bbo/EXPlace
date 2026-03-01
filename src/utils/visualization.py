"""
Visualization utilities for macro placement.
All drawing uses real coordinates via utils.coord_utils.grid_to_real.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

from utils.coord_utils import grid_to_real


def _grid_rect_to_real(x, y, size_x, size_y, ratio_x, ratio_y):
    """Convert rectangle from grid (x, y, w, h) to real (rx, ry, rw, rh)."""
    rx, ry = grid_to_real(x, y, ratio_x, ratio_y)
    rx2, ry2 = grid_to_real(x + size_x, y + size_y, ratio_x, ratio_y)
    return rx, ry, rx2 - rx, ry2 - ry


def visualize_macro_clusters(macro_clusters, macro_pos, out_path=None):
    """
    Visualize macros with different colors per cluster.

    Args:
        macro_clusters: List[List[node_id]]
        macro_pos: dict[node_id] -> (x, y, size_x, size_y)
        out_path: Optional path to save the figure. If None, will not save.
    """
    if macro_clusters is None or len(macro_clusters) == 0:
        return

    # Determine canvas extents from macro positions
    min_x = min((macro_pos[m][0] for m in macro_pos), default=0)
    min_y = min((macro_pos[m][1] for m in macro_pos), default=0)
    max_x = max((macro_pos[m][0] + macro_pos[m][2] for m in macro_pos), default=0)
    max_y = max((macro_pos[m][1] + macro_pos[m][3] for m in macro_pos), default=0)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    # Set figure size maintaining aspect ratio
    aspect = width / height if height > 0 else 1.0
    if aspect >= 1:
        fig_w, fig_h = 10, 10 / max(aspect, 1e-6)
    else:
        fig_h, fig_w = 10, 10 * aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    cmap = plt.get_cmap('tab20')

    # Draw macros colored by cluster
    for idx, cluster in enumerate(macro_clusters):
        color = cmap(idx % 20)
        for m in cluster:
            if m not in macro_pos:
                continue
            x, y, sx, sy = macro_pos[m]
            rect = Rectangle((x, y), sx, sy, linewidth=1, edgecolor=color, facecolor=color, alpha=0.6)
            ax.add_patch(rect)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.set_title('Macro Clusters', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    if out_path is not None:
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
    plt.close()


def visualize_pin_blocking_rectangles(pin_rectangles, grid_size, ratio_x, ratio_y, port_pos=None, out_path=None, title='Pin Blocking Rectangles'):
    """
    Visualize pin blocking rectangles with optional port locations.
    Coordinates are converted from grid to real via grid_to_real(., ratio_x, ratio_y).

    Args:
        pin_rectangles: List[Tuple[int, int, int, int]], each as (x, y, w, h) in grid units
        grid_size: Size of the square canvas ([0, grid_size))
        ratio_x, ratio_y: real length per grid cell (for grid_to_real)
        port_pos: Optional[List[Tuple[int, int]]], port coordinates (x, y) in grid units
        out_path: Optional[str], if provided, save the figure to this path
        title: Title of the figure
    """
    if pin_rectangles is None:
        pin_rectangles = []

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    canvas_x0, canvas_y0 = grid_to_real(0, 0, ratio_x, ratio_y)
    canvas_x1, canvas_y1 = grid_to_real(grid_size, grid_size, ratio_x, ratio_y)
    canvas_w = canvas_x1 - canvas_x0
    canvas_h = canvas_y1 - canvas_y0

    # Draw chip/grid boundary
    border = Rectangle((0, 0), canvas_w, canvas_h, fill=False, edgecolor='black', linewidth=1.2)
    ax.add_patch(border)

    # Draw blocking rectangles
    for (x, y, w, h) in pin_rectangles:
        if w <= 0 or h <= 0:
            continue
        rx, ry = grid_to_real(x, y, ratio_x, ratio_y)
        rx2, ry2 = grid_to_real(x + w, y + h, ratio_x, ratio_y)
        rect = Rectangle((rx, ry), rx2 - rx, ry2 - ry, linewidth=0.8, edgecolor='red', facecolor='red', alpha=0.35)
        ax.add_patch(rect)

    # Optionally draw port points
    if port_pos is not None and len(port_pos) > 0:
        xs = [grid_to_real(p[0], p[1], ratio_x, ratio_y)[0] for p in port_pos]
        ys = [grid_to_real(p[0], p[1], ratio_x, ratio_y)[1] for p in port_pos]
        ax.scatter(xs, ys, s=12, c='blue', marker='o', alpha=0.85, label='ports')
        ax.legend(loc='upper right', fontsize=8)

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, canvas_h)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if out_path is not None:
        try:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
    plt.close()


def get_distinct_colors():
    """Get a list of distinct colors for cluster visualization."""
    return [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#ff9896',  # Light Red
        '#98df8a',  # Light Green
        '#ffbb78',  # Light Orange
        '#c5b0d5',  # Light Purple
        '#c49c94',  # Light Brown
        '#f7b6d3',  # Light Pink
        '#c7c7c7',  # Light Gray
        '#dbdb8d',  # Light Olive
        '#9edae5',  # Light Cyan
        '#ad494a'   # Dark Red
    ]


def visualize_prototype(env, macro_pos_prototype, macro_clusters, args, grid, ratio_x, ratio_y,
                       dataflow_mat=None, id2index=None, port_pos=None, pin_blocking_rectangles=None):
    """
    Visualize macro prototype placement with clusters.
    Uses real coordinates via grid_to_real(., ratio_x, ratio_y).
    """
    num_clusters = len(macro_clusters)
    distinct_colors = get_distinct_colors()
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(num_clusters)]

    use_real = ratio_x is not None and ratio_y is not None
    if use_real:
        lim_x0, lim_y0 = grid_to_real(0, 0, ratio_x, ratio_y)
        lim_x1, lim_y1 = grid_to_real(grid, grid, ratio_x, ratio_y)
        aspect_ratio = (lim_x1 - lim_x0) / max(lim_y1 - lim_y0, 1e-9)
    else:
        aspect_ratio = ratio_x / ratio_y if ratio_y else 1.0

    if aspect_ratio > 1:
        fig_width = 12
        fig_height = 12 / aspect_ratio
    else:
        fig_height = 10
        fig_width = 10 * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    total_macros = len(macro_pos_prototype)

    for macro in macro_pos_prototype:
        x, y, size_x, size_y = macro_pos_prototype[macro]
        orig_x = x + args.halo
        orig_y = y + args.halo
        orig_size_x = size_x - 2 * args.halo
        orig_size_y = size_y - 2 * args.halo

        color = '#000000'
        for idx, cluster in enumerate(macro_clusters):
            if macro in cluster:
                color = colors[idx]
                break

        if use_real:
            hx, hy, hw, hh = _grid_rect_to_real(x, y, size_x, size_y, ratio_x, ratio_y)
            mx, my, mw, mh = _grid_rect_to_real(orig_x, orig_y, orig_size_x, orig_size_y, ratio_x, ratio_y)
            halo_rect = Rectangle((hx, hy), hw, hh, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((mx, my), mw, mh, linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
        else:
            halo_rect = Rectangle((x, y), size_x, size_y, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((orig_x, orig_y), orig_size_x, orig_size_y, linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(halo_rect)
        ax.add_patch(macro_rect)

    if dataflow_mat is not None and id2index is not None:
        macros = list(macro_pos_prototype.keys())
        max_width = 1.0
        min_width = 0.1
        flow_values = []
        for i in range(len(macros)):
            for j in range(i+1, len(macros)):
                macro_a, macro_b = macros[i], macros[j]
                if macro_a in id2index and macro_b in id2index:
                    flow_values.append(dataflow_mat[id2index[macro_a], id2index[macro_b]])
        if flow_values:
            threshold = np.percentile(flow_values, 95)
            max_flow = np.max(flow_values) if np.max(flow_values) > 0 else 1.0
            for i in range(len(macros)):
                for j in range(i+1, len(macros)):
                    macro_a, macro_b = macros[i], macros[j]
                    if macro_a not in id2index or macro_b not in id2index:
                        continue
                    flow = dataflow_mat[id2index[macro_a], id2index[macro_b]]
                    if flow <= threshold:
                        continue
                    linewidth = min_width + (max_width - min_width) * (flow / max_flow)
                    x1, y1, sx1, sy1 = macro_pos_prototype[macro_a]
                    x2, y2, sx2, sy2 = macro_pos_prototype[macro_b]
                    if use_real:
                        cx1, cy1 = grid_to_real(x1 + sx1 / 2.0, y1 + sy1 / 2.0, ratio_x, ratio_y)
                        cx2, cy2 = grid_to_real(x2 + sx2 / 2.0, y2 + sy2 / 2.0, ratio_x, ratio_y)
                    else:
                        cx1, cy1 = x1 + sx1 / 2.0, y1 + sy1 / 2.0
                        cx2, cy2 = x2 + sx2 / 2.0, y2 + sy2 / 2.0
                    ax.plot([cx1, cx2], [cy1, cy2], color='#1f4e79', linewidth=linewidth, alpha=0.7)

    if "port" in args.used_masks and port_pos is not None and len(port_pos) > 0:
        for port_pos_item in port_pos:
            px, py = port_pos_item[0], port_pos_item[1]
            if use_real:
                rx, ry = grid_to_real(px, py, ratio_x, ratio_y)
                port_circle = Circle((rx, ry), radius=min(2.0, (lim_x1 - lim_x0) / 200), color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            else:
                port_circle = Circle((px, py), radius=2.0, color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            ax.add_patch(port_circle)

    if pin_blocking_rectangles is not None and len(pin_blocking_rectangles) > 0:
        for rect_data in pin_blocking_rectangles:
            x, y, w, h = rect_data
            if w <= 0 or h <= 0:
                continue
            if use_real:
                rx, ry, rw, rh = _grid_rect_to_real(x, y, w, h, ratio_x, ratio_y)
                block_rect = Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            else:
                block_rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(block_rect)

    if use_real:
        ax.set_xlim(lim_x0, lim_x1)
        ax.set_ylim(lim_y0, lim_y1)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)
        ax.set_aspect(ratio_y / ratio_x if ratio_y else 1.0)
    ax.set_title(f'Macro Prototype Placement (Total: {total_macros} macros)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    vis_dir = getattr(args, 'visualization_dir', args.log_dir)
    plt.savefig(os.path.join(vis_dir, "macro_prototype.jpg"), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_placement(env, macro_pos, macro_clusters, args, grid, ratio_x, ratio_y, i_episode,
                        test_mode=False, path=None, dataflow_mat=None, id2index=None,
                        port_pos=None, pin_blocking_rectangles=None):
    """
    Visualize macro placement with clusters.
    Uses real coordinates via grid_to_real(., ratio_x, ratio_y).
    """
    num_clusters = len(macro_clusters)
    distinct_colors = get_distinct_colors()
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(num_clusters)]

    use_real = ratio_x is not None and ratio_y is not None
    if use_real:
        lim_x0, lim_y0 = grid_to_real(0, 0, ratio_x, ratio_y)
        lim_x1, lim_y1 = grid_to_real(grid, grid, ratio_x, ratio_y)
        aspect_ratio = (lim_x1 - lim_x0) / max(lim_y1 - lim_y0, 1e-9)
    else:
        aspect_ratio = ratio_x / ratio_y if ratio_y else 1.0

    if aspect_ratio > 1:
        fig_width = 12
        fig_height = 12 / aspect_ratio
    else:
        fig_height = 10
        fig_width = 10 * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    for macro in macro_pos:
        x, y, size_x, size_y = macro_pos[macro]
        orig_x = x + args.halo
        orig_y = y + args.halo
        orig_size_x = size_x - 2 * args.halo
        orig_size_y = size_y - 2 * args.halo

        color = '#000000'
        for idx, cluster in enumerate(macro_clusters):
            if macro in cluster:
                color = colors[idx]
                break

        if use_real:
            hx, hy, hw, hh = _grid_rect_to_real(x, y, size_x, size_y, ratio_x, ratio_y)
            mx, my, mw, mh = _grid_rect_to_real(orig_x, orig_y, orig_size_x, orig_size_y, ratio_x, ratio_y)
            halo_rect = Rectangle((hx, hy), hw, hh, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((mx, my), mw, mh, linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
        else:
            halo_rect = Rectangle((x, y), size_x, size_y, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((orig_x, orig_y), orig_size_x, orig_size_y, linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(halo_rect)
        ax.add_patch(macro_rect)

    if dataflow_mat is not None and id2index is not None:
        macros = list(macro_pos.keys())
        max_width, min_width = 1.0, 0.1
        flow_values = [dataflow_mat[id2index[macro_a], id2index[macro_b]]
                       for i in range(len(macros)) for j in range(i+1, len(macros))
                       for macro_a, macro_b in [(macros[i], macros[j])]
                       if macro_a in id2index and macro_b in id2index]
        if flow_values:
            threshold = np.percentile(flow_values, 95)
            max_flow = np.max(flow_values) if np.max(flow_values) > 0 else 1.0
            for i in range(len(macros)):
                for j in range(i+1, len(macros)):
                    macro_a, macro_b = macros[i], macros[j]
                    if macro_a not in id2index or macro_b not in id2index:
                        continue
                    flow = dataflow_mat[id2index[macro_a], id2index[macro_b]]
                    if flow <= threshold:
                        continue
                    linewidth = min_width + (max_width - min_width) * (flow / max_flow)
                    x1, y1, sx1, sy1 = macro_pos[macro_a]
                    x2, y2, sx2, sy2 = macro_pos[macro_b]
                    if use_real:
                        cx1, cy1 = grid_to_real(x1 + sx1 / 2.0, y1 + sy1 / 2.0, ratio_x, ratio_y)
                        cx2, cy2 = grid_to_real(x2 + sx2 / 2.0, y2 + sy2 / 2.0, ratio_x, ratio_y)
                    else:
                        cx1, cy1 = x1 + sx1 / 2.0, y1 + sy1 / 2.0
                        cx2, cy2 = x2 + sx2 / 2.0, y2 + sy2 / 2.0
                    ax.plot([cx1, cx2], [cy1, cy2], color='#1f4e79', linewidth=linewidth, alpha=0.7)

    if "port" in args.used_masks and port_pos is not None and len(port_pos) > 0:
        for port_pos_item in port_pos:
            px, py = port_pos_item[0], port_pos_item[1]
            if use_real:
                rx, ry = grid_to_real(px, py, ratio_x, ratio_y)
                port_circle = Circle((rx, ry), radius=min(2.0, (lim_x1 - lim_x0) / 200), color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            else:
                port_circle = Circle((px, py), radius=2.0, color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            ax.add_patch(port_circle)

    if pin_blocking_rectangles is not None and len(pin_blocking_rectangles) > 0:
        for rect_data in pin_blocking_rectangles:
            x, y, w, h = rect_data
            if w <= 0 or h <= 0:
                continue
            if use_real:
                rx, ry, rw, rh = _grid_rect_to_real(x, y, w, h, ratio_x, ratio_y)
                block_rect = Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            else:
                block_rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(block_rect)

    if use_real:
        ax.set_xlim(lim_x0, lim_x1)
        ax.set_ylim(lim_y0, lim_y1)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)
        ax.set_aspect(ratio_y / ratio_x)

    plt.tight_layout()
    vis_dir = getattr(args, 'visualization_dir', args.log_dir)
    if path is None:
        if test_mode:
            plt.savefig(os.path.join(vis_dir, "macro_{}_test.jpg".format(i_episode)), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(vis_dir, "macro_{}_train.jpg".format(i_episode)), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_step(env, macro_placed, macro_pos, macro_clusters, macro_to_place, place_idx,
                   args, grid, ratio_x, ratio_y, step_idx, corners=None, current_macro=None,
                   action=None, port_pos=None, pin_blocking_rectangles=None):
    """
    Visualize candidate corners and placed macros for each step.
    Uses real coordinates via grid_to_real(., ratio_x, ratio_y).
    """
    num_clusters = len(macro_clusters)
    distinct_colors = get_distinct_colors()
    colors = [distinct_colors[i % len(distinct_colors)] for i in range(num_clusters)]

    use_real = ratio_x is not None and ratio_y is not None
    if use_real:
        lim_x0, lim_y0 = grid_to_real(0, 0, ratio_x, ratio_y)
        lim_x1, lim_y1 = grid_to_real(grid, grid, ratio_x, ratio_y)
        aspect_ratio = (lim_x1 - lim_x0) / max(lim_y1 - lim_y0, 1e-9)
        radius_scale = (lim_x1 - lim_x0) / max(grid, 1) * 0.5
    else:
        aspect_ratio = ratio_x / ratio_y
        radius_scale = 0.5

    if aspect_ratio > 1:
        fig_width = 16
        fig_height = 16 / aspect_ratio
    else:
        fig_height = 14
        fig_width = 14 * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    for macro in macro_placed:
        x, y, size_x, size_y = macro_pos[macro]
        orig_x = x + args.halo
        orig_y = y + args.halo
        orig_size_x = size_x - 2 * args.halo
        orig_size_y = size_y - 2 * args.halo

        color = '#000000'
        for idx, cluster in enumerate(macro_clusters):
            if macro in cluster:
                color = colors[idx]
                break

        if use_real:
            hx, hy, hw, hh = _grid_rect_to_real(x, y, size_x, size_y, ratio_x, ratio_y)
            mx, my, mw, mh = _grid_rect_to_real(orig_x, orig_y, orig_size_x, orig_size_y, ratio_x, ratio_y)
            halo_rect = Rectangle((hx, hy), hw, hh, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((mx, my), mw, mh, linewidth=2, edgecolor=color, facecolor=color, alpha=0.7)
            tx, ty = mx + mw / 2, my + mh / 2
        else:
            halo_rect = Rectangle((x, y), size_x, size_y, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
            macro_rect = Rectangle((orig_x, orig_y), orig_size_x, orig_size_y, linewidth=2, edgecolor=color, facecolor=color, alpha=0.7)
            tx, ty = orig_x + orig_size_x / 2, orig_y + orig_size_y / 2
        ax.add_patch(halo_rect)
        ax.add_patch(macro_rect)
        ax.text(tx, ty, macro, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    if current_macro is not None and place_idx < len(macro_to_place) and corners is not None:
        _, _, size_x, size_y = macro_pos[current_macro]
        for i, (corner_x, corner_y) in enumerate(corners):
            if use_real:
                cx, cy = grid_to_real(corner_x, corner_y, ratio_x, ratio_y)
                px, py, pw, ph = _grid_rect_to_real(corner_x, corner_y, size_x, size_y, ratio_x, ratio_y)
                corner_circle = Circle((cx, cy), radius=radius_scale, color='#e74c3c', edgecolor='#c0392b', linewidth=2, alpha=0.8)
                preview_rect = Rectangle((px, py), pw, ph, linewidth=2, edgecolor='#e74c3c', facecolor='none', linestyle='--', alpha=0.6)
                text_y = py + ph + radius_scale
            else:
                cx, cy = corner_x, corner_y
                corner_circle = Circle((cx, cy), radius=0.5, color='#e74c3c', edgecolor='#c0392b', linewidth=2, alpha=0.8)
                preview_rect = Rectangle((corner_x, corner_y), size_x, size_y, linewidth=2, edgecolor='#e74c3c', facecolor='none', linestyle='--', alpha=0.6)
                text_y = corner_y + size_y + 0.5
            ax.add_patch(corner_circle)
            ax.add_patch(preview_rect)
            ax.text(cx, text_y, f'C{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')

            if action is not None and abs(corner_x - action[0]) < 0.1 and abs(corner_y - action[1]) < 0.1:
                if use_real:
                    sel_radius = radius_scale * 1.6
                    selected_circle = Circle((cx, cy), radius=sel_radius, color='#27ae60', edgecolor='#229954', linewidth=3, alpha=0.9)
                    selected_rect = Rectangle((px, py), pw, ph, linewidth=3, edgecolor='#27ae60', facecolor='#27ae60', alpha=0.3)
                    ax.add_patch(selected_circle)
                    ax.add_patch(selected_rect)
                    ax.text(px + pw / 2, py + ph / 2, f'SELECTED\n{current_macro}', ha='center', va='center', fontsize=10, fontweight='bold', color='#229954', bbox=dict(boxstyle="round,pad=0.3", facecolor='#d5f4e6', alpha=0.8))
                else:
                    selected_circle = Circle((corner_x, corner_y), radius=0.8, color='#27ae60', edgecolor='#229954', linewidth=3, alpha=0.9)
                    selected_rect = Rectangle((corner_x, corner_y), size_x, size_y, linewidth=3, edgecolor='#27ae60', facecolor='#27ae60', alpha=0.3)
                    ax.add_patch(selected_circle)
                    ax.add_patch(selected_rect)
                    ax.text(corner_x + size_x / 2, corner_y + size_y / 2, f'SELECTED\n{current_macro}', ha='center', va='center', fontsize=10, fontweight='bold', color='#229954', bbox=dict(boxstyle="round,pad=0.3", facecolor='#d5f4e6', alpha=0.8))

    if port_pos is not None and len(port_pos) > 0:
        for port_pos_item in port_pos:
            px, py = port_pos_item[0], port_pos_item[1]
            if use_real:
                rx, ry = grid_to_real(px, py, ratio_x, ratio_y)
                port_circle = Circle((rx, ry), radius=min(2.0, (lim_x1 - lim_x0) / 200), color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            else:
                port_circle = Circle((px, py), radius=2.0, color='orange', edgecolor='darkorange', linewidth=1.5, alpha=0.8)
            ax.add_patch(port_circle)

    if pin_blocking_rectangles is not None and len(pin_blocking_rectangles) > 0:
        for rect_data in pin_blocking_rectangles:
            x, y, w, h = rect_data
            if w <= 0 or h <= 0:
                continue
            if use_real:
                rx, ry, rw, rh = _grid_rect_to_real(x, y, w, h, ratio_x, ratio_y)
                block_rect = Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            else:
                block_rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(block_rect)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    if use_real:
        ax.set_xlim(lim_x0, lim_x1)
        ax.set_ylim(lim_y0, lim_y1)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)
        ax.set_aspect(ratio_y / ratio_x)
    
    title_parts = [f'Step {step_idx}']
    if current_macro is not None:
        title_parts.append(f'Current Macro: {current_macro}')
    if len(macro_placed) > 0:
        title_parts.append(f'Placed: {len(macro_placed)}/{len(macro_to_place)}')
    if current_macro is not None and corners is not None and len(corners) > 0:
        title_parts.append(f'Candidates: {len(corners)}')
    
    ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
    
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#1f4e79', alpha=0.7, label='Placed Macros'),
        plt.Circle((0, 0), 0.5, facecolor='#e74c3c', alpha=0.8, label='Candidate Corners'),
        plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#e74c3c', linestyle='--', label='Macro Preview'),
    ]
    if action is not None:
        legend_elements.append(plt.Circle((0, 0), 0.5, facecolor='#27ae60', alpha=0.9, label='Selected Position'))
    if port_pos is not None and len(port_pos) > 0:
        legend_elements.append(plt.Circle((0, 0), 0.5, facecolor='orange', alpha=0.8, label='Ports'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    vis_dir = getattr(args, 'visualization_dir', args.log_dir)
    steps_dir = os.path.join(vis_dir, "visualization_steps")
    os.makedirs(steps_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(steps_dir, f"step_{step_idx:03d}.jpg"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Step {step_idx} visualization saved: {len(macro_placed)} placed, {len(corners) if current_macro is not None and corners is not None else 0} candidates")


def plot_placement(dmp_placer, dmp_params, pos, figure_name):
    """
    Plot placement using DREAMPlace placer and flip the image.
    
    Args:
        dmp_placer: DREAMPlace placer instance
        dmp_params: DREAMPlace parameters
        pos: placement positions
        figure_name: output file path
    """
    dmp_placer.plot(
        dmp_params,
        None,
        None,
        pos,
        figure_name, 
    )

    img = Image.open(figure_name)
    out = img.transpose(Image.FLIP_TOP_BOTTOM)
    img.close()
    out.save(figure_name)
