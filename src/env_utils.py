import numpy as np
import ray
import pdb
import pickle
import json
import os
import torch as th
import math
from place_env.place_env import PlaceEnv
from problem_instance import ProblemInstance
from utils.visualization import visualize_macro_clusters, visualize_pin_blocking_rectangles



def merge_overlapping_rectangles(rectangles):
    """
    Merge overlapping rectangles and return the merged rectangle list.
    
    Args:
        rectangles: List of rectangles, each rectangle is (x, y, width, height)
        
    Returns:
        merged_rectangles: List of merged rectangles
    """
    if not rectangles:
        return []
    
    # Sort by x coordinate
    rectangles = sorted(rectangles, key=lambda r: (r[0], r[1]))
    merged = []
    
    for rect in rectangles:
        x, y, w, h = rect
        merged_rect = [x, y, w, h]
        
        # Check if can merge with already merged rectangles
        i = 0
        while i < len(merged):
            mx, my, mw, mh = merged[i]
            
            # Check if overlapping or adjacent
            if (x <= mx + mw and x + w >= mx and 
                y <= my + mh and y + h >= my):
                # Merge rectangles
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged_rect = [new_x, new_y, new_w, new_h]
                merged.pop(i)
            else:
                i += 1
        
        merged.append(merged_rect)
    
    return merged

def compute_pin_blocking_rectangles(port_pos, grid_size, ratio_x, ratio_y, block_width_raw=1000.0, block_height_raw=2000.0):
    """
    Compute and merge port blocking rectangles.
    
    Args:
        port_pos: List of port positions, each is [x, y] in grid units
        grid_size: Grid size
        ratio_x: original width per grid cell along X (orig_width / grid)
        ratio_y: original height per grid cell along Y (orig_height / grid)
        block_width_raw: width along edge in original units
        block_height_raw: height toward die center in original units
        
    Returns:
        merged_rectangles: List of merged blocking rectangles
    """
    if port_pos is None or len(port_pos) == 0:
        return []
    
    # Collect rectangles by edge category to avoid cross-edge merging
    rectangles_by_cat = {
        'left': [],
        'right': [],
        'bottom': [],
        'top': [],
        'internal': []
    }

    def clamp_rect(x, y, w, h, size):
        # Clamp rectangle to [0, size)
        x = max(0, x)
        y = max(0, y)
        w = max(0, min(w, size - x))
        h = max(0, min(h, size - y))
        return (x, y, w, h)

    # Convert original sizes (raw units) to grid cells along each axis
    # Use symmetric rounding to avoid systematic bias on specific edges
    edge_len_x = max(1, int(round(block_width_raw / max(ratio_x, 1e-9))))
    edge_len_y = max(1, int(round(block_width_raw / max(ratio_y, 1e-9))))
    inward_len_x = max(1, int(round(block_height_raw / max(ratio_x, 1e-9))))
    inward_len_y = max(1, int(round(block_height_raw / max(ratio_y, 1e-9))))

    bound_distance = 3
    for port_x, port_y in port_pos:
        # width = along edge; height = toward center
        if port_x <= bound_distance:  # Left edge: edge along Y, inward +X
            w = inward_len_x    # X dimension: inward height measured along X
            h = edge_len_y      # Y dimension: edge width measured along Y
            x = 0
            y = port_y - (h // 2)
            rect = clamp_rect(x, y, w, h, grid_size)
            if rect[2] > 0 and rect[3] > 0:
                rectangles_by_cat['left'].append(rect)
            continue
        elif port_x >= grid_size - bound_distance - 1:  # Right edge: edge along Y, inward -X
            w = inward_len_x
            h = edge_len_y
            x = port_x - w
            y = port_y - (h // 2)
            rect = clamp_rect(x, y, w, h, grid_size)
            if rect[2] > 0 and rect[3] > 0:
                rectangles_by_cat['right'].append(rect)
            continue
        elif port_y <= bound_distance:  # Bottom edge: edge along X, inward +Y
            w = edge_len_x
            h = inward_len_y
            x = port_x - (w // 2)
            y = 0
            rect = clamp_rect(x, y, w, h, grid_size)
            if rect[2] > 0 and rect[3] > 0:
                rectangles_by_cat['bottom'].append(rect)
            continue
        elif port_y >= grid_size - bound_distance - 1:  # Top edge: edge along X, inward -Y
            w = edge_len_x
            h = inward_len_y
            x = port_x - (w // 2)
            y = port_y - h
            rect = clamp_rect(x, y, w, h, grid_size)
            if rect[2] > 0 and rect[3] > 0:
                rectangles_by_cat['top'].append(rect)
            continue
    
    # Merge rectangles within each category only (no cross-edge merging)
    merged_rectangles = []
    for cat in ['left', 'right', 'bottom', 'top']:
        if len(rectangles_by_cat[cat]) > 0:
            merged_rectangles.extend(merge_overlapping_rectangles(rectangles_by_cat[cat]))

    visualize_pin_blocking_rectangles(merged_rectangles, grid_size, ratio_x, ratio_y, port_pos, out_path='check.jpg', title='Pin Blocking Rectangles')
    
    return merged_rectangles

def load_dataflow(args, problem):
    dataflow_mat = np.load(os.path.join("dataflow_info", f"{args.benchmark}/2/dataflow_mat.npy"))
    id2index_file = os.path.join("dataflow_info", f"{args.benchmark}/2/macro_name2index_map.pkl")
    with open(id2index_file, 'rb') as f:
        name2index = pickle.load(f)
    id2index = {}
    for macro_name in name2index.keys():
        macro_id =problem.dmp_placedb.node_name2id_map[macro_name]
        id2index[macro_id] = name2index[macro_name]
    return dataflow_mat, id2index

def load_env_data(args, problem):
    # Check if preprocessed file exists
    processed_data_dir = "preprocessed_data"
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    # Build preprocessed file path
    benchmark_name = args.benchmark
    preprocessed_file_path = os.path.join(processed_data_dir, f"{benchmark_name}.pt")
    # If preprocessed file exists, load it directly
    if os.path.exists(preprocessed_file_path):
        print(f"Found preprocessed file: {preprocessed_file_path}")
        try:
            env_params = th.load(preprocessed_file_path)
            if 'dataflow_mat' not in env_params:
                print(f"No dataflow mat found, recalculating...")
                dataflow_mat, id2index = load_dataflow(args, problem)
                env_params['dataflow_mat'] = dataflow_mat
                env_params['id2index'] = id2index
            print(f"Successfully loaded preprocessed file: {benchmark_name}")
            return env_params
        except Exception as e:
            print(f"Failed to load preprocessed file: {e}")
            print("Will recalculate environment parameters...")
    
    # If no preprocessed file, initialize problem and calculate env_params
    print(f"No preprocessed file found, calculating environment parameters: {benchmark_name}")
    problem = ProblemInstance(args, args.benchmark)

    # Pre-compute macro positions with halo adjustment    
    macro_pos = {}
    for macro in problem.macro_pos:
        pos_x, pos_y, size_x, size_y = problem.macro_pos[macro]
        # Adjust position and size to account for halo
        pos_x = max(0, pos_x - args.halo)
        pos_y = max(0, pos_y - args.halo)
        size_x = size_x + 2 * args.halo
        size_y = size_y + 2 * args.halo
        macro_pos[macro] = (pos_x, pos_y, size_x, size_y)

    macro_clusters = problem.macro_cluster_list

    # Visualize clusters with different colors
    out_img = os.path.join(processed_data_dir, f"{benchmark_name}_clusters.jpg")
    visualize_macro_clusters(macro_clusters, macro_pos, out_path=out_img)
    
    # Pre-compute pin blocking rectangles
    grid_size = problem.grid
    pin_blocking_rectangles = compute_pin_blocking_rectangles(
        problem.port_pos, grid_size, problem.ratio_x, problem.ratio_y, block_width_raw=1000.0, block_height_raw=2000.0
    )
    dataflow_mat, id2index = load_dataflow(args, problem)
    env_params = {
        'macro_pos': macro_pos,
        'macro_clusters': macro_clusters,
        'dataflow_mat': dataflow_mat,
        'id2index': id2index,
        'node_id_to_name': problem.node_id_to_name,
        'node_to_net_dict': problem.node_to_net_dict,
        'net_info': problem.net_info,
        'node_info': problem.node_info,
        'port_to_net_dict': problem.port_to_net_dict,
        'port_info': problem.port_info,
        'port_pos': np.array(problem.port_pos),
        'pin_blocking_rectangles': pin_blocking_rectangles,
        'ratio_x': problem.ratio_x,
        'ratio_y': problem.ratio_y,
        'ratio_sum': problem.ratio_x + problem.ratio_y,
    }
    # Save preprocessed file
    try:
        th.save(env_params, preprocessed_file_path)
        print(f"Successfully saved preprocessed file: {preprocessed_file_path}")
    except Exception as e:
        print(f"Failed to save preprocessed file: {e}")

    return env_params

def create_single_env(args, problem):
    """Create a single SelectEnv instance."""
    env_params = load_env_data(args, problem)
    # Add ratio-related attributes from problem to args
    args.ratio_x = env_params['ratio_x']
    args.ratio_y = env_params['ratio_y']
    args.ratio_sum = env_params['ratio_x'] + env_params['ratio_y']
    env = PlaceEnv(args, env_params=env_params)
    return env

def init_ray_envs(args, num_envs, problem):
    """Initialize multiple SelectEnv environments in parallel using Ray."""
    # Prepare environment data only once
    env_params = load_env_data(args, problem)
    # Add ratio-related attributes from problem to args
    args.ratio_x = env_params['ratio_x']
    args.ratio_y = env_params['ratio_y']
    args.ratio_sum = env_params['ratio_x'] + env_params['ratio_y']
    args.n_macro = len(env_params['macro_pos'] if 'macro_pos' in env_params.keys() else env_params['macro_pos_prototype'])

    # Remove placedb-related and other runtime attributes that shouldn't be serialized
    attributes_to_remove = ['placedb', 'problem']
    for attr in attributes_to_remove:
        if hasattr(args, attr):
            delattr(args, attr)
    
    # Determine GPU allocation strategy
    # If using GPU, specify the number of available GPUs
    import torch
    num_gpus_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    ray.init(
        ignore_reinit_error=True, 
        log_to_driver=False, 
        num_cpus=num_envs,
        num_gpus=num_gpus_available,  # Tell Ray how many GPUs are available
        # object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB
    )

    # Each worker requests a fraction of GPU to allow multiple workers to share GPUs
    # Adjust gpu_fraction based on your needs (e.g., 0.1 allows 10 workers per GPU)
    gpu_fraction = 1.0 / num_envs if num_gpus_available > 0 else 0
    
    @ray.remote(num_gpus=gpu_fraction)
    class RayEnvWorker:
        def __init__(self, args, env_params):
            # Set CUDA device for this worker
            # Ray automatically sets CUDA_VISIBLE_DEVICES, so cuda:0 refers to the assigned GPU
            import torch
            if torch.cuda.is_available() and hasattr(args, 'device') and 'cuda' in args.device:
                # Use cuda:0 which Ray maps to the assigned GPU
                args.device = 'cuda:0'
            
            # Each worker creates its own environment but shares data
            self.env = PlaceEnv(args, env_params=env_params)
            
        def reset(self, reward_scaling_flag=False, corner_flag=False, visualize_flag=False):
            return self.env.reset(reward_scaling_flag, corner_flag, visualize_flag)
        def step(self, action):
            return self.env.step(action)
        def visualize_placement(self, i_episode):
            return self.env.visualize_placement(i_episode)
        def visualize_prototype(self):
            return self.env.visualize_prototype()
    
    # Create ray workers, each will create its own environment
    ray_envs = [RayEnvWorker.remote(args, env_params) for _ in range(num_envs)]
    return ray_envs 