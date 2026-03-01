import os
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from state_parsing import StateParsing
from matplotlib.patches import Rectangle, Circle
from utils.visualization import visualize_prototype, visualize_placement, visualize_step
import pdb


placement_rules = {
    "Quadrant I": {
        "Left": lambda x, y, w, h, w_t, h_t: (x - w_t, y + h - h_t),
        "Down": lambda x, y, w, h, w_t, h_t: (x + w - w_t, y - h_t),
        "Up": lambda x, y, w, h, w_t, h_t: (x + w - w_t, y + h),
        "Right": lambda x, y, w, h, w_t, h_t: (x + w, y + h - h_t)
    },
    "Quadrant II": {
        "Right": lambda x, y, w, h, w_t, h_t: (x + w, y + h - h_t),
        "Down": lambda x, y, w, h, w_t, h_t: (x, y - h_t),
        "Up": lambda x, y, w, h, w_t, h_t: (x, y + h),
        "Left": lambda x, y, w, h, w_t, h_t: (x - w_t, y + h - h_t)
    },
    "Quadrant III": {
        "Right": lambda x, y, w, h, w_t, h_t: (x + w, y),
        "Up": lambda x, y, w, h, w_t, h_t: (x, y + h),
        "Left": lambda x, y, w, h, w_t, h_t: (x - w_t, y),
        "Down": lambda x, y, w, h, w_t, h_t: (x, y - h_t),
    },
    "Quadrant IV": {
        "Left": lambda x, y, w, h, w_t, h_t: (x - w_t, y),
        "Up": lambda x, y, w, h, w_t, h_t: (x + w - w_t, y + h),
        "Down": lambda x, y, w, h, w_t, h_t: (x + w - w_t, y - h_t),
        "Right": lambda x, y, w, h, w_t, h_t: (x + w, y),
    }
}

class PlaceEnv():
    def __init__(self, args, env_params=None):
        self.args = args
        self.grid = args.grid

        self.n_macro = args.n_macro
        
        # Ratio-related attributes for regularity mask
        self.ratio_x = args.ratio_x
        self.ratio_y = args.ratio_y
        self.ratio_sum = args.ratio_sum
        self.coef_x = self.ratio_x / self.ratio_sum
        self.coef_y = self.ratio_y / self.ratio_sum

        self.place_idx = 0
        self.macro_pos = {}
        self.port_pos = None
        
        # Store initial parameters that remain unchanged
        self.macro_pos_prototype = env_params.get('macro_pos', {}) if 'macro_pos' in env_params.keys() else env_params.get('macro_pos_prototype', {})
        self.port_pos = env_params.get('port_pos', None)
        self.pin_blocking_rectangles = env_params.get('pin_blocking_rectangles', [])
        self.macro_clusters = env_params.get('macro_clusters', [])
        self.dataflow_mat = env_params.get('dataflow_mat', None)
        self.id2index = env_params.get('id2index', {})
        self.macro_to_place_order = env_params.get('macro_to_place_order', None)
        self.node_id_to_name = env_params.get('node_id_to_name', {})
        self.node_to_net_dict = env_params.get('node_to_net_dict', {})
        self.net_info = env_params.get('net_info', {})
        self.node_info = env_params.get('node_info', {})
        self.port_to_net_dict = env_params.get('port_to_net_dict', {})
        self.port_info = env_params.get('port_info', {})

        self.macro_to_place = []
        self.macro_placed = []
        
        self.reward_history = {mask_name: [] for mask_name in args.used_masks[:-1]} # last one is position mask
        self.reward_history['total'] = []
        self.masks = {mask_name: None for mask_name in args.used_masks}
        self.masks_norm = {mask_name: None for mask_name in args.used_masks}
        self.next_masks_norm = {mask_name: None for mask_name in args.used_masks}
        self.corners = None

        # for reward scaling
        self.reward_max = {mask_name: -np.inf for mask_name in args.used_masks[:-1]}
        self.reward_min = {mask_name: np.inf for mask_name in args.used_masks[:-1]}
        self.reward_scaling_flag = False

        # for corner
        self.corner_flag = args.corner_flag

        # for visualization
        self.visualize_flag = getattr(args, 'visualize_flag', False)

        self.state_parsing = StateParsing(args)

        # get mask by name
        self.mask_functions = {
            'wire': self.get_wire_mask,
            'df': self.get_dataflow_mask,
            'hier': self.get_hierarchy_mask,
            'port': self.get_port_mask,
            'displacement': self.get_displacement_mask,
            'pos': self.get_position_mask,
            'reg': self.get_regularity_mask
        }
        
        # trade-off parameters dict
        self.trade_off_coeff = {mask_name: args.trade_off_coeff[i] for i, mask_name in enumerate(args.used_masks[:-1])} # last one is position mask
        
        # Prune dataflow_mat at initialization, only keep the top 10% strongest connections
        if self.args.dataflow_cutoff > 0:
            keep_ratio = 1 - self.args.dataflow_cutoff
            self.prune_dataflow_mat(keep_ratio=keep_ratio)

    def reset(self, reward_scaling_flag=False, corner_flag=False, visualize_flag=False):
        self.place_idx = 0
        self.macro_to_place.clear()
        self.macro_placed.clear()
        for mask_name in self.reward_history:
            self.reward_history[mask_name].clear()

        self.reward_scaling_flag = reward_scaling_flag
        self.macro_pos = self.macro_pos_prototype.copy()
        self.macro_placed = []

        if corner_flag:
            self.corner_flag = True
        self.visualize_flag = visualize_flag

        self.macro_to_place = list(self.macro_pos.keys())
        self.set_place_order_maskplace()
        self.reset_net_to_macro()

        size_x = self.macro_pos[self.macro_to_place[self.place_idx]][2]
        size_y = self.macro_pos[self.macro_to_place[self.place_idx]][3]
        
        self.canvas = np.zeros((self.grid, self.grid))
        if self.args.regulator_flag:
            self.prototype_canvas = np.zeros((self.grid, self.grid))
            # Draw the canvas of the GP prototype
            for macro in self.macro_to_place:
                pos_x, pos_y, size_x, size_y = self.macro_pos_prototype[macro]
                assert pos_x + size_x <= self.grid, (pos_x, size_x, self.grid)
                assert pos_y + size_y <= self.grid, (pos_y, size_y, self.grid)
                self.prototype_canvas = self.__draw_canvas(self.prototype_canvas, pos_x, pos_y, size_x, size_y)
        else:
            self.prototype_canvas = None
        
        # Calculate mask for the first macro
        first_macro = self.macro_to_place[self.place_idx]
        for mask_name in self.args.used_masks:
            if mask_name == 'df' or mask_name == 'hier':
                self.masks[mask_name] = np.zeros((self.grid, self.grid))
            else:
                self.masks[mask_name] = self.mask_functions[mask_name](first_macro)

        # for the second macro to look ahead
        second_macro = self.macro_to_place[self.place_idx + 1]
        for mask_name in self.args.used_masks:
            if mask_name == 'df' or mask_name == 'hier':
                self.next_masks_norm[mask_name] = np.zeros((self.grid, self.grid))
            else:
                self.next_masks_norm[mask_name] = self.mask_functions[mask_name](second_macro)

        for mask_name in self.masks.keys():
            if mask_name == "pos":
                self.masks_norm[mask_name] = self.masks[mask_name]
                self.next_masks_norm[mask_name] = self.next_masks_norm[mask_name]
                continue
            self.masks_norm[mask_name], self.next_masks_norm[mask_name] = self.__mask_normalization(self.masks[mask_name], self.next_masks_norm[mask_name])

        self.state = self.state_parsing.get_state(
            place_idx=self.place_idx,
            canvas=self.canvas,
            prototype_canvas=self.prototype_canvas,
            masks=self.masks_norm,
            next_masks=self.next_masks_norm,
            size_x=size_x,
            size_y=size_y
        )

        return self.state.copy()

    def reset_net_to_macro(self):
        # reset net_to_macro
        self.net_to_macro = {}
        
        for port_name in self.port_to_net_dict:
            for net_name in self.port_to_net_dict[port_name]:
                pin_x = round(self.port_info[port_name]['x'] / self.ratio_x)
                pin_y = round(self.port_info[port_name]['y'] / self.ratio_y)

                if net_name in self.net_to_macro:
                    self.net_to_macro[net_name][port_name] = (pin_x, pin_y)
                else:
                    self.net_to_macro[net_name] = {}
                    self.net_to_macro[net_name][port_name] = (pin_x, pin_y)
        
        if self.args.regulator_flag:
            for macro in self.macro_pos_prototype:
                for net_name in self.node_to_net_dict[macro]:
                    x, y, _, _ = self.macro_pos_prototype[macro]
                    pin_x = round((x * self.ratio_x + self.node_info[macro]['x']/2 + \
                            self.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
                    pin_y = round((y * self.ratio_y + self.node_info[macro]['y']/2 + \
                            self.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
            
                    if net_name in self.net_to_macro:
                        self.net_to_macro[net_name][macro] = (pin_x, pin_y)
                    else:
                        self.net_to_macro[net_name] = {}
                        self.net_to_macro[net_name][macro] = (pin_x, pin_y)

    def step(self, action):
        # decode action
        x = round(action // self.grid)
        y = round(action % self.grid)        
        macro = self.macro_to_place[self.place_idx]

        if self.corner_flag:
            corners = self.find_all_corners(macro)
            # find the nearest corner
            nearest_corner = self.find_nearest_corner(corners, x, y)
            x, y = nearest_corner
        
            # Visualize current step before action execution if visualization is enabled
            if self.visualize_flag:
                self.visualize_step(step_idx=self.place_idx, corners=corners, current_macro=macro, action=(x, y))

        # update placed macros
        _, _, size_x, size_y = self.macro_pos[macro]
        # assert (0 <= x and x + size_x <= self.grid and 0 <= y and y + size_y <= self.grid), "Macro is out of bounds"
        
        self.canvas = self.__draw_canvas(self.canvas, x, y, size_x, size_y)
        self.macro_pos[macro] = (x, y, size_x, size_y)
        self.macro_placed.append(macro)
        
        # reward computation
        costs = {cost_name: self.masks[cost_name][x, y] for cost_name in self.reward_max.keys()}

        if 'wire' in self.args.used_masks:
            for net_name in self.node_to_net_dict[macro]:
                pin_x = round((x * self.ratio_x + self.node_info[macro]['x']/2 + \
                        self.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
                pin_y = round((y * self.ratio_y + self.node_info[macro]['y']/2 + \
                        self.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
                if net_name in self.net_to_macro:
                    self.net_to_macro[net_name][macro] = (pin_x, pin_y)
                else:
                    self.net_to_macro[net_name] = {}
                    self.net_to_macro[net_name][macro] = (pin_x, pin_y)

        # reward scaling
        if self.args.use_reward_scaling and self.reward_scaling_flag:
            for cost_name in costs.keys():
                self.reward_max[cost_name] = max(self.reward_max[cost_name], costs[cost_name])
                self.reward_min[cost_name] = min(self.reward_min[cost_name], costs[cost_name])

        if self.args.use_reward_scaling:
            costs_norm = {cost_name: (costs[cost_name] - self.reward_min[cost_name] + 1e-10) / (self.reward_max[cost_name] - self.reward_min[cost_name] + 1e-10) for cost_name in costs.keys()}
            reward = sum([-self.trade_off_coeff[cost_name] * costs_norm[cost_name] for cost_name in costs.keys()])

        else:
            reward = sum([-self.trade_off_coeff[cost_name] * costs[cost_name] for cost_name in costs.keys()])

        # recording reward history
        for mask_name in costs.keys():
            self.reward_history[mask_name].append(costs[mask_name])
        self.reward_history['total'].append(reward)
        
        # state transition, i.e., feature representation of the next step
        for mask_name in self.args.used_masks:
            self.masks[mask_name] = np.zeros((self.grid, self.grid))
            self.next_masks_norm[mask_name] = np.zeros((self.grid, self.grid))

        self.place_idx += 1 # next macro
        if self.place_idx < len(self.macro_to_place):
            next_macro = self.macro_to_place[self.place_idx]
            for mask_name in self.args.used_masks:
                self.masks[mask_name] = self.mask_functions[mask_name](next_macro)

        if self.place_idx < len(self.macro_to_place) - 1:
            next_next_macro = self.macro_to_place[self.place_idx + 1]
            for mask_name in self.args.used_masks:
                self.next_masks_norm[mask_name] = self.mask_functions[mask_name](next_next_macro)

        # episode ends
        done = False
        if self.place_idx >= len(self.macro_to_place):
            total_reward = np.sum(self.reward_history['total'])
            self.state = np.zeros(1)
            done = True
        
        for mask_name in self.args.used_masks:
            if mask_name == "pos":
                self.masks_norm[mask_name] = self.masks[mask_name]
                self.next_masks_norm[mask_name] = self.next_masks_norm[mask_name]
                continue
            self.masks_norm[mask_name], self.next_masks_norm[mask_name] = self.__mask_normalization(self.masks[mask_name], self.next_masks_norm[mask_name])

        self.state = self.state_parsing.get_state(
            place_idx=self.place_idx,
            canvas=self.canvas,
            prototype_canvas=self.prototype_canvas,
            masks=self.masks_norm,
            next_masks=self.next_masks_norm,
            size_x=size_x,
            size_y=size_y
        )

        info = {
            "scaled_reward" : 0 if (not done) else total_reward,
            "macro_pos": copy.deepcopy(self.macro_pos) if done else None,
            "macro": macro
        }
        for reward_name in self.reward_history:
            info[reward_name] = np.sum(self.reward_history[reward_name])
        
        # clear next_masks_norm
        self.next_masks_norm = {mask_name: None for mask_name in self.args.used_masks}

        return self.state.copy(), reward, done, info

    def set_place_order_maskplace(self):
        # original order of maskplace (Lai et al., NeurIPS2021)
        original_order = sorted(self.macro_to_place, key=lambda x: self.node_id_to_name.index(x))
        # reshape order by macro clustering: when a macro from a cluster first appears,
        # append the rest of that cluster (in original order) right after it
        if not self.macro_clusters:
            self.macro_to_place = original_order
            return
        orig_index = {m: i for i, m in enumerate(original_order)}
        new_order = []
        added = set()
        for m in original_order:
            if m in added:
                continue
            new_order.append(m)
            added.add(m)
            for cluster in self.macro_clusters:
                if m in cluster:
                    others = [x for x in cluster if x != m and x not in added]
                    others.sort(key=lambda x: orig_index[x])
                    for x in others:
                        new_order.append(x)
                        added.add(x)
                    break
        self.macro_to_place = new_order

    def find_nearest_corner(self, corners, x, y):
        distances = np.linalg.norm(corners - np.array([x, y]), axis=1)
        return corners[np.argmin(distances)]

    def find_all_corners(self, next_macro):
        # find all corners to place the macro
        # 1. Since the placed macros are also at corners, we can only consider two corners: Up/Dowm, and Left/Right
        # 2. Check feasibility of each corner
        # Use a dict to store corners for easy duplicate checking
        corners_dict = {}
        _, _, size_x, size_y = self.macro_pos[next_macro]
        
        # add macro corners
        for macro in self.macro_placed:
            x, y, width, height = self.macro_pos[macro]
            
            # determine placement rules from quadrant
            quadrant = self.determine_quadrant(x, y)
            rules = placement_rules[f"Quadrant {quadrant}"]

            for _, rule_func in rules.items():
                new_x, new_y = rule_func(x, y, width, height, size_x, size_y)
                # Use dict to check if (new_x, new_y) already exists
                if (new_x, new_y) in corners_dict:
                    continue
                feasibility = self.check_corner(new_x, new_y, size_x, size_y)
                if feasibility:
                    corners_dict[(new_x, new_y)] = 0
        
        # add canvas corners
        b = self.args.core_area_block
        canvas_corners = [[b, b], [b, self.grid - size_y - b], [self.grid - size_x - b, self.grid - size_y - b], [self.grid - size_x - b, b]]
        for (x, y) in canvas_corners:
            if (x, y) in corners_dict:
                continue
            feasibility = self.check_corner(x, y, size_x, size_y)
            if feasibility:
                corners_dict[(x, y)] = 0

        # Convert dict to list format [[x, y], ...]
        corners = np.array([[x, y] for (x, y) in corners_dict.keys()])

        return corners

    def check_corner(self, x, y, size_x, size_y):
        # Ensure macro is fully inside the grid
        if not (0 <= x and x + size_x <= self.grid and 0 <= y and y + size_y <= self.grid):
            return False
        # Ensure non-overlapping with placed macros
        if not self.masks["pos"][x, y]:
            return True
        return False
        
    def determine_quadrant(self, x, y):
        mid_x, mid_y = self.grid / 2, self.grid / 2
        if x >= mid_x and y >= mid_y:
            return "I"
        elif x < mid_x and y >= mid_y:
            return "II"
        elif x < mid_x and y < mid_y:
            return "III"
        elif x >= mid_x and y < mid_y:
            return "IV"
    
    def __draw_canvas(self, canvas, x, y, size_x, size_y):
        canvas[x : x+size_x, y : y+size_y] = 1.0
        canvas[x : x + size_x, y] = 0.5
        if y + size_y -1 < self.grid:
            canvas[x : x + size_x, max(0, y + size_y -1)] = 0.5
        canvas[x, y: y + size_y] = 0.5
        if x + size_x - 1 < self.grid:
            canvas[max(0, x+size_x-1), y: y + size_y] = 0.5

        return canvas

    def get_wire_mask(self, macro):
        mask = np.zeros(shape=(self.grid, self.grid))  
        
        for net_name in self.node_to_net_dict[macro]:
            if net_name in self.net_to_macro:
                delta_pin_x = round((self.macro_pos[macro][2]/2 + \
                    self.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
                delta_pin_y = round((self.macro_pos[macro][3]/2 + \
                    self.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
                
                if self.args.regulator_flag:
                    pin_x, pin_y = self.net_to_macro[net_name][macro]
                    del self.net_to_macro[net_name][macro]

                pin_array = np.array(list(self.net_to_macro[net_name].values()))
                max_x = max(pin_array[:, 0])
                min_x = min(pin_array[:, 0])
                max_y = max(pin_array[:, 1])
                min_y = min(pin_array[:, 1])
        
                start_x = min_x - delta_pin_x
                end_x = max_x - delta_pin_x
                start_y = min_y - delta_pin_y
                end_y = max_y - delta_pin_y

                start_x = max(start_x, 0)
                start_y = max(start_y, 0)
                end_x = max(end_x, 0)
                end_y = max(end_y, 0)

                start_x = min(start_x, self.grid)
                start_y = min(start_y, self.grid)
                end_x  = min(end_x, self.grid)
                end_y  = min(end_y, self.grid)

                if not 'weight' in self.net_info[net_name]:
                    weight = 1.0
                else:
                    weight = self.net_info[net_name]['weight']

                for i in range(0, start_x):
                    mask[i, :] += (start_x - i) * weight * self.coef_x
                for i in range(end_x+1, self.grid):
                    mask[i, :] +=  (i- end_x) * weight * self.coef_x
                for j in range(0, start_y):
                    mask[:, j] += (start_y - j) * weight * self.coef_y
                for j in range(end_y+1, self.grid):
                    mask[:, j] += (j - end_y) * weight * self.coef_y
                
                if self.args.regulator_flag:
                    mask -= mask[self.macro_pos_prototype[macro][0], self.macro_pos_prototype[macro][1]]
                    self.net_to_macro[net_name][macro] = (pin_x, pin_y)

        return mask

    def get_dataflow_mask(self, macro_to_place):
        # pdb.set_trace()
        # Initialize full mask of shape (grid, grid)
        mask = np.zeros((self.grid, self.grid))
        _, _, size_x, size_y = self.macro_pos[macro_to_place]

        center_offset_x = size_x / 2.0
        center_offset_y = size_y / 2.0

        if len(self.macro_placed) == 0:  # First step, zero cost
            return mask

        # Generate meshgrid for all positions (i, j) in the grid
        x = np.arange(self.grid)
        y = np.arange(self.grid)
        xx, yy = np.meshgrid(x, y, indexing='ij')  # Shape: (grid, grid)

        # Compute center positions for each (i, j)
        center_a = np.stack([
            xx + center_offset_x,
            yy + center_offset_y
        ], axis=-1)  # Shape: (grid, grid, 2)

        # Precompute placed macro centers and flow weights
        num_placed = len(self.macro_placed)
        centers_b = np.zeros((num_placed, 2))
        flows = np.zeros((num_placed,))
        index_a = self.id2index[macro_to_place]

        for idx, macro_b in enumerate(self.macro_placed):
            x_b, y_b, sx_b, sy_b = self.macro_pos[macro_b]
            centers_b[idx] = [x_b + sx_b / 2.0, y_b + sy_b / 2.0]
            index_b = self.id2index[macro_b]
            flows[idx] = self.dataflow_mat[index_a, index_b]

        # Compute distances from all (grid x grid) center_a to each center_b
        # center_a: (grid, grid, 2), centers_b: (num_placed, 2)
        # Broadcast subtraction: (grid, grid, num_placed, 2)
        diff = center_a[:, :, None, :] - centers_b[None, None, :, :]
        # Scale by coef so distance is in real units
        diff_real = diff * np.array([self.coef_x, self.coef_y])
        dists = np.linalg.norm(diff_real, axis=-1)  # Shape: (grid, grid, num_placed)

        # Multiply each distance by its corresponding flow weight
        weighted_dists = dists * flows[None, None, :]  # Broadcasting

        # Sum over all placed macros to get final cost at each grid location
        mask = np.sum(weighted_dists, axis=-1)  # Shape: (grid, grid)

        return mask
    
    def get_position_mask(self, macro_to_place):
        """Calculate position mask to avoid overlapping with placed macros.
        Marks positions that would cause overlap with existing macros or go out of bounds.
        
        Args:
            macro_to_place: ID of the macro to be placed
            
        Returns:
            mask: A grid x grid numpy array where 1 indicates invalid positions
        """
        mask = np.zeros((self.grid, self.grid))
        _, _, size_x, size_y = self.macro_pos[macro_to_place]
        
        # Mark positions that would overlap with placed macros
        for macro in self.macro_placed:
            start_x = max(0, self.macro_pos[macro][0] - size_x + 1)
            start_y = max(0, self.macro_pos[macro][1] - size_y + 1)
            end_x = min(self.macro_pos[macro][0] + self.macro_pos[macro][2] - 1, self.grid)
            end_y = min(self.macro_pos[macro][1] + self.macro_pos[macro][3] - 1, self.grid)
            mask[start_x: end_x + 1, start_y: end_y + 1] = 1

        # Mark positions that would go out of bounds and core area block
        block = self.args.core_area_block
        mask[self.grid - size_x + 1 - block:, :] = 1
        mask[:, self.grid - size_y + 1 - block:] = 1
        mask[:block, :] = 1
        mask[:, :block] = 1
        
        return mask
    
    def get_regularity_mask(self, macro_to_place):
        x, y, size_x, size_y = self.macro_pos_prototype[macro_to_place]
        mask = np.zeros((self.grid, self.grid))
        start_x = 1
        start_y = 1
        end_x = self.grid - size_x - 1
        end_y = self.grid - size_y - 1

        # mask (vectorized)
        rows = np.arange(self.grid)
        cols = np.arange(self.grid)

        # For each row r, x_mask_1 adds coef_x for i in [start_x, min(r, end_x)]
        # => count = r - start_x + 1 within [start_x, end_x], else 0
        x1_row = np.where((rows >= start_x) & (rows <= end_x), rows - start_x + 1, 0)
        # For each row r, x_mask_2 adds coef_x for i in [r, end_x] intersect [start_x, end_x]
        # => count = end_x - r + 1 within [start_x, end_x], else 0
        x2_row = np.where((rows >= start_x) & (rows <= end_x), end_x - rows + 1, 0)
        x_row = self.coef_x * np.minimum(x1_row, x2_row)
        x_mask = x_row[:, None].repeat(self.grid, axis=1)

        # Symmetric logic on columns for y-masks
        y1_col = np.where((cols >= start_y) & (cols <= end_y), cols - start_y + 1, 0)
        y2_col = np.where((cols >= start_y) & (cols <= end_y), end_y - cols + 1, 0)
        y_col = self.coef_y * np.minimum(y1_col, y2_col)
        y_mask = y_col[None, :].repeat(self.grid, axis=0)

        mask = x_mask + y_mask
        # mask = np.minimum(x_mask, y_mask)
        if self.args.regulator_flag:
            mask -= mask[x, y]

        return mask

    def get_hierarchy_mask(self, macro_to_place):
        """Calculate hierarchy mask using vectorized operations to avoid nested loops.
        The mask represents the area increment when placing a macro at each position.
        
        Args:
            macro_to_place: ID of the macro to be placed
            
        Returns:
            mask: A grid x grid numpy array containing area increments
        """
        _, _, size_x, size_y = self.macro_pos[macro_to_place]
        
        # Initialize mask with zeros
        mask = np.zeros((self.grid, self.grid))

        # Find the cluster that contains the macro_to_place
        target_cluster = None
        for cluster in self.macro_clusters:
            if macro_to_place in cluster:
                target_cluster = cluster
                break
        
        if target_cluster is None:
            return mask
            
        # Find placed macros in the same cluster
        placed_macros_in_cluster = [m for m in target_cluster if m in self.macro_placed]
        
        if not placed_macros_in_cluster:
            return mask
            
        # Calculate original bounding box
        min_x = min(self.macro_pos[m][0] for m in placed_macros_in_cluster)
        min_y = min(self.macro_pos[m][1] for m in placed_macros_in_cluster)
        max_x = max(self.macro_pos[m][0] + self.macro_pos[m][2] for m in placed_macros_in_cluster)
        max_y = max(self.macro_pos[m][1] + self.macro_pos[m][3] for m in placed_macros_in_cluster)
        original_area = ((max_x - min_x) * self.coef_x) * ((max_y - min_y) * self.coef_y)
        
        # Create coordinate grids for all possible macro positions
        x_positions = np.arange(self.grid - size_x + 1)
        y_positions = np.arange(self.grid - size_y + 1)
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='ij')  # Shape: (grid-size_x+1, grid-size_y+1)
        
        # Vectorized calculation of new bounding box for each position
        new_min_x = np.minimum(min_x, xx)
        new_min_y = np.minimum(min_y, yy)
        new_max_x = np.maximum(max_x, xx + size_x)
        new_max_y = np.maximum(max_y, yy + size_y)
        
        # Calculate new area for each position
        new_area = ((new_max_x - new_min_x) * self.coef_x) * ((new_max_y - new_min_y) * self.coef_y)
        
        # Area increment is the difference between new and original areas
        area_increment = new_area - original_area
        
        # Assign to mask
        mask[:self.grid - size_x + 1, :self.grid - size_y + 1] = area_increment
                
        return mask
    
    def get_port_mask(self, macro_to_place):
        """Calculate port mask based on overlap area between macro and port blocking rectangles.
        Each port has a blocking rectangle extending inward from the port position.
        Uses vectorized operations for better performance.
        
        Args:
            macro_to_place: ID of the macro to be placed
            
        Returns:
            mask: A grid x grid numpy array containing overlap areas with port blocking rectangles
        """
        _, _, size_x, size_y = self.macro_pos[macro_to_place]
        
        # Initialize mask with zeros
        mask = np.zeros((self.grid, self.grid))
        
        # If no ports available, return zero mask
        if self.port_pos is None or len(self.port_pos) == 0:
            return mask
        
        # Create coordinate grids for all possible macro positions
        x_positions = np.arange(self.grid - size_x + 1)
        y_positions = np.arange(self.grid - size_y + 1)
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='ij')  # Shape: (grid-size_x+1, grid-size_y+1)
        
        # Reshape to (num_positions, 2) for vectorized operations
        positions = np.stack([xx.flatten(), yy.flatten()], axis=-1)  # Shape: (num_positions, 2)
        num_positions = positions.shape[0]
        
        # Initialize total overlap array (real units: x and y scaled separately then area)
        total_overlap_real = np.zeros(num_positions)

        for block_rect in self.pin_blocking_rectangles:
            overlap_real = self._calculate_rectangle_overlap_vectorized(
                positions, size_x, size_y, block_rect, self.coef_x, self.coef_y
            )
            total_overlap_real += overlap_real

        # Reshape back to grid shape
        mask[:self.grid - size_x + 1, :self.grid - size_y + 1] = total_overlap_real.reshape(
            self.grid - size_x + 1, self.grid - size_y + 1
        )
        if self.args.regulator_flag:
            # relative improvement over the old position
            old_x, old_y, _, _ = self.macro_pos_prototype[macro_to_place]
            mask -= mask[old_x, old_y]

        return mask
    
    def get_displacement_mask(self, macro_to_place):
        """Calculate displacement mask between current position and guide position using vectorized operations.
        Uses Euclidean distance (L2 norm) to measure displacement.
        
        Args:
            macro_to_place: ID of the macro to be placed
            
        Returns:
            mask: A grid x grid numpy array containing displacement values
        """        
        # Get guide position
        guide_x, guide_y, _, _ = self.macro_pos_prototype[macro_to_place]
        
        # Create coordinate grids
        x = np.arange(self.grid)
        y = np.arange(self.grid)
        xx, yy = np.meshgrid(x, y, indexing='ij')  # Shape: (grid-size_x+1, grid-size_y+1)
        
        # Calculate Euclidean distance in real units
        dx = (xx - guide_x) * self.coef_x
        dy = (yy - guide_y) * self.coef_y
        mask = np.sqrt(dx * dx + dy * dy)
        return mask

    def _calculate_rectangle_overlap_vectorized(self, positions, macro_w, macro_h, block_rect, coef_x, coef_y):
        """Calculate overlap areas between multiple macro positions and a single blocking rectangle.
        X and y are scaled separately, then area = (width * ratio_x) * (height * ratio_y).

        Args:
            positions: (num_positions, 2) array of macro positions (x, y)
            macro_w: Width of macro
            macro_h: Height of macro
            block_rect: (x, y, width, height) of blocking rectangle
            coef_x, coef_y: scale grid width/height (e.g. ratio_x/ratio_sum for consistency with wire/reg)

        Returns:
            overlap_areas: (num_positions,) array of overlap areas (x and y scaled separately, then area)
        """
        block_x, block_y, block_w, block_h = block_rect

        # Macro rectangle coordinates for all positions
        macro_left = positions[:, 0]
        macro_right = macro_left + macro_w
        macro_bottom = positions[:, 1]
        macro_top = macro_bottom + macro_h

        # Blocking rectangle coordinates
        block_left = block_x
        block_right = block_x + block_w
        block_bottom = block_y
        block_top = block_y + block_h

        # Intersection in grid units
        left = np.maximum(macro_left, block_left)
        right = np.minimum(macro_right, block_right)
        bottom = np.maximum(macro_bottom, block_bottom)
        top = np.minimum(macro_top, block_top)

        width = np.maximum(0, right - left)
        height = np.maximum(0, top - bottom)
        # Scale x and y separately (same as wire/regularity: ratio/ratio_sum), then area
        overlap_areas = (width * coef_x) * (height * coef_y)
        return overlap_areas

    def __mask_normalization(self, mask1, mask2=None):
        mask1_ = copy.deepcopy(mask1)
        if mask2 is not None:
            mask2_ = copy.deepcopy(mask2)
            if np.abs(mask1).max() > 0 or np.abs(mask2).max() > 0:
                mask1_ /= max(np.abs(mask1_).max(), np.abs(mask2_).max())
                mask2_ /= max(np.abs(mask1).max(), np.abs(mask2).max())
            return mask1_, mask2_
        else:
            mask1_ /= max(np.abs(mask1_).max(), 1e-10)
            return mask1_, None
    
    def prune_dataflow_mat(self, keep_ratio=0.1):
        """
        Prune dataflow_mat, only keep the top keep_ratio strongest connections, set others to 0.
        :param keep_ratio: Ratio to keep (e.g., 0.1 means keep top 10% strongest connections)
        """
        mat = self.dataflow_mat
        if mat is None:
            return
        # Only consider nonzero elements
        nonzero = mat[mat > 0]
        if len(nonzero) == 0:
            return
        # Calculate the threshold, keep only the top keep_ratio connections
        threshold = np.percentile(nonzero, 100 * (1 - keep_ratio))
        # Set values below threshold to 0
        pruned_mat = np.where(mat >= threshold, mat, 0)
        self.dataflow_mat = pruned_mat

    #########################################################
    # Visualization
    #########################################################
    def visualize_prototype(self):
        dataflow_mat = getattr(self, 'dataflow_mat', None)
        id2index = getattr(self, 'id2index', None)
        port_pos = getattr(self, 'port_pos', None)
        pin_blocking_rectangles = getattr(self, 'pin_blocking_rectangles', None)
        
        visualize_prototype(
            self, self.macro_pos_prototype, self.macro_clusters, self.args,
            self.grid, self.ratio_x, self.ratio_y,
            dataflow_mat=dataflow_mat, id2index=id2index,
            port_pos=port_pos, pin_blocking_rectangles=pin_blocking_rectangles
        )
    
    def visualize_placement(self, i_episode, test_mode=False, path=None):
        dataflow_mat = getattr(self, 'dataflow_mat', None)
        id2index = getattr(self, 'id2index', None)
        port_pos = getattr(self, 'port_pos', None)
        pin_blocking_rectangles = getattr(self, 'pin_blocking_rectangles', None)
        
        visualize_placement(
            self, self.macro_pos, self.macro_clusters, self.args,
            self.grid, self.ratio_x, self.ratio_y, i_episode,
            test_mode=test_mode, path=path,
            dataflow_mat=dataflow_mat, id2index=id2index,
            port_pos=port_pos, pin_blocking_rectangles=pin_blocking_rectangles
        )
    
    def visualize_step(self, step_idx, corners=None, current_macro=None, action=None):
        """
        Visualize candidate corners and placed macros for each step
        
        Args:
            step_idx: Current step index
            current_macro: Name of the macro to be placed
            action: Selected action position (x, y)
        """
        port_pos = getattr(self, 'port_pos', None)
        pin_blocking_rectangles = getattr(self, 'pin_blocking_rectangles', None)
        
        visualize_step(
            self, self.macro_placed, self.macro_pos, self.macro_clusters, self.macro_to_place,
            self.place_idx, self.args, self.grid, self.ratio_x, self.ratio_y, step_idx,
            corners=corners, current_macro=current_macro, action=action,
            port_pos=port_pos, pin_blocking_rectangles=pin_blocking_rectangles
        )