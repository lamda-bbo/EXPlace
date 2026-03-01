import numpy as np
import torch as th
import pdb


def get_batch_state_from_buffer(buffer, index):
    buffer_batch = [buffer[i] for i in index]

    # Find the maximum number of corners in the batch
    max_corners = max(len(t.state['corners']) for t in buffer_batch)
    
    # Initialize lists to store padded corners and padding masks
    padded_corners = []
    corner_padding_masks = []
    
    for t in buffer_batch:
        corners = t.state['corners']
        current_corners = len(corners)
        
        # Create padding mask: 1 for real corners, 0 for padded corners
        padding_mask = [1] * current_corners + [0] * (max_corners - current_corners)
        corner_padding_masks.append(padding_mask)
        
        # Pad corners with zeros
        if current_corners < max_corners:
            
            # Create zero padding
            padding_size = max_corners - current_corners
            zero_padding = np.zeros((padding_size, corners.shape[1]))
            
            # Concatenate original corners with zero padding
            padded_corner = np.concatenate([corners, zero_padding], axis=0)
            padded_corners.append(padded_corner)
        else:
            # No padding needed
            padded_corners.append(corners)
    
    return {
        'corners': th.tensor(padded_corners),
        'nodes': th.tensor([t.state['nodes'] for t in buffer_batch]),
        'idx': th.tensor([t.state['idx'] for t in buffer_batch]),
        'prototype': th.tensor([t.state['prototype'] for t in buffer_batch]),
        'corner_mask': th.tensor(corner_padding_masks, dtype=th.bool)  # True for real corners, False for padded
    }

class StateParsing:
    def __init__(self, args) -> None:
        self.args = args
        self.grid = args.grid
        self.g2 = self.grid * self.grid
    
    def get_state(self,
        place_idx, 
        canvas,
        masks,
        next_masks,
        size_x,
        size_y,
        prototype_canvas=None,
        ):
        if prototype_canvas is None:
            state = np.concatenate((np.array([place_idx]), canvas.flatten()), axis=0)
        else:
            state = np.concatenate((np.array([place_idx]), canvas.flatten(), prototype_canvas.flatten()), axis=0)

        for mask_name in masks.keys():
            state = np.concatenate((state, masks[mask_name].flatten()), axis=0)
        
        for mask_name in next_masks.keys():
            state = np.concatenate((state, next_masks[mask_name].flatten()), axis=0)
        
        state = np.concatenate((state, np.array([size_x/self.grid, size_y/self.grid])), axis=0)
        return state[np.newaxis, :]

    def state2canvas(self, state):
        """Extract canvas (and optional prototype_canvas) for sequential global input. Returns [B, canvas_num, grid, grid]."""
        canvas_num = 2 if self.args.prototype_flag else 1
        if len(state.shape) == 1:
            return state[1 : 1 + canvas_num * self.g2].reshape(canvas_num, self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 : 1 + canvas_num * self.g2].reshape(-1, canvas_num, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2input_global(self, state):
        # contains canvas, prototype_canvas, masks without position mask
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            state_1 = state[1 : 1 + self.g2 * (mask_num - 1 + canvas_num)].reshape(mask_num - 1 + canvas_num, self.grid, self.grid)
            state_2 = state[1 + self.g2 * (mask_num + canvas_num) : 1 + self.g2 * (mask_num * 2 - 1 + canvas_num)].reshape(mask_num - 1, self.grid, self.grid)
            return th.cat((state_1, state_2), dim=0)
        elif len(state.shape) == 2:
            state_1 = state[:, 1 : 1 + self.g2 * (mask_num - 1 + canvas_num)].reshape(-1, mask_num - 1 + canvas_num, self.grid, self.grid)
            state_2 = state[:, 1 + self.g2 * (mask_num + canvas_num) : 1 + self.g2 * (mask_num * 2 - 1 + canvas_num)].reshape(-1, mask_num - 1, self.grid, self.grid)
            return th.cat((state_1, state_2), dim=1)
        else:
            raise NotImplementedError

    def state2input_local(self, state):
        # contains all masks
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            return state[1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num * 2 + canvas_num)].reshape(mask_num * 2, self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num * 2 + canvas_num)].reshape(-1, mask_num * 2, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2position_mask(self, state):
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        if len(state.shape) == 1:
            return state[1 + self.g2 * (mask_num - 1 + canvas_num) : 1 + self.g2 * (mask_num + 2)].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            return state[:, 1 + self.g2 * (mask_num - 1 + canvas_num) : 1 + self.g2 * (mask_num + canvas_num)].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError
    
    def state2soft_mask(self, state):
        canvas_num = 2 if self.args.prototype_flag else 1
        mask_num = (state.shape[1] - canvas_num * self.g2 - 3) // (self.g2 * 2)
        trade_off_coeff = {mask_name: self.args.trade_off_coeff[i] for i, mask_name in enumerate(self.args.used_masks[:-1])} # last one is position mask
        trade_off_coeff["pos"] = 10.0
        trade_off_coeff = np.array(list(trade_off_coeff.values()))
        if len(state.shape) == 1:
            trade_off_coeff = th.tensor(trade_off_coeff)[:, None].repeat(1, self.g2).to(state.device)
            masks = state[1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num + canvas_num)].reshape(mask_num, self.g2)
            soft_mask = th.sum(masks * trade_off_coeff, dim=0)
            return soft_mask
        elif len(state.shape) == 2:
            trade_off_coeff = th.tensor(trade_off_coeff)[None, :, None].repeat(state.shape[0], 1, self.g2).to(state.device)
            masks = state[:, 1 + self.g2 * canvas_num : 1 + self.g2 * (mask_num + canvas_num)].reshape(-1, mask_num, self.g2)
            soft_mask = th.sum(masks * trade_off_coeff, dim=1)
            return soft_mask
        else:
            raise NotImplementedError
        