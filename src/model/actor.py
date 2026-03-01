import torch as th
import torch.nn as nn
import pdb

from state_parsing import StateParsing

class Actor(nn.Module):
    def __init__(self, args, cnn, cnn_coarse) -> None:
        super(Actor, self).__init__()
        self.args = args
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.arch = getattr(args, 'actor_arch', 'parallel')
        self.local_summary_k = getattr(args, 'local_summary_k', 1)
        if self.arch == 'parallel':
            self.merge = nn.Conv2d(2, 1, 1)
        else:
            # sequential: merge (k local summary + 1 global) -> 1
            self.merge = nn.Conv2d(self.local_summary_k + 1, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.state_parsing = StateParsing(args)
        self.grid = args.grid

    def forward(self, x):
        if self.arch == 'parallel':
            cnn_res, coarse_res = self._forward_parallel(x)
        else:
            cnn_res, coarse_res = self._forward_sequential(x)
        cnn_res = self.merge(th.cat([cnn_res, coarse_res], dim=1))

        # decode with hard position mask
        position_mask = self.state_parsing.state2position_mask(x)
        soft_mask = self.state_parsing.state2soft_mask(x)
        mask_min = soft_mask.min(dim=-1, keepdim=True).values + self.args.soft_coefficient
        soft_mask = soft_mask.le(mask_min).logical_not().double()

        position_mask = position_mask.flatten(start_dim=1, end_dim=2)
        x = cnn_res.reshape(-1, self.grid * self.grid)

        x = th.where(position_mask.double() + soft_mask >= 1.0, -1.0e10, x.double())
        x = self.softmax(x)

        return x

    def _forward_parallel(self, x):
        # Local branch: all masks -> 1 channel
        cnn_input = self.state_parsing.state2input_local(x)
        cnn_res = self.cnn(cnn_input)
        # Global branch: canvas + masks (no pos) -> 1 channel
        coarse_input = self.state_parsing.state2input_global(x)
        coarse_res, _ = self.cnn_coarse(coarse_input)
        return cnn_res, coarse_res

    def _forward_sequential(self, x):
        # Local branch: all masks -> k summary channels
        cnn_input = self.state_parsing.state2input_local(x)
        local_out = self.cnn(cnn_input)  # [B, k, H, W]
        # Global branch: canvas + local summary (k) -> 1 channel
        canvas = self.state_parsing.state2canvas(x)
        coarse_input = th.cat([canvas, local_out], dim=1)
        coarse_res, _ = self.cnn_coarse(coarse_input)
        return local_out, coarse_res
