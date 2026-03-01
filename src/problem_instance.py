import numpy as np
import os
import copy
import math
import re
import pickle
from pathlib import Path
import time
import logging
import torch as th
import networkit as nk
from networkit import graphio

from dreamplace.Params import Params
from dreamplace.PlaceDB import PlaceDB
from dreamplace.NonLinearPlace import NonLinearPlace
import dreamplace.Timer as Timer
import dreamplace.ops.place_io.place_io as place_io

from itertools import combinations
from PIL import Image
from utils.visualization import plot_placement
from utils.coord_utils import grid_to_real, real_to_grid


def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)

class ProblemInstance():
    def __init__(self, args, benchmark, init=True):
        self.benchmark = benchmark
        self.args = args
        self.gp_hpwl = None
        self.regularity = None
        self.database = {}
        self.dmp_params = self._load_dmp_params()
        self.dmp_placedb = PlaceDB()
        self.dmp_placedb(self.dmp_params)
        if args.use_timer_for_evaluation:
            self.timer = Timer.Timer()
            self.timer(self.dmp_params, self.dmp_placedb)
            # This must be done to explicitly execute the parser builders.
            # The parsers in OpenTimer are all in lazy mode.
            self.timer.update_timing()
        else:
            self.timer = None
        self.dmp_placer = NonLinearPlace(self.dmp_params, self.dmp_placedb, timer=self.timer)
        self.results = {}

        self.max_width = self.dmp_placedb.xh - self.dmp_placedb.xl
        self.max_height = self.dmp_placedb.yh - self.dmp_placedb.yl
        self.num_movable_nodes = self.dmp_placedb.num_movable_nodes

        self.grid = self.args.grid
        self.ratio_x = self.max_width / self.grid
        self.ratio_y = self.max_height / self.grid

        # if not init, only load the dmp_params and dmp_placedb
        if not init:
            return

        self.macro_names = []
        self.macros = []
        self.macro_x = []
        self.macro_y = []
        self.macro_size_x = []
        self.macro_size_y = []
        # Calculate average node area
        self.port_indices = []     
        total_area = 0

        for node_name in self.dmp_placedb.node_names:
            node = self.dmp_placedb.node_name2id_map[node_name.decode('utf-8')]
            if node < (self.dmp_placedb.num_physical_nodes - self.dmp_placedb.num_terminal_NIs):  # exclude IO ports
                total_area += self.dmp_placedb.node_size_x[node] * self.dmp_placedb.node_size_y[node]
            else:
                self.port_indices.append(node)  # store the port indices
        avg_area = total_area / len(self.dmp_placedb.node_names)
        
        # Identify macros based on area and height criteria
        ratio = 0.0001 if self.benchmark == "superblue10" or self.benchmark == "superblue7" else 0.001
        for node_name in self.dmp_placedb.node_names:
            node = self.dmp_placedb.node_name2id_map[node_name.decode('utf-8')]
            if node < (self.dmp_placedb.num_physical_nodes - self.dmp_placedb.num_terminal_NIs):  # exclude IO ports
                area = self.dmp_placedb.node_size_x[node] * self.dmp_placedb.node_size_y[node]
                height = self.dmp_placedb.node_size_y[node]
                if "superblue" in args.benchmark:
                    is_macro = (area > ratio * total_area)
                else:
                    is_macro = (area > 10 * avg_area or height > 2 * self.dmp_placedb.row_height)
                if is_macro:
                    self.macros.append(node)
                    self.macro_names.append(node_name.decode('utf-8'))
                    self.macro_x.append(self.dmp_placedb.node_x[node])
                    self.macro_y.append(self.dmp_placedb.node_y[node])
                    self.macro_size_x.append(self.dmp_placedb.node_size_x[node])
                    self.macro_size_y.append(self.dmp_placedb.node_size_y[node])

        self.macro_names = np.array(self.macro_names).astype(np.str_)
        self.n_macro = len(self.macro_names)
        self.node_names = self.dmp_placedb.node_names.astype(np.str_)

        self.node_size_x = self.dmp_placedb.node_size_x
        self.node_size_y = self.dmp_placedb.node_size_y

        self.pin_offset_x = self.dmp_placedb.pin_offset_x
        self.pin_offset_y = self.dmp_placedb.pin_offset_y

        self.net_names = self.dmp_placedb.net_names.astype(np.str_)
        self.net2pin_map = self.dmp_placedb.net2pin_map
        self.pin2node_map = self.dmp_placedb.pin2node_map
        

        self.node_info, self.node_info_raw_id_name = self.get_node_info()
        self.node_cnt = self.n_macro

        start_time = time.time()
        self.net_info = self.get_net_info()
        time_net_info = time.time()
        print(f"get net info time: {time_net_info - start_time} seconds")

        # compute macro clusters
        self.macro_cluster_list = self.community_partition()
        time_macro_cluster_list = time.time()
        print(f"get macro cluster list time: {time_macro_cluster_list - time_net_info} seconds")
        # compute dataflow mat
        self.dataflow_mat, self.macro2index_map = self.compute_virtual_connections()
        print(f"compute dataflow mat time: {time.time() - time_net_info} seconds")

        assert self.dmp_placedb.xl == 0 and self.dmp_placedb.yl == 0
        self.max_height = self.dmp_placedb.yh
        self.max_width  = self.dmp_placedb.xh
        self.port_to_net_dict = {}
        self.port_info = {}

        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        time_node_to_net_dict = time.time()
        print(f"get node to net dict time: {time_node_to_net_dict - time_macro_cluster_list} seconds")
        self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)
        time_node_id_to_name = time.time()
        print(f"get node id to name time: {time_node_id_to_name - time_node_to_net_dict} seconds")

        self.macro_pos = {}
        self.port_pos = []
        for node in self.node_info:
            raw_x = self.node_info[node]["raw_x"]
            raw_y = self.node_info[node]["raw_y"]
            pos_x, pos_y = self._real_to_grid(raw_x, raw_y)
            size_x = math.ceil(max(1, self.node_info[node]['x'] / self.ratio_x))
            size_y = math.ceil(max(1, self.node_info[node]['y'] / self.ratio_y))
            node_id = self.dmp_placedb.node_name2id_map[node]
            self.macro_pos[node_id] = (pos_x, pos_y, size_x, size_y)
        for node in self.port_indices:
            pos_x, pos_y = self._real_to_grid(self.dmp_placedb.node_x[node], self.dmp_placedb.node_y[node])
            self.port_pos.append([pos_x, pos_y])

        # convert name keys to ID keys
        self.node_to_net_dict = {self.dmp_placedb.node_name2id_map[node]: self.node_to_net_dict[node] for node in self.node_to_net_dict}
        for net in self.net_info:
            self.net_info[net]["nodes"] = {self.dmp_placedb.node_name2id_map[node]: self.net_info[net]["nodes"][node] for node in self.net_info[net]["nodes"]}
            self.net_info[net]["ports"] = {self.dmp_placedb.node_name2id_map[node]: self.net_info[net]["ports"][node] for node in self.net_info[net]["ports"]}

        self.node_info = {self.dmp_placedb.node_name2id_map[node]: self.node_info[node] for node in self.node_info}
        self.node_id_to_name = [self.dmp_placedb.node_name2id_map[node] for node in self.node_id_to_name]
    
    def _load_dmp_params(self):
        params = Params()
        if "superblue" in self.benchmark:
            params.load(os.path.join("DREAMPlace/test/iccad2015.ot", f'{self.benchmark}.json'))
        else:
            params.load(os.path.join("DREAMPlace/test/or_cases", f'{self.benchmark}.json'))
        return params

    def _real_to_grid(self, raw_x, raw_y):
        """Real -> natural (site/row) -> grid index."""
        return real_to_grid(
            raw_x, raw_y,
            self.ratio_x, self.ratio_y,
        )

    def _grid_to_real(self, grid_x, grid_y):
        """Grid index (with halo) -> natural site/row -> real (site/row aligned)."""
        return grid_to_real(
            grid_x, grid_y,
            self.ratio_x, self.ratio_y,
        )

    def init_dmp_db(self):

        return self.dmp_caller.init_db()

    def evaluate(self, macro_pos):
        if len(macro_pos) == 0:
            return np.inf

        self.apply(macro_pos=macro_pos)

        metric = self.dmp_placer(self.dmp_params, self.dmp_placedb)[-1]
        if isinstance(metric, list):
            gp_hpwl = metric[0][0].hpwl.item()
        else:
            gp_hpwl = metric.hpwl.item()

        self.results['placement'] = (self.dmp_placedb.node_x.copy(),
                                    self.dmp_placedb.node_y.copy())
        self.results['figure'] = copy.copy(self.dmp_placer.pos[0].data.clone().cpu().numpy())
        
        # Evaluate timing metrics (TNS and WNS) if timing optimization is enabled
        if self.args.use_timer_for_evaluation:
            tns, wns = self.evaluate_timing()
        else:
            tns, wns = 0, 0
        # tns, wns = 0, 0
       
        return gp_hpwl, tns, wns

    def evaluate_timing(self):
        """
        Evaluate timing metrics (TNS and WNS) using the timing operator.
        Returns:
            tuple: (tns, wns) timing metrics
        """ 
        # Get timing operator from the placer's op_collections
        timing_op = self.dmp_placer.op_collections.timing_op
        time_unit = timing_op.timer.time_unit()
        
        # Perform timing analysis on current placement
        # The timing operator takes the current position as input
        timing_op(self.dmp_placer.pos[0].data.clone().cpu())
        timing_op.timer.update_timing()
        
        # Report TNS and WNS
        # Note: OpenTimer considers early,late,rise,fall for tns/wns
        # The following values are normalized by time units
        tns = timing_op.timer.report_tns_elw(split=1) / (time_unit * 1e17)
        wns = timing_op.timer.report_wns(split=1) / (time_unit * 1e15)
        
        return tns, wns

    def apply(self, macro_pos):
        xl, yl = self.dmp_placedb.xl, self.dmp_placedb.yl
        xh, yh = self.dmp_placedb.xh, self.dmp_placedb.yh
        for node_id in macro_pos:
            pos_x, pos_y, _, _ = macro_pos[node_id]
            pos_x += self.args.halo
            pos_y += self.args.halo
            rx, ry = self._grid_to_real(pos_x, pos_y)
            self.dmp_placedb.node_x[node_id] = max(xl, min(xh, rx))
            self.dmp_placedb.node_y[node_id] = max(yl, min(yh, ry))

        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y)

        with th.no_grad():
            self.dmp_placer.pos[0].data.copy_(
                th.from_numpy(self.dmp_placer._initialize_position(self.dmp_params, self.dmp_placedb)).to(self.dmp_placer.device) )
    
    def save_placement(self, path):
        self.dmp_placedb.node_x[:] = self.results['placement'][0].copy()
        self.dmp_placedb.node_y[:] = self.results['placement'][1].copy()
        # unscale locations
        node_x, node_y = self.dmp_placedb.unscale_pl(self.dmp_params.shift_factor, 
                                                     self.dmp_params.scale_factor)
        # update raw database
        place_io.PlaceIOFunction.apply(self.dmp_placedb.rawdb, node_x, node_y)
        self.dmp_placedb.write(
            self.dmp_params,
            path
        )

    def plot(self, hpwl, figure_name):
        pos = self.results['figure']
        plot_placement(self.dmp_placer, self.dmp_params, pos, figure_name)
        
    def set_mp_hpwl(self, mp_hpwl):
        self.mp_hpwl = mp_hpwl
    
    def set_gp_hpwl(self, gp_hpwl):
        self.gp_hpwl = gp_hpwl
    
    def set_regularity(self, regularity):
        self.regularity = regularity

    def get_node_info(self):
        node_info = {}
        node_info_raw_id_name ={}
        for id, (macro_name, size_x, size_y, raw_x, raw_y) in enumerate(zip(self.macro_names, self.macro_size_x, self.macro_size_y, self.macro_x, self.macro_y)):
            node_info[macro_name] = {"id": id, "x": size_x, "y": size_y, "raw_x": raw_x, "raw_y": raw_y}
            node_info_raw_id_name[id] = macro_name
        
        return node_info, node_info_raw_id_name
    
    def get_net_info(self):
        """Build net_info only for nets connected to at least two macros, by iterating over macro pins (node2pin_map + pin2net_map) instead of all nets."""
        node2pin_map = self.dmp_placedb.node2pin_map
        pin2net_map = np.asarray(self.dmp_placedb.pin2net_map)

        # net_id -> { node_name: {x_offset, y_offset} } for macro nodes on that net
        net_to_macro_nodes = {}

        for macro_node_id in self.macros:
            pin_ids = node2pin_map[int(macro_node_id)]
            for pin_id in pin_ids:
                pin_id = int(pin_id)
                net_id = int(pin2net_map[pin_id])
                o_x = float(self.pin_offset_x[pin_id] - self.node_size_x[macro_node_id] / 2)
                o_y = float(self.pin_offset_y[pin_id] - self.node_size_y[macro_node_id] / 2)
                node_name = self.node_names[macro_node_id]
                if net_id not in net_to_macro_nodes:
                    net_to_macro_nodes[net_id] = {}
                net_to_macro_nodes[net_id][node_name] = {"x_offset": o_x, "y_offset": o_y}

        net_info = {}
        for net_id, nodes_dict in net_to_macro_nodes.items():
            if len(nodes_dict) <= 1:
                continue
            net_name = self.net_names[net_id]
            net_info[net_name] = {"nodes": nodes_dict, "ports": {}}

        net_cnt = 0
        for net_name in net_info:
            net_info[net_name]["id"] = net_cnt
            net_cnt += 1
        print("adjust net size = {}".format(len(net_info)))
        return net_info
    
    def community_partition(self):
        cell_graph_builder = GraphBuilder(self.dmp_placedb)
        cell_graph = cell_graph_builder.build_graph()

        plm = nk.community.PLM(cell_graph, gamma=self.args.gamma, par="none")
        plm.run()
        communities = plm.getPartition()
        
        macro_cluster_dict = {i: [] for i in range(len(communities))}
        for node_id, node in enumerate(self.node_names):
            if node in self.macro_names:
                module = communities[node_id]
                macro_cluster_dict[module].append(self.dmp_placedb.node_name2id_map[node])

        # Further split each community by macro size: group macros if width or height difference < 10%
        macro_cluster_list = []
        for macro_list in macro_cluster_dict.values():
            if len(macro_list) <= 1:
                if macro_list: macro_cluster_list.append(macro_list)
                continue
            sizes = [(self.dmp_placedb.node_size_x[m], self.dmp_placedb.node_size_y[m]) for m in macro_list]
            parent = list(range(len(macro_list)))
            def find(x):
                if parent[x] != x: parent[x] = find(parent[x])
                return parent[x]
            for i in range(len(macro_list)):
                for j in range(i + 1, len(macro_list)):
                    sx1, sy1, sx2, sy2 = sizes[i][0], sizes[i][1], sizes[j][0], sizes[j][1]
                    if (sx1 > 0 and sx2 > 0 and abs(sx1 - sx2) / max(sx1, sx2) < 0.1) or \
                       (sy1 > 0 and sy2 > 0 and abs(sy1 - sy2) / max(sy1, sy2) < 0.1):
                        parent[find(i)] = find(j)
            groups = {}
            for idx, m in enumerate(macro_list):
                groups.setdefault(find(idx), []).append(m)
            macro_cluster_list.extend(groups.values())
        return macro_cluster_list

    def get_node2type_map(self):
        # for dataflow
        gate2cell_map = self.timer.raw_timer.get_gate_cell_mapping()
        node2type_map = np.zeros(len(self.node_names), dtype=np.bool_)
        for node_id, node_name in enumerate(self.node_names):
            if node_name in self.macro_names:
                node2type_map[node_id] = True
            else:
                node_type = gate2cell_map.get(node_name, None)
                if node_type is not None:
                    if "DFF" in node_type:
                        node2type_map[node_id] = True
        return node2type_map

    def build_dataflow_graph(self, macros:list):
        graph_builder = DataflowGraphBuilder(self.dmp_placedb)
        if hasattr(self.args, "dataflow_graph_path"):
            graph_builder.load_graph(self.args.dataflow_graph_path)
            return graph_builder
            
        node2pin_map = self.dmp_placedb.node2pin_map
        pin2net_map = self.dmp_placedb.pin2net_map
        net2pin_map = self.dmp_placedb.net2pin_map
        pin2node_map = self.dmp_placedb.pin2node_map
        pin_direct = self.dmp_placedb.pin_direct.astype(np.str_)

        node2type_map = self.get_node2type_map()

        # get port
        port_names = self.node_names[
            self.num_movable_nodes + self.dmp_placedb.num_terminals: 
            self.num_movable_nodes + self.dmp_placedb.num_terminals + self.dmp_placedb.num_terminal_NIs
        ]
        port_ids = []
        pin_names = self.dmp_placedb.pin_names.astype(np.str_)
        for port_name in port_names:
            port_id = np.where(pin_names == port_name)[0].item()
            port_ids.append(port_id)

        # source node (connected to input port)
        source_nodes = set()
        for port_id in port_ids:
            if pin_direct[port_id] == "OUTPUT":
                port_net = pin2net_map[port_id]
                net_pins = net2pin_map[port_net]
                input_pins = net_pins[pin_direct[net_pins] == "INPUT"]
                input_nodes = pin2node_map[input_pins]
                source_nodes.update(input_nodes)

        def get_downstream_neighbors(node_id):
            neighbors = set()
            node_pin_ids = node2pin_map[node_id]
            node_pin_directs = pin_direct[node_pin_ids]
            nets = pin2net_map[node_pin_ids]
            net_pin_id_lists = net2pin_map[nets]
            for pin_dir, net_pin_ids in zip(node_pin_directs, net_pin_id_lists):
                if pin_dir != "OUTPUT":
                    continue
                net_pin_directs = pin_direct[net_pin_ids]
                input_mask = (net_pin_directs == "INPUT")
                if np.any(input_mask):
                    input_pins = net_pin_ids[input_mask]
                    input_nodes = pin2node_map[input_pins]
                    for nb in input_nodes:
                        nb = int(nb)
                        if nb != node_id:
                            neighbors.add(nb)
            return neighbors

        node2downstream_special_nodes_map = {}
        def dfs(node_id, last_special_node, path_set):
            if node_id in node2downstream_special_nodes_map:
                if last_special_node is not None:
                    for sink_node in node2downstream_special_nodes_map[node_id]:
                        if sink_node != last_special_node:
                            graph_builder.add_edge(last_special_node, sink_node)
                
                return node2downstream_special_nodes_map[node_id]

            if node_id in path_set:
                return []

            path_set.add(node_id)

            is_special = bool(node2type_map[node_id])
            next_last = last_special_node
            if is_special:
                node2downstream_special_nodes_map[node_id] = [node_id]
                if last_special_node is not None and node_id != last_special_node:
                    graph_builder.add_edge(last_special_node, node_id)
                next_last = node_id

            downstream_special_nodes = []
            for neighbor_node in get_downstream_neighbors(node_id):
                downstream_special_nodes.extend(dfs(neighbor_node, next_last, path_set))

            if node2downstream_special_nodes_map.get(node_id, None) is None:
                node2downstream_special_nodes_map[node_id] = list(set(downstream_special_nodes))

            path_set.remove(node_id)
            return node2downstream_special_nodes_map[node_id]

        def iterative_dfs(start_node_id, start_last_special_node):
            # Stack elements: (node_id, last_special_node, path_set, neighbors_iterator, downstream_special_nodes)
            stack = [(start_node_id, start_last_special_node, set(), None, [])]
            path_set = set()
            
            while stack:
                node_id, last_special_node, current_path, neighbors_iter, downstream_special_nodes = stack[-1]
                
                # First visit to this node
                if neighbors_iter is None:
                    # Check cache
                    if node_id in node2downstream_special_nodes_map:
                        if last_special_node is not None:
                            for sink_node in node2downstream_special_nodes_map[node_id]:
                                if sink_node != last_special_node:
                                    graph_builder.add_edge(last_special_node, sink_node)
                        stack.pop()
                        continue
                    
                    # Check for cycles
                    if node_id in path_set:
                        stack.pop()
                        continue
                    
                    # Enter node
                    path_set.add(node_id)
                    current_path.add(node_id)
                    
                    # Process special nodes
                    is_special = bool(node2type_map[node_id])
                    next_last = last_special_node
                    if is_special:
                        node2downstream_special_nodes_map[node_id] = [node_id]
                        if last_special_node is not None and node_id != last_special_node:
                            graph_builder.add_edge(last_special_node, node_id)
                        next_last = node_id
                    
                    # Get neighbors iterator
                    neighbors = list(get_downstream_neighbors(node_id))
                    stack[-1] = (node_id, last_special_node, current_path, iter(neighbors), [])
                    continue
                
                # Process neighbors
                try:
                    neighbor_node = next(neighbors_iter)
                    # Recursive call to neighbor (push to stack)
                    stack.append((neighbor_node, next_last, current_path.copy(), None, []))
                except StopIteration:
                    # All neighbors processed, compute downstream special nodes
                    if node_id not in node2downstream_special_nodes_map:
                        # Deduplicate and cache
                        unique_downstream = list(dict.fromkeys(downstream_special_nodes))
                        node2downstream_special_nodes_map[node_id] = unique_downstream
                    
                    # Exit node
                    path_set.remove(node_id)
                    current_path.remove(node_id)
                    
                    # Return result to parent node
                    if len(stack) > 1:
                        parent_node, parent_last, parent_path, parent_iter, parent_downstream = stack[-2]
                        parent_downstream.extend(node2downstream_special_nodes_map[node_id])
                        stack[-2] = (parent_node, parent_last, parent_path, parent_iter, parent_downstream)
                    
                    stack.pop()

        for i, s in enumerate(source_nodes):
            t = time.time()
            iterative_dfs(int(s), None)

            t_interval = time.time() - t
            if t_interval > 10:
                print(f"finish {i+1} in {t_interval} seconds")
        
        
        path = os.path.join("dataflow_graph", self.benchmark)
        if not os.path.exists(path):
            os.makedirs(path)
        graph_builder.save_graph(path)

        return graph_builder

    def compute_virtual_connections(self, k_max=2):
        graph_builder = self.build_dataflow_graph(self.macros)
        dataflow_graph = graph_builder.convert2undirected()

        macro2index_map = {node: i for i, node in enumerate(self.macros)}
        
        def compute_dataflow_mat():
            res = np.zeros((self.n_macro, self.n_macro), dtype=np.float32)
            count = np.zeros((self.n_macro, self.n_macro), dtype=np.int32)
            for i, m in enumerate(self.macros):
                m_int = graph_builder.internal_id(m)
                if m_int == -1:
                    continue
                
                t = time.time()
                queue = [(m_int, 0, [m_int])]
                while len(queue) > 0:
                    node_int, k, path_history = queue.pop(0)
                    
                    neighbors_int = list(dataflow_graph.iterNeighbors(node_int))
                    for neighbor_int in neighbors_int:
                        neighbor_ext = graph_builder.external_id(neighbor_int)
                        if neighbor_ext in macro2index_map:
                            if m != neighbor_ext:
                                if neighbor_int not in path_history:
                                    res[macro2index_map[m], macro2index_map[neighbor_ext]] += 1 / ( 2 ** (k+1))
                                    count[macro2index_map[m], macro2index_map[neighbor_ext]] += 1
                        if k+1 < k_max and neighbor_int not in path_history:
                            queue.append((neighbor_int, k+1, path_history + [neighbor_int]))
                
                # print(f"finish {i+1} in {time.time() - t} seconds")
            
            return res

        # for k_max in [2, 4]:
        res = compute_dataflow_mat()
        res = (res + res.T) / 2
        path = os.path.join("dataflow_info", self.benchmark, f"{k_max}")
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "dataflow_mat.npy"), res)
        with open(os.path.join(path, "macro2index_map.pkl"), "wb") as f:
            pickle.dump(macro2index_map, f)
        macro_name2index_map = self.convert_idmap2namemap(macro2index_map)
        with open(os.path.join(path, "macro_name2index_map.pkl"), "wb") as f:
            pickle.dump(macro_name2index_map, f)

        return res, macro2index_map

    def compute_hpwl_mat(self, macros, placedb):
        res = np.zeros((self.n_macro, self.n_macro), dtype=np.int32)
        macro2index_map = {node: i for i, node in enumerate(macros)}
        for net_name in self.net_info:
            nodes = self.net_info[net_name]["nodes"]
            for node1_name, node2_name in list(combinations(nodes, 2)):
                node1_id = placedb.node_name2id_map[node1_name]
                node2_id = placedb.node_name2id_map[node2_name]
                if node1_id in macro2index_map and node2_id in macro2index_map:
                    res[macro2index_map[node1_id], macro2index_map[node2_id]] += 1
                    res[macro2index_map[node2_id], macro2index_map[node1_id]] += 1

        path = os.path.join("hpwl_info", self.benchmark)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, "hpwl_mat.npy"), res)
        return res

    def convert_idmap2namemap(self, macro2index_map):
        macro_name2index_map = {}
        for macro_id, index in macro2index_map.items():
            macro_name = self.node_names[macro_id]
            macro_name2index_map[macro_name] = index
        return macro_name2index_map

        

    
def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict

def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}

    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])

    node_net_num_fea= {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)

    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        for node_name in node_info:
            if node_name not in candidates and node_name not in visited_node:
                candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
                else:
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*30000 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y']*1 +int(hash(v)%10000)*1e-6)
        else:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
                else:
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node])) 
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
        
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    
    return node_id_to_name_res


class GraphBuilder:
    def __init__(self, placedb):
        self.net_names = placedb.net_names
        self.net_weights = placedb.net_weights
        self.net2pin_map = placedb.net2pin_map

        self.pin2node_map = placedb.pin2node_map
        self.node_names = placedb.node_names

        self.graph = nk.Graph(n=len(self.node_names), weighted=True, directed=False)

    def add_nodes(self):
        pass

    def add_edges(self):
        node_name_to_index = {name: i for i, name in enumerate(self.node_names)}

        for net_index, pins in enumerate(self.net2pin_map):
            weight = self.net_weights[net_index] if net_index < len(self.net_weights) else 1
            connected_nodes = set()
            for pin_id in pins:
                node_index = self.pin2node_map[pin_id]
                connected_nodes.add(node_index)

            if len(connected_nodes) > 2:
                virtual_node_index = self.graph.addNode()
                for node_index in connected_nodes:
                    self.graph.addEdge(virtual_node_index, node_index, weight)
            else:
                connected_nodes = list(connected_nodes)
                if len(connected_nodes) == 2:
                    self.graph.addEdge(connected_nodes[0], connected_nodes[1], weight)

    def build_graph(self):
        self.add_nodes()
        self.add_edges()
        return self.graph


class DataflowGraphBuilder:
    def __init__(self, placedb, directed=True):
        self.node_names = placedb.node_names.astype(np.str_)
        self.ext2int = {}
        self.int2ext = []
        self.graph = nk.Graph(n=0, weighted=False, directed=directed)

    def internal_id(self, ext_id):
        return self.ext2int.get(ext_id, -1)

    def external_id(self, int_id):
        if int_id < len(self.int2ext):
            return self.int2ext[int_id]
        else:
            return -1

    def add_edge(self, ext_id1, ext_id2):
        def to_internal(ext_id):
            if ext_id not in self.ext2int:
                self.ext2int[ext_id] = self.graph.addNode()
                self.int2ext.append(ext_id)
                assert len(self.int2ext) - 1 == self.ext2int[ext_id], (len(self.int2ext[-1]) - 1, self.ext2int[ext_id])
            return self.ext2int[ext_id]
        
        int_id1 = to_internal(ext_id1)
        int_id2 = to_internal(ext_id2)
        self.graph.addEdge(int_id1, int_id2)

        # if len(self.ext2int) <= 10:
        #     degree1 = self.graph.degree(int_id1)
        #     degree2 = self.graph.degree(int_id2)
        #     in_degree1 = self.graph.degreeIn(int_id1)
        #     in_degree2 = self.graph.degreeIn(int_id2)
        #     out_degree1 = self.graph.degreeOut(int_id1)
        #     out_degree2 = self.graph.degreeOut(int_id2)
        #     print(f"Edge {len(self.ext2int)}: {self.node_names[ext_id1]} -> {self.node_names[ext_id2]}")
        #     print(f"  source {ext_id1}: degree={degree1}, in_degree={in_degree1}, out_degree={out_degree1}")
        #     print(f"  sink {ext_id2}: degree={degree2}, in_degree={in_degree2}, out_degree={out_degree2}")
        #     print(f"  validation: degree == out_degree? source: {degree1 == out_degree1}, sink: {degree2 == out_degree2}")
        #     print()

    def save_graph(self, path):
        writer = graphio.EdgeListWriter("\t", 0)  # 分隔符、首节点编号
        writer.write(self.graph, os.path.join(path, "dataflow_graph"))

        content = {
            "ext2int": self.ext2int,
            "int2ext": self.int2ext
        }

        with open(os.path.join(path, "dataflow_graph_map.pkl"), "wb") as f:
            pickle.dump(content, f)

    def load_graph(self, path):
        reader = graphio.EdgeListReader(separator="\t", firstNode=0, directed=self.graph.isDirected())
        self.graph = reader.read(os.path.join(path, "dataflow_graph"))
    
        with open(os.path.join(path, "dataflow_graph_map.pkl"), "rb") as f:
            content = pickle.load(f)
        self.ext2int = content["ext2int"]
        self.int2ext = content["int2ext"]

    def convert2undirected(self):
        undirected_graph = nk.Graph(n=self.graph.numberOfNodes(), weighted=False, directed=False)
        for u, v in self.graph.iterEdges():
            if not undirected_graph.hasEdge(u, v):
                undirected_graph.addEdge(u, v)
        return undirected_graph