import numpy as np
import logging
from collections import defaultdict


class ImprovedNeighborFinder:
    def __init__(self, adj_list, uniform=False):
        self.uniform = uniform
        self.logger = logging.getLogger(__name__)
        
        try:
            self.node_to_neighbors = defaultdict(list)
            for node, neighbors in enumerate(adj_list):
                if not isinstance(neighbors, list):
                    self.logger.warning(f"Invalid neighbors for node {node}: {neighbors}")
                    continue
                    
                for neighbor in neighbors:
                    try:
                        if len(neighbor) != 3:
                            self.logger.warning(f"Invalid neighbor tuple for node {node}: {neighbor}")
                            continue
                            
                        dst_node, timestamp, edge_feat = neighbor
                        self.node_to_neighbors[node].append((dst_node, timestamp, edge_feat))
                    except Exception as e:
                        self.logger.error(f"Error processing neighbor {neighbor} for node {node}: {str(e)}")
                        continue

            for node in self.node_to_neighbors:
                self.node_to_neighbors[node].sort(key=lambda x: x[1])
                
            self.logger.info(f"Initialized NeighborFinder with {len(self.node_to_neighbors)} nodes")
            
        except Exception as e:
            self.logger.error(f"Error initializing NeighborFinder: {str(e)}")
            raise
            
    def find_before(self, src_idx, cut_time):
        try:
            if src_idx not in self.node_to_neighbors:
                self.logger.warning(f"Node {src_idx} not found in graph")
                return []
                
            neighbors = self.node_to_neighbors[src_idx]
            if not neighbors:
                return []

            left, right = 0, len(neighbors) - 1
            insert_idx = len(neighbors)
            
            while left <= right:
                mid = (left + right) // 2
                if neighbors[mid][1] >= cut_time:
                    insert_idx = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            return neighbors[:insert_idx]
            
        except Exception as e:
            self.logger.error(f"Error in find_before for node {src_idx}: {str(e)}")
            return []
            
    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        try:
            assert len(src_idx_l) == len(cut_time_l), \
                "Source indices and cutoff times must have same length"
            
            batch_size = len(src_idx_l)

            neighbor_nodes = np.zeros((batch_size, num_neighbors), dtype=np.int32)
            neighbor_times = np.zeros((batch_size, num_neighbors), dtype=np.float32)
            
            for batch_idx, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
                try:
                    neighbors = self.find_before(src_idx, cut_time)
                    if not neighbors:
                        continue

                    if len(neighbors) > num_neighbors:
                        if self.uniform:
                            sampled_idx = np.random.choice(
                                len(neighbors), num_neighbors, replace=False
                            )
                            sampled = [neighbors[i] for i in sampled_idx]
                        else:
                            sampled = neighbors[-num_neighbors:]
                    else:
                        sampled = neighbors
                        if len(sampled) < num_neighbors:
                            num_repeats = num_neighbors - len(sampled)
                            if len(sampled) > 0:
                                repeat_idx = np.random.choice(len(sampled), num_repeats)
                                sampled.extend([sampled[i] for i in repeat_idx])

                    for neighbor_idx, (node, timestamp, _) in enumerate(sampled[:num_neighbors]):
                        neighbor_nodes[batch_idx, neighbor_idx] = node
                        neighbor_times[batch_idx, neighbor_idx] = timestamp
                        
                except Exception as e:
                    self.logger.error(f"Error processing node {src_idx} in batch: {str(e)}")
                    continue
            
            return neighbor_nodes, neighbor_times
            
        except Exception as e:
            self.logger.error(f"Error in get_temporal_neighbor: {str(e)}")
            return (
                np.zeros((batch_size, num_neighbors), dtype=np.int32),
                np.zeros((batch_size, num_neighbors), dtype=np.float32)
            )
            
    def preprocess(self, src_idx_l, cut_time_l, num_layers, num_neighbors):
        try:
            pass
        except Exception as e:
            self.logger.error(f"Error in preprocess: {str(e)}")
            pass 