import numpy as np
import logging
from collections import defaultdict

class NeighborFinder:
    def __init__(self, adj_list, uniform=False):

        self.logger = logging.getLogger(__name__)
        
        try:
            self.node_to_neighbors = defaultdict(list)
            self.node_to_edge_timestamps = defaultdict(list)

            for src, dst, ts in adj_list:
                self.node_to_neighbors[src].append(dst)
                self.node_to_edge_timestamps[src].append(ts)

                self.node_to_neighbors[dst].append(src)
                self.node_to_edge_timestamps[dst].append(ts)

            for node in self.node_to_neighbors:
                self.node_to_neighbors[node] = np.array(self.node_to_neighbors[node])
                self.node_to_edge_timestamps[node] = np.array(self.node_to_edge_timestamps[node])
            
            self.uniform = uniform
            
        except Exception as e:
            self.logger.error(f"Error initializing NeighborFinder: {str(e)}")
            raise
            
    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):

        try:
            assert len(src_idx_l) == len(cut_time_l), "Source indices and cut times must have same length"
            
            ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors), dtype=np.int32)
            ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors), dtype=np.float32)
            
            for i, (src, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
                neighbors = self.node_to_neighbors[src]
                timestamps = self.node_to_edge_timestamps[src]

                mask = timestamps <= cut_time
                neighbors = neighbors[mask]
                timestamps = timestamps[mask]
                
                if len(neighbors) > 0:
                    if self.uniform:
                        if len(neighbors) > num_neighbors:
                            indices = np.random.choice(len(neighbors), num_neighbors, replace=False)
                            ngh_node_batch[i] = neighbors[indices]
                            ngh_t_batch[i] = timestamps[indices]
                        else:
                            ngh_node_batch[i, :len(neighbors)] = neighbors
                            ngh_t_batch[i, :len(neighbors)] = timestamps
                    else:
                        sorted_indices = np.argsort(timestamps)
                        neighbors = neighbors[sorted_indices]
                        timestamps = timestamps[sorted_indices]
                        
                        if len(neighbors) > num_neighbors:
                            ngh_node_batch[i] = neighbors[-num_neighbors:]
                            ngh_t_batch[i] = timestamps[-num_neighbors:]
                        else:
                            ngh_node_batch[i, :len(neighbors)] = neighbors
                            ngh_t_batch[i, :len(neighbors)] = timestamps
            
            return ngh_node_batch, ngh_t_batch
            
        except Exception as e:
            self.logger.error(f"Error in get_temporal_neighbor: {str(e)}")
            return np.zeros((len(src_idx_l), num_neighbors), dtype=np.int32), \
                   np.zeros((len(src_idx_l), num_neighbors), dtype=np.float32) 