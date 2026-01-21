import numpy as np
import numba
import logging


@numba.njit(cache=True)
def _find_neighbors_recent(src_idx_l, cut_time_l, num_neighbors,
                           csr_offsets, csr_nodes, csr_times, num_nodes_csr):
    batch_size = len(src_idx_l)
    out_nodes = np.zeros((batch_size, num_neighbors), dtype=np.int32)
    out_times = np.zeros((batch_size, num_neighbors), dtype=np.float64)
    for i in range(batch_size):
        src = src_idx_l[i]
        cut_t = cut_time_l[i]
        if src < 0 or src >= num_nodes_csr:
            continue
        start = csr_offsets[src]
        end = csr_offsets[src + 1]
        if start == end:
            continue
        lo = start
        hi = end
        while lo < hi:
            mid = (lo + hi) // 2
            if csr_times[mid] < cut_t:
                lo = mid + 1
            else:
                hi = mid
        n_avail = lo - start
        if n_avail == 0:
            continue
        if n_avail >= num_neighbors:
            offset = lo - num_neighbors
            for j in range(num_neighbors):
                out_nodes[i, j] = csr_nodes[offset + j]
                out_times[i, j] = csr_times[offset + j]
        else:
            for j in range(n_avail):
                out_nodes[i, j] = csr_nodes[start + j]
                out_times[i, j] = csr_times[start + j]
            if n_avail > 0:
                for j in range(n_avail, num_neighbors):
                    pick = np.random.randint(0, n_avail)
                    out_nodes[i, j] = csr_nodes[start + pick]
                    out_times[i, j] = csr_times[start + pick]
    return out_nodes, out_times


@numba.njit(cache=True)
def _find_neighbors_uniform(src_idx_l, cut_time_l, num_neighbors,
                             csr_offsets, csr_nodes, csr_times, num_nodes_csr):
    batch_size = len(src_idx_l)
    out_nodes = np.zeros((batch_size, num_neighbors), dtype=np.int32)
    out_times = np.zeros((batch_size, num_neighbors), dtype=np.float64)
    for i in range(batch_size):
        src = src_idx_l[i]
        cut_t = cut_time_l[i]
        if src < 0 or src >= num_nodes_csr:
            continue
        start = csr_offsets[src]
        end = csr_offsets[src + 1]
        if start == end:
            continue
        lo = start
        hi = end
        while lo < hi:
            mid = (lo + hi) // 2
            if csr_times[mid] < cut_t:
                lo = mid + 1
            else:
                hi = mid
        n_avail = lo - start
        if n_avail == 0:
            continue
        if n_avail >= num_neighbors:
            for j in range(num_neighbors):
                pick = np.random.randint(0, n_avail)
                out_nodes[i, j] = csr_nodes[start + pick]
                out_times[i, j] = csr_times[start + pick]
        else:
            for j in range(n_avail):
                out_nodes[i, j] = csr_nodes[start + j]
                out_times[i, j] = csr_times[start + j]
            for j in range(n_avail, num_neighbors):
                pick = np.random.randint(0, n_avail)
                out_nodes[i, j] = csr_nodes[start + pick]
                out_times[i, j] = csr_times[start + pick]
    return out_nodes, out_times


@numba.njit(cache=True)
def _batch_path_counts_jit(ngh_nodes, ngh_times, csr_offsets, csr_times, num_nodes_csr):
    batch_size = ngh_nodes.shape[0]
    num_neighbors = ngh_nodes.shape[1]
    result = np.zeros((batch_size, num_neighbors), dtype=np.float64)
    for i in range(batch_size):
        for j in range(num_neighbors):
            node = ngh_nodes[i, j]
            edge_t = ngh_times[i, j]
            if node == 0 and edge_t == 0.0:
                continue
            if node < 0 or node >= num_nodes_csr:
                continue
            start = csr_offsets[node]
            end = csr_offsets[node + 1]
            if start == end:
                continue
            lo = start
            hi = end
            while lo < hi:
                mid = (lo + hi) // 2
                if csr_times[mid] <= edge_t:
                    lo = mid + 1
                else:
                    hi = mid
            result[i, j] = float(end - lo)
    return result


class ImprovedNeighborFinder:
    def __init__(self, adj_list, uniform=False):
        self.uniform = uniform
        self.logger = logging.getLogger(__name__)
        self._out_timestamps = {}
        all_nodes_list = []
        all_times_list = []
        num_nodes = len(adj_list)
        offsets = np.zeros(num_nodes + 1, dtype=np.int64)
        for node, neighbors in enumerate(adj_list):
            if not isinstance(neighbors, list) or len(neighbors) == 0:
                continue
            valid = []
            for n in neighbors:
                if len(n) == 3:
                    valid.append((int(n[0]), float(n[2])))
            if not valid:
                continue
            valid.sort(key=lambda x: x[1])
            nodes_arr = np.array([v[0] for v in valid], dtype=np.int32)
            times_arr = np.array([v[1] for v in valid], dtype=np.float64)
            all_nodes_list.append(nodes_arr)
            all_times_list.append(times_arr)
            offsets[node + 1] = len(nodes_arr)
            self._out_timestamps[node] = times_arr.copy()
        for i in range(1, num_nodes + 1):
            offsets[i] += offsets[i - 1]
        total_edges = int(offsets[num_nodes])
        if total_edges > 0:
            self._csr_nodes = np.zeros(total_edges, dtype=np.int32)
            self._csr_times = np.zeros(total_edges, dtype=np.float64)
            pos = 0
            for nodes_arr, times_arr in zip(all_nodes_list, all_times_list):
                length = len(nodes_arr)
                self._csr_nodes[pos:pos + length] = nodes_arr
                self._csr_times[pos:pos + length] = times_arr
                pos += length
        else:
            self._csr_nodes = np.zeros(0, dtype=np.int32)
            self._csr_times = np.zeros(0, dtype=np.float64)
        self._csr_offsets = offsets
        self._num_nodes_csr = num_nodes
        self._warmup_done = False

    def _warmup_jit(self):
        if self._warmup_done:
            return
        dummy_src = np.array([0], dtype=np.int32)
        dummy_t = np.array([0.0], dtype=np.float64)
        _find_neighbors_recent(dummy_src, dummy_t, 1,
                               self._csr_offsets, self._csr_nodes,
                               self._csr_times, self._num_nodes_csr)
        _find_neighbors_uniform(dummy_src, dummy_t, 1,
                                self._csr_offsets, self._csr_nodes,
                                self._csr_times, self._num_nodes_csr)
        self._warmup_done = True

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        assert len(src_idx_l) == len(cut_time_l)
        self._warmup_jit()
        src_arr = np.asarray(src_idx_l, dtype=np.int32)
        time_arr = np.asarray(cut_time_l, dtype=np.float64)
        if self.uniform:
            out_nodes, out_times = _find_neighbors_uniform(
                src_arr, time_arr, num_neighbors,
                self._csr_offsets, self._csr_nodes,
                self._csr_times, self._num_nodes_csr)
        else:
            out_nodes, out_times = _find_neighbors_recent(
                src_arr, time_arr, num_neighbors,
                self._csr_offsets, self._csr_nodes,
                self._csr_times, self._num_nodes_csr)
        return out_nodes, out_times.astype(np.float32)

    def batch_path_counts(self, ngh_node_batch, ngh_t_batch):
        self._warmup_jit()
        nodes = np.asarray(ngh_node_batch, dtype=np.int32)
        times = np.asarray(ngh_t_batch, dtype=np.float64)
        return _batch_path_counts_jit(
            nodes, times,
            self._csr_offsets, self._csr_times, self._num_nodes_csr)

    def get_outgoing_timestamps(self, node_id):
        return self._out_timestamps.get(int(node_id), np.array([]))

    def preprocess(self, src_idx_l, cut_time_l, num_layers, num_neighbors):
        pass