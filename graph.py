import numpy as np
import torch

from neighbor_finder import ImprovedNeighborFinder


def load_graph(filepath):
    src, dst, ts, idx = [], [], [], []
    with open(filepath) as f:
        for i, line in enumerate(f):
            p = line.strip().split()
            if len(p) < 3: continue
            if len(p) >= 4:
                src.append(int(p[0])); dst.append(int(p[1])); ts.append(float(p[3])); idx.append(i)
            else:
                src.append(int(p[0])); dst.append(int(p[1])); ts.append(float(p[2])); idx.append(i)
    return np.array(src, np.int32), np.array(dst, np.int32), np.array(ts, np.float64), np.array(idx, np.int32)


def build_ngh(src, dst, ts, idx):
    if len(src) == 0: return ImprovedNeighborFinder([[]])
    mx = max(src.max(), dst.max())
    adj = [[] for _ in range(mx+1)]
    for s, d, ei, t in zip(src, dst, idx, ts): adj[d].append((s, ei, t))
    return ImprovedNeighborFinder(adj)


def compute_node_features(src, dst, feat_size, feat_dim):
    in_deg = np.zeros(feat_size, dtype=np.float32)
    out_deg = np.zeros(feat_size, dtype=np.float32)
    for s in src:
        if s < feat_size: out_deg[s] += 1
    for d in dst:
        if d < feat_size: in_deg[d] += 1
    in_deg = np.log1p(in_deg)
    out_deg = np.log1p(out_deg)
    total_deg = in_deg + out_deg
    feats = np.zeros((feat_size, feat_dim), dtype=np.float32)
    feats[:, 0] = in_deg
    feats[:, 1] = out_deg
    feats[:, 2] = total_deg
    return feats


def update_model_features(model, src, dst, feat_size, feat_dim):
    model_feat_size = model.n_feat_th.shape[0]
    actual_feat_size = max(feat_size, model_feat_size)
    feats = compute_node_features(src, dst, actual_feat_size, feat_dim)
    model.n_feat_th.data.copy_(torch.from_numpy(feats).to(model.n_feat_th.device))


def scan_max_node(infos):
    mx = 0
    for info in infos:
        s, d, _, _ = load_graph(info['graph'])
        if len(s) > 0: mx = max(mx, int(s.max()), int(d.max()))
    return mx + 1