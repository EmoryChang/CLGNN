import json
import pickle
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from graph import NeighborFinder


def preprocess(data_name):
    u_list, i_list, ts_list = [], [], []
    idx_list = []

    with open(data_name) as f:
        # s = next(f)
        # print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(' ')
            # values = [v.split(' ') for v in e]
            u = int(e[0])
            i = int(e[1])
            ts_float = float(e[2])
            ts = int(ts_float)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(idx)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})




def compute_P(df, delta=None, save_path=None):

    T_out = defaultdict(list)
    for v, t in zip(df['i'], df['ts']):
        T_out[v].append(t)
    for v in T_out:
        T_out[v].sort()

    def count_valid_future_ts(v, t):
        times = T_out.get(v, [])
        if delta is None:
            return sum(1 for t_v in times if t_v > t)
        else:
            return sum(1 for t_v in times if 0 < t_v - t <= delta)

    df['P'] = [count_valid_future_ts(v, t) for v, t in zip(df['i'], df['ts'])]

    df = df.sort_values('idx').reset_index(drop=True)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
def run(data_name, delta=None):

    raw_path = f'./data/test/Real/{data_name}.txt'
    out_dir = f'./data/test/Real/processed/'
    os.makedirs(out_dir, exist_ok=True)
    out_df_path = os.path.join(out_dir, f'ml_{data_name}.csv')
    out_node_feat_path = os.path.join(out_dir, f'ml_{data_name}_node.npy')


    df = preprocess(raw_path)

    df = compute_P(df, delta=delta)

    df.to_csv(out_df_path, index=False)

    num_nodes = max(df['u'].max(), df['i'].max()) + 1
    feat_dim = 128
    node_feat = np.zeros((num_nodes, feat_dim)) 
    np.save(out_node_feat_path, node_feat)

    print(f"edge include P: {out_df_path}")
    print(f"node feature: {out_node_feat_path}")

run('edit-sewiki')
