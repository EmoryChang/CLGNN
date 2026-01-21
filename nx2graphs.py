import os
import pandas as pd


def load_scores(fp):
    df = pd.read_csv(fp, names=['node_id', 'score'])
    return [int(n) for n in df['node_id']], df['score'].tolist()


def discover(data_dir, exclude=None, scores_subdir='scores'):
    ds = []
    sd = os.path.join(data_dir, scores_subdir)
    if not os.path.isdir(data_dir) or not os.path.isdir(sd): return ds
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith('.txt'): continue
        nm = fn[:-4]
        if exclude and nm in exclude: continue
        sf = os.path.join(sd, f'graph_{nm}_scores.csv')
        if os.path.isfile(sf): ds.append({'name': nm, 'graph': os.path.join(data_dir, fn), 'scores': sf})
    return ds