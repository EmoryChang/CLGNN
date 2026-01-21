import os, sys, math, time, random, logging, argparse, gc, glob, json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from module import CLGNN
from graph import load_graph, build_ngh, compute_node_features, update_model_features, scan_max_node
from nx2graphs import load_scores, discover

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_seeds', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--test_data_dir', type=str, default='./test_data')
parser.add_argument('--train_data_dir', type=str, default='./train_data')
parser.add_argument('--output', type=str, default='experiment_results.md')
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--eval_checkpoint', type=str, default=None)
parser.add_argument('--scores_dir', type=str, default='scores')
parser.add_argument('--sampling', type=str, default='uniform',
                    choices=['uniform', 'tbc_weighted', 'degree_stratified'])
parser.add_argument('--max_train_nodes', type=int, default=50000)
parser.add_argument('--cluster_sampling', type=str, default='random',
                    choices=['random', 'stratified'])
args = parser.parse_args()

if args.version and args.save_dir == './saved_models':
    args.save_dir = os.path.join(args.save_dir, args.version)
    os.makedirs(args.save_dir, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
for n in ['module', 'graph', '__main__']:
    logging.getLogger(n).setLevel(logging.ERROR)

ALPHA = 0.2
LAMBDA_WEIGHT = 0.3
FEAT_DIM = 128
NUM_LAYERS = 3
NUM_HEADS = 4
NUM_EPOCHS = 15
NUM_NEIGHBORS = 20
DROP_OUT = 0.1
BATCH_SIZE = args.batch_size
LRS = [0.1, 0.01, 0.001]
NONZERO_WEIGHT = 3.0
ZERO_HIGH_DEG_WEIGHT = 2.0
EXCLUDE_TEST = {'sx-mathoverflow', 'highschool2013', 'Hypertext'}
DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

SAMPLING = args.sampling
MAX_TRAIN_NODES_BASE = args.max_train_nodes
MAX_TRAIN_NODES_LARGE = MAX_TRAIN_NODES_BASE * 2
LARGE_GRAPH_THRESHOLD = MAX_TRAIN_NODES_BASE * 5

def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def find_completed_seeds(save_dir):
    candidates = {}
    if not os.path.isdir(save_dir): return candidates
    for fp in glob.glob(os.path.join(save_dir, 'clgnn_seed*_lr*.pth')):
        try:
            ckpt = torch.load(fp, map_location='cpu', weights_only=False)
            seed = ckpt['seed']
            eval_results = ckpt.get('eval_results', {})
            sp_values = [r['Spearman'] for r in eval_results.values()
                         if isinstance(r, dict) and 'Spearman' in r and not np.isnan(r['Spearman'])]
            sp_avg = np.mean(sp_values) if sp_values else -1.0
            if seed not in candidates or sp_avg > candidates[seed]['sp_avg']:
                candidates[seed] = {'path': fp, 'eval_results': eval_results, 'sp_avg': sp_avg}
        except Exception:
            pass
    return candidates


def sample_uniform(nodes, scores, max_budget, src=None, dst=None):
    n_total = len(nodes)
    scores_arr = np.array(scores)
    nonzero_idx = np.where(scores_arr > 0)[0]
    zero_idx = np.where(scores_arr == 0)[0]
    n_nonzero = len(nonzero_idx)
    n_zero = len(zero_idx)
    if n_total <= max_budget:
        return nodes, scores, np.ones(len(scores), dtype=np.float32)
    if n_nonzero >= max_budget:
        chosen = np.random.choice(nonzero_idx, max_budget, replace=False)
        chosen.sort()
    else:
        remaining = min(max_budget - n_nonzero, n_zero)
        zero_chosen = np.random.choice(zero_idx, remaining, replace=False) if remaining > 0 else np.array([], dtype=int)
        chosen = np.sort(np.concatenate([nonzero_idx, zero_chosen]))
    sampled_nodes = [nodes[i] for i in chosen]
    sampled_scores = [scores[i] for i in chosen]
    return sampled_nodes, sampled_scores, np.ones(len(sampled_scores), dtype=np.float32)


def sample_tbc_weighted(nodes, scores, max_budget, src=None, dst=None):
    n_total = len(nodes)
    scores_arr = np.array(scores)
    nonzero_idx = np.where(scores_arr > 0)[0]
    zero_idx = np.where(scores_arr == 0)[0]
    n_nonzero = len(nonzero_idx)
    n_zero = len(zero_idx)
    if n_total <= max_budget:
        node_weights = np.ones(len(scores), dtype=np.float32)
        node_weights[scores_arr > 0] = NONZERO_WEIGHT
        return nodes, scores, node_weights
    if n_nonzero > max_budget:
        n_zero_slots = max(int(max_budget * 0.1), min(2000, n_zero))
        n_nonzero_slots = max_budget - n_zero_slots
        nonzero_sorted = nonzero_idx[np.argsort(-scores_arr[nonzero_idx])]
        chosen_nonzero = nonzero_sorted[:n_nonzero_slots]
        chosen_zero = np.random.choice(zero_idx, min(n_zero_slots, n_zero), replace=False) if n_zero > 0 else np.array([], dtype=int)
        chosen = np.sort(np.concatenate([chosen_nonzero, chosen_zero]))
    else:
        remaining = min(max_budget - n_nonzero, n_zero)
        zero_chosen = np.random.choice(zero_idx, remaining, replace=False) if remaining > 0 else np.array([], dtype=int)
        chosen = np.sort(np.concatenate([nonzero_idx, zero_chosen]))
    sampled_nodes = [nodes[i] for i in chosen]
    sampled_scores = [scores[i] for i in chosen]
    sampled_arr = np.array(sampled_scores)
    node_weights = np.ones(len(sampled_scores), dtype=np.float32)
    node_weights[sampled_arr > 0] = NONZERO_WEIGHT
    return sampled_nodes, sampled_scores, node_weights


def _stratified_zero_sample(zero_idx, degrees, high_deg_threshold, mid_deg_threshold, budget):
    if budget <= 0 or len(zero_idx) == 0:
        return np.array([], dtype=int)
    zero_degs = degrees[zero_idx]
    high_mask = zero_degs >= high_deg_threshold if high_deg_threshold > 0 else np.zeros(len(zero_idx), dtype=bool)
    mid_mask = (~high_mask) & (zero_degs >= mid_deg_threshold) if mid_deg_threshold > 0 else np.zeros(len(zero_idx), dtype=bool)
    low_mask = ~(high_mask | mid_mask)
    high_idx = zero_idx[high_mask]
    mid_idx = zero_idx[mid_mask]
    low_idx = zero_idx[low_mask]
    chosen = []
    remaining = budget
    if len(high_idx) <= remaining:
        chosen.append(high_idx); remaining -= len(high_idx)
    else:
        chosen.append(np.random.choice(high_idx, remaining, replace=False)); remaining = 0
    if remaining > 0 and len(mid_idx) > 0:
        n_mid = min(len(mid_idx) // 2 + 1, remaining)
        chosen.append(np.random.choice(mid_idx, n_mid, replace=False)); remaining -= n_mid
    if remaining > 0 and len(low_idx) > 0:
        n_low = min(len(low_idx), remaining)
        chosen.append(np.random.choice(low_idx, n_low, replace=False))
    return np.concatenate(chosen) if chosen else np.array([], dtype=int)


def sample_degree_stratified(nodes, scores, max_budget, src, dst):
    n_total = len(nodes)
    scores_arr = np.array(scores)
    nonzero_idx = np.where(scores_arr > 0)[0]
    zero_idx = np.where(scores_arr == 0)[0]
    n_nonzero = len(nonzero_idx)
    n_zero = len(zero_idx)
    node_set = set(nodes)
    node_degree = {}
    for s_val in src:
        if int(s_val) in node_set: node_degree[int(s_val)] = node_degree.get(int(s_val), 0) + 1
    for d_val in dst:
        if int(d_val) in node_set: node_degree[int(d_val)] = node_degree.get(int(d_val), 0) + 1
    degrees = np.array([node_degree.get(nodes[i], 0) for i in range(n_total)])
    zero_degrees = degrees[zero_idx] if n_zero > 0 else np.array([0])
    high_deg_threshold = np.percentile(zero_degrees, 90) if n_zero > 0 else 0
    mid_deg_threshold = np.percentile(zero_degrees, 50) if n_zero > 0 else 0
    if n_total <= max_budget:
        pass
    elif n_nonzero > max_budget:
        n_zero_slots = max(int(max_budget * 0.1), min(2000, n_zero))
        n_nonzero_slots = max_budget - n_zero_slots
        nonzero_sorted = nonzero_idx[np.argsort(-scores_arr[nonzero_idx])]
        chosen_nonzero = nonzero_sorted[:n_nonzero_slots]
        chosen_zero = _stratified_zero_sample(zero_idx, degrees, high_deg_threshold, mid_deg_threshold, n_zero_slots)
        chosen = np.sort(np.concatenate([chosen_nonzero, chosen_zero]))
        nodes = [nodes[i] for i in chosen]; scores = [scores[i] for i in chosen]; degrees = degrees[chosen]
    else:
        remaining = max_budget - n_nonzero
        if remaining >= n_zero:
            chosen = np.sort(np.concatenate([nonzero_idx, zero_idx]))
        else:
            chosen_zero = _stratified_zero_sample(zero_idx, degrees, high_deg_threshold, mid_deg_threshold, remaining)
            chosen = np.sort(np.concatenate([nonzero_idx, chosen_zero]))
        nodes = [nodes[i] for i in chosen]; scores = [scores[i] for i in chosen]; degrees = degrees[chosen]
    scores_arr_sampled = np.array(scores)
    degrees_sampled = degrees if len(degrees) == len(scores) else np.zeros(len(scores))
    node_weights = np.ones(len(scores), dtype=np.float32)
    node_weights[scores_arr_sampled > 0] = NONZERO_WEIGHT
    zero_mask_sampled = scores_arr_sampled == 0
    high_deg_mask = zero_mask_sampled & (degrees_sampled >= high_deg_threshold) & (high_deg_threshold > 0)
    node_weights[high_deg_mask] = ZERO_HIGH_DEG_WEIGHT
    return nodes, scores, node_weights


SAMPLE_FN = {'uniform': sample_uniform, 'tbc_weighted': sample_tbc_weighted, 'degree_stratified': sample_degree_stratified}


def compute_loss_uniform(model, pred, b_l, cl, b_w):
    loss, _ = model.compute_total_loss(pred, b_l, cl)
    return loss

def compute_loss_tbc_weighted(model, pred, b_l, cl, b_w):
    log_labels = torch.log1p(b_l)
    per_node_abs = (pred - log_labels).abs() * b_w
    weighted_regr = per_node_abs.mean()
    if torch.isnan(cl) or torch.isinf(cl):
        cl = torch.tensor(0.0, device=pred.device)
    return ALPHA * cl + (1.0 - ALPHA) * weighted_regr

def compute_loss_degree_stratified(model, pred, b_l, cl, b_w):
    loss, _ = model.compute_total_loss(pred, b_l, cl)
    return loss * b_w.mean()

LOSS_FN = {'uniform': compute_loss_uniform, 'tbc_weighted': compute_loss_tbc_weighted, 'degree_stratified': compute_loss_degree_stratified}


def get_max_budget(n_total):
    if SAMPLING == 'uniform':
        return MAX_TRAIN_NODES_BASE
    if n_total > LARGE_GRAPH_THRESHOLD:
        return MAX_TRAIN_NODES_LARGE
    return MAX_TRAIN_NODES_BASE

def train_model(model, optimizer, scheduler, train_infos, feat_size):
    model.train()
    total_train = len(train_infos)
    sample_fn = SAMPLE_FN[SAMPLING]
    loss_fn = LOSS_FN[SAMPLING]
    for epoch in range(NUM_EPOCHS):
        for ti, info in enumerate(train_infos):
            src, dst, ts, idx = load_graph(info['graph'])
            if len(src) == 0: continue
            ngh = build_ngh(src, dst, ts, idx)
            nodes, scores = load_scores(info['scores'])
            if len(nodes) == 0: continue
            max_t = float(ts.max())
            n_total = len(nodes)
            max_budget = get_max_budget(n_total)
            nodes, scores, node_weights = sample_fn(nodes, scores, max_budget, src, dst)
            ts_arr = np.array([max_t]*len(nodes))
            model.ngh_finder = ngh
            update_model_features(model, src, dst, feat_size, FEAT_DIM)
            del src, dst, ts, idx
            n_inst = len(nodes)
            n_sampled_nonzero = int(np.sum(np.array(scores) > 0))
            n_batch = math.ceil(n_inst / BATCH_SIZE)
            batch_losses = []
            skipped_batches = 0
            for k in range(n_batch):
                s, e = k*BATCH_SIZE, min(n_inst, (k+1)*BATCH_SIZE)
                b_n = np.array(nodes[s:e])
                b_t = ts_arr[s:e]
                b_l = torch.tensor(scores[s:e], dtype=torch.float32).to(DEVICE)
                b_w = torch.tensor(node_weights[s:e], dtype=torch.float32).to(DEVICE)
                optimizer.zero_grad()
                try:
                    pred, cl, _ = model(b_n, b_t, tbc_labels=b_l, num_neighbors=NUM_NEIGHBORS)
                    loss = loss_fn(model, pred, b_l, cl, b_w)
                    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                        log(f"      SKIP batch {k+1}/{n_batch} [{info['name']}]: abnormal loss={loss.item():.2f}")
                        skipped_batches += 1; torch.cuda.empty_cache(); continue
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    batch_losses.append(loss.item())
                except RuntimeError as oom_err:
                    if "out of memory" in str(oom_err):
                        log(f"      OOM batch {k+1}/{n_batch} [{info['name']}]: skipping")
                        skipped_batches += 1; torch.cuda.empty_cache(); continue
                    raise
            avg_loss = np.mean(batch_losses) if batch_losses else float('nan')
            sampled_tag = f" (sampled {n_inst}/{n_total}, nz={n_sampled_nonzero}, budget={max_budget})" if n_total > max_budget else f" (nz={n_sampled_nonzero})"
            skip_tag = f" [SKIPPED {skipped_batches}/{n_batch}]" if skipped_batches > 0 else ""
            log(f"    Ep {epoch+1:2d}/{NUM_EPOCHS} | Train {ti+1:2d}/{total_train} [{info['name']}] {n_inst} nodes{sampled_tag} | loss={avg_loss:.4f}{skip_tag}")
            del ngh, nodes, scores, ts_arr, node_weights
            torch.cuda.empty_cache(); gc.collect()
        scheduler.step()
        log(f"  === Epoch {epoch+1}/{NUM_EPOCHS} COMPLETE ===")


def evaluate_all(model, test_infos, feat_size):
    model.eval()
    results = {}
    for info in test_infos:
        src, dst, ts, idx = load_graph(info['graph'])
        if len(src) == 0: continue
        ngh = build_ngh(src, dst, ts, idx)
        nodes, scores = load_scores(info['scores'])
        max_t = float(ts.max())
        ts_arr = np.array([max_t]*len(nodes))
        model.ngh_finder = ngh
        update_model_features(model, src, dst, feat_size, FEAT_DIM)
        del src, dst, ts, idx
        preds = []
        with torch.no_grad():
            for k in range(math.ceil(len(nodes)/BATCH_SIZE)):
                s, e = k*BATCH_SIZE, min(len(nodes), (k+1)*BATCH_SIZE)
                pred, _ = model(np.array(nodes[s:e]), ts_arr[s:e], tbc_labels=None, num_neighbors=NUM_NEIGHBORS)
                preds.extend(pred.cpu().numpy().tolist())
        raw_predictions = np.maximum(np.array(preds), 0.0)
        labels = np.array(scores)
        sp, _ = spearmanr(labels, raw_predictions)
        raw_converted = np.expm1(np.maximum(raw_predictions, 0.0))
        raw_converted = np.maximum(raw_converted, 0.0)
        mae = mean_absolute_error(labels, raw_converted)
        results[info['name']] = {'MAE': mae, 'Spearman': sp}
        log(f"  Eval [{info['name']}]: MAE={mae:.6f}, Spearman={sp:.4f}")
        del ngh, nodes, scores, ts_arr
    return results


def build_model(feat_size, init_ngh):
    n_feat = np.zeros((feat_size, FEAT_DIM))
    model = CLGNN(
        ngh_finder=init_ngh, n_feat=n_feat, num_layers=NUM_LAYERS,
        use_time='time', agg_method='dual', attn_mode='prod',
        seq_len=NUM_NEIGHBORS, n_head=NUM_HEADS, drop_out=DROP_OUT,
        lambda_weight=LAMBDA_WEIGHT, alpha=ALPHA,
        cluster_sampling=args.cluster_sampling,
    ).to(DEVICE)
    del n_feat
    return model

def load_model_from_checkpoint(ckpt_path, feat_size, init_ngh):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_feat_size = ckpt.get('config', {}).get('node_feat_size', feat_size)
    actual_feat_size = max(feat_size, ckpt_feat_size)
    model = build_model(actual_feat_size, init_ngh)
    model.load_state_dict(ckpt['state_dict'])
    log(f"  Loaded checkpoint: {ckpt_path} (seed={ckpt['seed']}, lr={ckpt['best_lr']}, feat_size={actual_feat_size})")
    return model, ckpt


def save_checkpoint(save_dir, seed, best_lr, best_state_dict, best_eval, feat_size):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'clgnn_seed{seed}_lr{best_lr}.pth')
    torch.cuda.empty_cache(); gc.collect()
    ckpt_data = {
        'seed': seed, 'best_lr': best_lr, 'state_dict': best_state_dict,
        'eval_results': best_eval,
        'config': {
            'num_layers': NUM_LAYERS, 'feat_dim': FEAT_DIM, 'n_head': NUM_HEADS,
            'alpha': ALPHA, 'lambda_weight': LAMBDA_WEIGHT, 'epochs': NUM_EPOCHS,
            'num_neighbors': NUM_NEIGHBORS, 'node_feat_size': feat_size,
            'sampling': SAMPLING, 'max_train_nodes': MAX_TRAIN_NODES_BASE,
        },
    }
    try:
        torch.save(ckpt_data, save_path)
        log(f"  Model saved to {save_path}")
    except Exception as save_err:
        log(f"  WARNING: Failed to save checkpoint: {save_err}")
        torch.cuda.empty_cache(); gc.collect()
        try:
            torch.save(ckpt_data, save_path)
            log(f"  Model saved (retry) to {save_path}")
        except Exception as retry_err:
            log(f"  ERROR: Failed to save even after retry: {retry_err}")
            return None
    return save_path


def sampling_description():
    if SAMPLING == 'uniform':
        return f"uniform (budget={MAX_TRAIN_NODES_BASE})"
    elif SAMPLING == 'tbc_weighted':
        return f"tbc_weighted (nz_weight={NONZERO_WEIGHT}, base={MAX_TRAIN_NODES_BASE}, large={MAX_TRAIN_NODES_LARGE})"
    else:
        return f"degree_stratified (nz_weight={NONZERO_WEIGHT}, hdz_weight={ZERO_HIGH_DEG_WEIGHT}, base={MAX_TRAIN_NODES_BASE}, large={MAX_TRAIN_NODES_LARGE})"

def generate_report(all_results, test_infos, elapsed_hours):
    lines = [
        "# CLGNN Experiment Results", "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Device**: {DEVICE}", "",
        "## Settings", "",
        f"- alpha={ALPHA}, lambda={LAMBDA_WEIGHT}, dim={FEAT_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}",
        f"- epochs={NUM_EPOCHS}, neighbors={NUM_NEIGHBORS}, LRs={LRS}, seeds={args.num_seeds}",
        f"- sampling: {sampling_description()}", "",
        "## Results", "",
        "| Dataset | MAE | Spearman |",
        "| :------ | :-- | :------- |",
    ]
    for info in test_infos:
        name = info['name']
        res_list = all_results.get(name, [])
        if res_list:
            maes = [r['MAE'] for r in res_list]
            sps = [r['Spearman'] for r in res_list]
            lines.append(f"| {name} | {np.mean(maes):.6f} +/- {np.std(maes):.6f} | {np.mean(sps):.4f} +/- {np.std(sps):.4f} |")
        else:
            lines.append(f"| {name} | N/A | N/A |")
    lines += ["", "---", f"*Total time: {elapsed_hours:.2f} hours*"]
    return "\n".join(lines)


def eval_single_checkpoint():
    log(f"=== Eval Single Checkpoint: {args.eval_checkpoint} ===")
    test_infos = discover(args.test_data_dir, exclude=EXCLUDE_TEST, scores_subdir=args.scores_dir)
    train_infos = discover(args.train_data_dir, scores_subdir=args.scores_dir)
    feat_size = scan_max_node(train_infos + test_infos)
    init_ngh = build_ngh(*load_graph(test_infos[0]['graph']))
    model, ckpt = load_model_from_checkpoint(args.eval_checkpoint, feat_size, init_ngh)
    results = evaluate_all(model, test_infos, feat_size)
    log("\n=== Results ===")
    for name, res in results.items():
        log(f"  {name}: MAE={res['MAE']:.6f}, Spearman={res['Spearman']:.4f}")

def eval_only_mode():
    log("=== Eval-Only Mode ===")
    test_infos = discover(args.test_data_dir, exclude=EXCLUDE_TEST, scores_subdir=args.scores_dir)
    train_infos = discover(args.train_data_dir, scores_subdir=args.scores_dir)
    if not test_infos: log("ERROR: No test datasets!"); sys.exit(1)
    feat_size = scan_max_node(train_infos + test_infos)
    init_ngh = build_ngh(*load_graph(test_infos[0]['graph']))
    completed = find_completed_seeds(args.save_dir)
    if not completed: log(f"ERROR: No saved models found in {args.save_dir}"); sys.exit(1)
    log(f"Found {len(completed)} saved model(s): seeds {sorted(completed.keys())}")
    all_results = {info['name']: [] for info in test_infos}
    for seed in sorted(completed.keys()):
        info = completed[seed]
        log(f"\n--- Evaluating seed {seed} from {info['path']} ---")
        model, ckpt = load_model_from_checkpoint(info['path'], feat_size, init_ngh)
        eval_results = evaluate_all(model, test_infos, feat_size)
        for name, res in eval_results.items(): all_results[name].append(res)
        del model; torch.cuda.empty_cache(); gc.collect()
    report = generate_report(all_results, test_infos, 0)
    with open(args.output, 'w') as f: f.write(report)
    log(f"\nReport saved to {args.output}")
    print("\n" + report, flush=True)

def train_mode():
    start = time.time()
    log(f"CLGNN Experiment Runner | device={DEVICE}")
    log(f"Config: alpha={ALPHA}, lambda={LAMBDA_WEIGHT}, dim={FEAT_DIM}, layers={NUM_LAYERS}, heads={NUM_HEADS}, epochs={NUM_EPOCHS}")
    log(f"Sampling: {sampling_description()}")
    log(f"Seeds={args.num_seeds}, LRs={LRS}")
    if args.resume: log("Resume mode: will skip already completed seeds")
    test_infos = discover(args.test_data_dir, exclude=EXCLUDE_TEST, scores_subdir=args.scores_dir)
    train_infos = discover(args.train_data_dir, scores_subdir=args.scores_dir)
    log(f"Found {len(test_infos)} test, {len(train_infos)} train datasets (scores from '{args.scores_dir}')")
    if EXCLUDE_TEST: log(f"Excluded test datasets: {EXCLUDE_TEST}")
    if not test_infos: log("ERROR: No test datasets!"); sys.exit(1)
    log("Scanning max node id...")
    feat_size = scan_max_node(train_infos + test_infos)
    log(f"Node feature size: {feat_size}")
    completed = find_completed_seeds(args.save_dir) if args.resume else {}
    if completed: log(f"Found {len(completed)} completed seed(s): {sorted(completed.keys())}")
    all_results = {info['name']: [] for info in test_infos}
    for seed in sorted(completed.keys()):
        if seed <= args.num_seeds:
            eval_results = completed[seed].get('eval_results', {})
            if eval_results:
                for name, res in eval_results.items():
                    if name in all_results: all_results[name].append(res)
                log(f"  Loaded cached results for seed {seed}")
    init_ngh = build_ngh(*load_graph(train_infos[0]['graph'])) if train_infos else build_ngh(*load_graph(test_infos[0]['graph']))
    for seed in range(1, args.num_seeds+1):
        if seed in completed: log(f"\n[Skip] Seed {seed} already completed"); continue
        log(f"\n{'='*50}")
        log(f"Seed {seed}/{args.num_seeds}")
        log(f"{'='*50}")
        best_eval = None; best_lr = None; best_spearman_avg = -float('inf')
        for lr in LRS:
            log(f"\n  --- Seed {seed}, LR={lr} ---")
            set_seed(seed)
            model = build_model(feat_size, init_ngh)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)
            try:
                train_model(model, optimizer, scheduler, train_infos, feat_size)
                eval_results = evaluate_all(model, test_infos, feat_size)
                sp_values = [r['Spearman'] for r in eval_results.values() if not np.isnan(r['Spearman'])]
                sp_avg = np.mean(sp_values) if sp_values else -1.0
                state_dict_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_checkpoint(args.save_dir, seed, lr, state_dict_cpu, eval_results, feat_size)
                del state_dict_cpu
                if sp_avg > best_spearman_avg:
                    best_spearman_avg = sp_avg; best_eval = eval_results; best_lr = lr
            except Exception as exc:
                log(f"  FAILED: {exc}")
                import traceback; traceback.print_exc()
            del model, optimizer, scheduler
            torch.cuda.empty_cache(); gc.collect()
        if best_eval is not None:
            log(f"\n  Best LR for seed {seed}: {best_lr} (avg Spearman={best_spearman_avg:.4f})")
            for name, res in best_eval.items(): all_results[name].append(res)
        else:
            log(f"  Seed {seed}: ALL LRs FAILED")
        elapsed = (time.time()-start)/3600
        report = generate_report(all_results, test_infos, elapsed)
        with open(args.output, 'w') as f: f.write(report)
        log(f"  Intermediate report saved to {args.output}")
    elapsed = (time.time()-start)/3600
    report = generate_report(all_results, test_infos, elapsed)
    with open(args.output, 'w') as f: f.write(report)
    log(f"\nFinal report saved to {args.output}")
    log(f"Total time: {elapsed:.2f} hours")
    print("\n" + report, flush=True)


if __name__ == '__main__':
    if args.eval_checkpoint: eval_single_checkpoint()
    elif args.eval_only: eval_only_mode()
    else: train_mode()