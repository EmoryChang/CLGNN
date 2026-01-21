import math
import logging
import pickle
import time
import random
import sys
import argparse

import scipy
import torch
import pandas as pd
import numpy as np

from scipy.stats import kendalltau
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


from module import CLGNN
from nx2graphs import load_real_data, load_real_true_TKC, load_train_real_data, load_real_train_true_TKC
from utils import loss_cal


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 64)
        self.fc_2 = torch.nn.Linear(64, 32)
        self.fc_3 = torch.nn.Linear(32, 1)

        self.act = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc_1.weight)
        torch.nn.init.kaiming_normal_(self.fc_2.weight)
        torch.nn.init.kaiming_normal_(self.fc_3.weight)

        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


parser = argparse.ArgumentParser('Interface for CLGNN experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-sewiki')
parser.add_argument('--bs', type=int, default=5000, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument("--local_rank", type=int)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
LR_MODEL_SAVE_PATH = f'./saved_models/{args.agg_method}-{args.attn_mode}-{args.data}_mlp.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
import math

def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    eps = 1e-6
    tgan.ngh_finder = sampler
    tgan.eval()

    TEST_BATCH_SIZE = BATCH_SIZE
    num_test_instance = len(src)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    test_pred_tbc_list = []

    with torch.no_grad():
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            predicted_tbc, node_embeddings = tgan(
                test_src_l_cut, 
                test_ts_l_cut, 
                tbc_labels=None,  
                num_neighbors=NUM_NEIGHBORS
            )
            test_pred_tbc_list.extend(predicted_tbc.cpu().numpy().tolist())

    test_pred_tbc_list = np.array(test_pred_tbc_list)
    test_pred_tbc_list = np.exp(test_pred_tbc_list) - eps

    label = np.array(label)


    mae = mean_absolute_error(label, test_pred_tbc_list)
    rmse = mean_squared_error(label, test_pred_tbc_list, squared=False)
    spearman_corr, _ = spearmanr(label, test_pred_tbc_list)
    kendall_corr, _ = kendalltau(label, test_pred_tbc_list)


    hit_ks = [10, 30, 50]
    hit_results = {}

    pred_topk_idx = np.argsort(test_pred_tbc_list)[::-1] 
    true_topk_idx = np.argsort(label)[::-1]  

    for k in hit_ks:
        pred_top_k = set(pred_topk_idx[:k])
        true_top_k = set(true_topk_idx[:k])
        hits = len(pred_top_k.intersection(true_top_k)) / k
        hit_results[f"Hits@{k}"] = hits


    hit_ps = [0.1, 0.3, 0.5] 
    n = len(label)  

    for p in hit_ps:
        k = max(1, int(n * p)) 
        pred_top_k = set(pred_topk_idx[:k])
        true_top_k = set(true_topk_idx[:k])
        hits = len(pred_top_k.intersection(true_top_k)) / k
        hit_results[f"Hits@{int(p * 100)}%"] = hits


    label = np.array(label)
    pred = np.array(test_pred_tbc_list)

    low_idx = np.where(label == 0)[0]
    mid_idx = np.where((label > 0) & (label <= np.median(label[label > 0])))[0]
    high_idx = np.where(label > np.median(label[label > 0]))[0]

    mae_low = mean_absolute_error(label[low_idx], pred[low_idx]) if len(low_idx) > 0 else None
    mae_mid = mean_absolute_error(label[mid_idx], pred[mid_idx]) if len(mid_idx) > 0 else None
    mae_high = mean_absolute_error(label[high_idx], pred[high_idx]) if len(high_idx) > 0 else None

    results = {
        'MAE': mae,
        'RMSE': rmse,
        'Spearman': spearman_corr,
        'Kendall': kendall_corr,
        'MAE_Low': mae_low,
        'MAE_Mid': mae_mid,
        'MAE_High': mae_high,
    }
    results.update(hit_results)

    output_path = 'tbc_predictions.txt'
    with open(output_path, 'w') as f:
        f.write('true_value,predicted_value\n')
        for true, pred in zip(label, test_pred_tbc_list):
            f.write(f'{true},{pred}\n')
    print(f'the prediction results have been saved in {output_path}')

    return results


n_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
# test_real_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((4200000, 128))


def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


setSeeds(89)


train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, train_real_node, train_real_time, \
    train_real_ngh_finder = load_train_real_data(UNIFORM)

test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, test_real_node, test_real_time, \
    test_real_ngh_finder = load_real_data(dataName=DATA)


nodeList_train_real, train_label_l_real = load_real_train_true_TKC()
nodeList_test_real, test_label_l_real = load_real_true_TKC('{}'.format(DATA))
train_ts_list, test_ts_list, train_real_ts_list = [], [], []

for idx in range(len(nodeList_train_real)):
    train_real_ts_list.append(np.array([train_real_time[idx]] * len(nodeList_train_real[idx])))

test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))

TEST_BATCH_SIZE = BATCH_SIZE
num_test_instance = len(nodeList_test_real)
num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    test_src_l_cut = np.array(nodeList_test_real[s_idx:e_idx])
    test_ts_l_cut = np.array(test_real_ts_list[s_idx:e_idx])
    test_real_ngh_finder.preprocess(tuple(test_src_l_cut), tuple(test_ts_l_cut), NUM_LAYER, NUM_NEIGHBORS)


device = torch.device('cuda:{}'.format(GPU))


clgnn = CLGNN(
    train_real_ngh_finder[0], 
    test_real_feat, 
    num_layers=NUM_LAYER, 
    use_time=USE_TIME,
    agg_method='dual',  
    attn_mode=ATTN_MODE, 
    seq_len=SEQ_LEN, 
    n_head=NUM_HEADS, 
    drop_out=DROP_OUT,
    lambda_weight=0.5,  
    alpha=0.3  
)
clgnn = clgnn.to(device)


optimizer = torch.optim.Adam(clgnn.parameters(), LEARNING_RATE)


print('train start')
start_train = time.time()
for epoch in range(NUM_EPOCH):
    logger.info('start {} epoch'.format(epoch))
    train_loss, train_reg_loss, train_contrast_loss = [], [], []
    clgnn.train()
    clgnn.to(device)

    for j in tqdm(range(len(train_real_ts_l))):
        clgnn.ngh_finder = train_real_ngh_finder[j]
        m_loss, m_reg_loss, m_contrast_loss = [], [], []

        TRAIN_BATCH_SIZE = BATCH_SIZE
        num_train_instance = len(nodeList_train_real[j])
        num_train_batch = math.ceil(num_train_instance / TRAIN_BATCH_SIZE)
        
        for k in range(num_train_batch):
            s_idx = k * TRAIN_BATCH_SIZE
            e_idx = min(num_train_instance, s_idx + TRAIN_BATCH_SIZE)
            src_l_cut = np.array(nodeList_train_real[j][s_idx:e_idx])
            label_l_cut = train_label_l_real[j][s_idx:e_idx]
            ts_l_cut = train_real_ts_list[j][s_idx:e_idx]
            
            optimizer.zero_grad()
            scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)
            

            true_label = torch.from_numpy(np.array(label_l_cut)).float().to(device)
            

            predicted_tbc, contrastive_loss, node_embeddings = clgnn(
                src_l_cut, 
                ts_l_cut, 
                tbc_labels=true_label,
                num_neighbors=NUM_NEIGHBORS
            )
            

            total_loss, regression_loss = clgnn.compute_total_loss(
                predicted_tbc, true_label, contrastive_loss
            )
            

            m_loss.append(total_loss.item())
            m_reg_loss.append(regression_loss.item())
            m_contrast_loss.append(contrastive_loss.item())
            

            total_loss.backward()
            nn.utils.clip_grad_norm_(clgnn.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_loss.append(np.mean(m_loss))
        train_reg_loss.append(np.mean(m_reg_loss))
        train_contrast_loss.append(np.mean(m_contrast_loss))

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean total loss: {:.6f}'.format(np.mean(train_loss)))
    logger.info('Epoch mean regression loss: {:.6f}'.format(np.mean(train_reg_loss)))
    logger.info('Epoch mean contrastive loss: {:.6f}'.format(np.mean(train_contrast_loss)))

train_end = time.time()
print('train end, train_time={}'.format(train_end - start_train))


real_data_start_time = time.time()
test_results = eval_real_data('test for real data', clgnn, None, test_real_ngh_finder,
                                              nodeList_test_real, test_real_ts_list, test_label_l_real)
real_data_end_time = time.time()
logger.info('Test real data statistics:\n'
            'Hits@10: {:.4f}, Hits@30: {:.4f}, Hits@50: {:.4f}\n'
            'Hits@10%: {:.4f}, Hits@30%: {:.4f}, Hits@50%: {:.4f}\n'
            'MAE: {:.6f}, RMSE: {:.6f}\n'
            'MAE_Low: {:.6f}, MAE_Mid: {:.6f}, MAE_High: {:.6f}\n'
            'Spearman: {:.4f}, Kendall: {:.4f}, Time: {:.2f}s'.format(
                test_results['Hits@10'], test_results['Hits@30'], test_results['Hits@50'],
                test_results['Hits@10%'], test_results['Hits@30%'], test_results['Hits@50%'],
                test_results['MAE'], test_results['RMSE'],
                test_results['MAE_Low'], test_results['MAE_Mid'], test_results['MAE_High'],
                test_results['Spearman'], test_results['Kendall'],
                real_data_end_time - real_data_start_time
))

torch.save(clgnn.state_dict(), './saved_models/clgnn_integrated_model.pth')
print("Integrated CLGNN model saved successfully")
