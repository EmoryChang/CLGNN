import logging
import warnings

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='Number of distinct clusters')
import math


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class PathCountEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(PathCountEncode, self).__init__()
        self.path_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, expand_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(expand_dim // 2, expand_dim)
        )

    def forward(self, path_counts):
        log_path_counts = torch.log(1 + path_counts.float())
        log_path_counts = log_path_counts.unsqueeze(-1)
        path_embed = self.path_mlp(log_path_counts)
        return path_embed


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        try:
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            if mask is not None:
                attn = attn.masked_fill(mask, -1e10)
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)
            return output, attn
        except Exception as e:
            logging.error(f"Error in ScaledDotProductAttention forward: {str(e)}")
            return v, torch.zeros_like(attn)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.2):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        try:
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
            sz_b, len_q, _ = q.size()
            sz_b, len_k, _ = k.size()
            sz_b, len_v, _ = v.size()
            residual = q
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
            if mask is not None:
                mask = mask.repeat(n_head, 1, 1)
            output, attn = self.attention(q, k, v, mask=mask)
            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)
            return output, attn.view(sz_b, n_head, len_q, len_k).mean(dim=1)
        except Exception as e:
            logging.error(f"Error in MultiHeadAttention forward: {str(e)}")
            return q, torch.zeros(sz_b, len_q, len_k, device=q.device)


class MapBasedMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        q = torch.unsqueeze(q, dim=2)
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        k = torch.unsqueeze(k, dim=1)
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        mask = mask.repeat(n_head, 1, 1)
        q_k = torch.cat([q, k], dim=3)
        attn = self.weight_map(q_k).squeeze(dim=3)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)
        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        self.d_T = expand_dim // 2
        self.expand_dim = expand_dim
        self.linear = torch.nn.Linear(1, self.d_T, bias=False)

    def forward(self, ts):
        squeezed = False
        if ts.dim() == 1:
            ts = ts.unsqueeze(0)
            squeezed = True
        batch_size, seq_len = ts.size()
        ts = ts.unsqueeze(-1)
        projected = self.linear(ts)
        cos_components = torch.cos(projected)
        sin_components = torch.sin(projected)
        harmonic = torch.cat([cos_components, sin_components], dim=-1)
        harmonic = harmonic * math.sqrt(1.0 / self.expand_dim)
        if squeezed:
            harmonic = harmonic.squeeze(0)
        return harmonic


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.att_dim = feat_dim + time_dim
        self.act = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, hidden_size=self.feat_dim,
                                  num_layers=1, batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, mask):
        seq_x = torch.cat([seq, seq_t], dim=2)
        _, (hn, _) = self.lstm(seq_x)
        hn = hn[-1, :, :]
        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, mask):
        src_x = src
        seq_x = torch.cat([seq, seq_t], dim=2)
        hn = seq_x.mean(dim=1)
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    def __init__(self, feat_dim, time_dim, attn_mode='prod', n_head=2, drop_out=0.2):
        super(AttnModel, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + time_dim)
        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)
        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head, dropout=drop_out)
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head, dropout=drop_out)
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, mask):
        src_ext = torch.unsqueeze(src, dim=1)
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_t], dim=2)
        k = torch.cat([seq, seq_t], dim=2)
        mask = torch.unsqueeze(mask, dim=2)
        mask = mask.permute([0, 2, 1])
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)
        output = output.squeeze()
        attn = attn.squeeze()
        output = self.merger(output, src)
        return output, attn


class DualMessageAggregator(torch.nn.Module):
    def __init__(self, feat_dim, time_dim, n_head=2, drop_out=0.2, lambda_weight=0.3):
        super(DualMessageAggregator, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.n_head = n_head
        self.lambda_weight = lambda_weight
        self.msg_dim = 2 * feat_dim + time_dim + feat_dim
        self.edge_attention = MultiHeadAttention(
            n_head=n_head, d_model=self.msg_dim,
            d_k=self.msg_dim // n_head, d_v=self.msg_dim // n_head, dropout=drop_out
        )
        self.mean_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.msg_dim + feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.msg_dim + feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, src_node_feat, src_node_t_embed, ngh_feat, ngh_t_embed,
                path_encodings, path_counts, mask):
        try:
            device = src_node_feat.device
            batch_size = src_node_feat.size(0)
            num_neighbors = ngh_feat.size(1)
            src_expanded = src_node_feat.unsqueeze(1).expand(-1, num_neighbors, -1)
            messages = torch.cat([ngh_feat, src_expanded, ngh_t_embed, path_encodings], dim=2)
            valid_mask = ~mask
            path_weights_masked = path_counts * valid_mask.float()
            weight_sum_per_neighbor = path_weights_masked.sum(dim=1, keepdim=True) + 1e-8
            num_valid_neighbors = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            normalized_weights = path_weights_masked / weight_sum_per_neighbor
            mean_aggregated = torch.sum(messages * normalized_weights.unsqueeze(-1), dim=1)
            mean_aggregated = mean_aggregated / num_valid_neighbors
            mean_combined = torch.cat([mean_aggregated, src_node_feat], dim=1)
            h_mean = self.mean_mlp(mean_combined)
            query_base = torch.cat([src_node_feat, src_node_t_embed], dim=1)
            pad_dim = self.msg_dim - query_base.size(1)
            if pad_dim > 0:
                query_pad = torch.zeros(batch_size, pad_dim, device=device)
                query = torch.cat([query_base, query_pad], dim=1)
            else:
                query = query_base
            query = query.unsqueeze(1)
            attn_mask = mask.unsqueeze(1)
            attn_output, attn_weights = self.edge_attention(q=query, k=messages, v=messages, mask=attn_mask)
            attn_output = attn_output.squeeze(1)
            attn_combined = torch.cat([attn_output, src_node_feat], dim=1)
            h_attention = self.attention_mlp(attn_combined)
            final_embedding = self.lambda_weight * h_mean + (1 - self.lambda_weight) * h_attention
            return final_embedding, attn_weights.squeeze(1) if attn_weights is not None else None
        except Exception as e:
            self.logger.error(f"Error in DualMessageAggregator forward: {str(e)}")
            return src_node_feat, None


class CLGNN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, num_layers=3, use_time='time',
                 agg_method='dual', attn_mode='prod', seq_len=20,
                 n_head=2, drop_out=0.1, lambda_weight=0.3, alpha=0.2,
                 cluster_sampling='random'):
        super(CLGNN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=True)
        self.feat_dim = self.n_feat_th.shape[1]
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        self.path_encoder = PathCountEncode(expand_dim=self.feat_dim)
        self.message_agg = torch.nn.ModuleList([
            DualMessageAggregator(feat_dim=self.feat_dim, time_dim=self.feat_dim,
                                  n_head=n_head, drop_out=drop_out, lambda_weight=lambda_weight)
            for _ in range(num_layers)
        ])
        self.value_net = ValueNet(input_dim=self.feat_dim, hidden_dim=64, drop_out=drop_out)
        self.k_contrast_net = KContrastNet(
            embed_dim=self.feat_dim, tau=0.1, gamma_pos=0.1, gamma_neg=0.5,
            max_k=5, bootstrap_b=10, sample_ratio=0.4, cluster_sampling=cluster_sampling
        )

    def tem_conv(self, src_idx_l, src_node_features, src_ngh_node_batch, src_ngh_t_batch, src_ngh_feat, cut_time_l, curr_layers):
        try:
            device = src_idx_l.device
            if src_idx_l.dim() == 0: src_idx_l = src_idx_l.unsqueeze(0)
            if src_node_features.dim() == 1: src_node_features = src_node_features.unsqueeze(0)
            if src_ngh_node_batch.dim() == 1: src_ngh_node_batch = src_ngh_node_batch.unsqueeze(0)
            if src_ngh_t_batch.dim() == 1: src_ngh_t_batch = src_ngh_t_batch.unsqueeze(0)
            if src_ngh_feat.dim() == 2: src_ngh_feat = src_ngh_feat.unsqueeze(0)
            if cut_time_l.dim() == 0: cut_time_l = cut_time_l.unsqueeze(0)
            if curr_layers == 0:
                return src_node_features
            time_diffs = cut_time_l.unsqueeze(1) - src_ngh_t_batch
            time_encodings = self.time_encoder(time_diffs)
            path_counts = self.compute_path_counts(src_idx_l, src_ngh_node_batch, src_ngh_t_batch, cut_time_l)
            path_encodings = self.path_encoder(path_counts)
            mask = (src_ngh_node_batch == 0) & (src_ngh_t_batch == 0)
            src_node_t_embed = self.time_encoder(torch.zeros(src_idx_l.size(0), device=device).unsqueeze(1)).squeeze(1)
            curr_embeddings, _ = self.message_agg[curr_layers - 1](
                src_node_features, src_node_t_embed, src_ngh_feat,
                time_encodings, path_encodings, path_counts, mask
            )
            return curr_embeddings
        except Exception as e:
            self.logger.error(f"Error in temporal convolution: {str(e)}")
            return src_node_features

    def forward(self, src_idx_l, cut_time_l, tbc_labels=None, num_neighbors=20):
        try:
            device = self.n_feat_th.device
            batch_size = len(src_idx_l)
            if isinstance(src_idx_l, torch.Tensor): src_idx_l = src_idx_l.cpu().numpy()
            elif isinstance(src_idx_l, list): src_idx_l = np.array(src_idx_l)
            if isinstance(cut_time_l, torch.Tensor): cut_time_l = cut_time_l.cpu().numpy()
            elif isinstance(cut_time_l, list): cut_time_l = np.array(cut_time_l)
            src_ngh_node_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l, cut_time_l, num_neighbors=num_neighbors)
            src_idx_t = torch.from_numpy(src_idx_l).long().to(device)
            cut_time_t = torch.from_numpy(cut_time_l).float().to(device)
            src_ngh_node_batch = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_t_batch = torch.from_numpy(src_ngh_t_batch).float().to(device)
            src_node_features = self.n_feat_th[src_idx_t]
            src_ngh_node_flat = src_ngh_node_batch.flatten()
            src_ngh_t_flat = src_ngh_t_batch.flatten()
            src_ngh_feat_flat = self.n_feat_th[src_ngh_node_flat]
            ngh_ngh_node_batch, ngh_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_ngh_node_flat.cpu().numpy(), src_ngh_t_flat.cpu().numpy(), num_neighbors=num_neighbors)
            ngh_ngh_node_batch = torch.from_numpy(ngh_ngh_node_batch).long().to(device)
            ngh_ngh_t_batch = torch.from_numpy(ngh_ngh_t_batch).float().to(device)
            src_ngh_feat = self.tem_conv(
                src_ngh_node_flat, src_ngh_feat_flat, ngh_ngh_node_batch, ngh_ngh_t_batch,
                self.n_feat_th[ngh_ngh_node_batch].view(-1, num_neighbors, self.feat_dim),
                src_ngh_t_flat, self.num_layers - 1)
            node_embeddings = self.tem_conv(
                src_idx_t, src_node_features, src_ngh_node_batch, src_ngh_t_batch,
                src_ngh_feat.view(batch_size, num_neighbors, -1), cut_time_t, self.num_layers)
            tbc_scores = self.value_net(node_embeddings)
            if tbc_labels is not None:
                if isinstance(tbc_labels, np.ndarray):
                    tbc_labels = torch.from_numpy(tbc_labels).float().to(device)
                elif isinstance(tbc_labels, list):
                    tbc_labels = torch.tensor(tbc_labels, dtype=torch.float32, device=device)
                if tbc_labels.dim() > 1: tbc_labels = tbc_labels.squeeze()
                if tbc_scores.dim() > 1: tbc_scores = tbc_scores.squeeze()
                contrastive_loss = self.k_contrast_net(node_embeddings, tbc_labels)
                return tbc_scores, contrastive_loss, node_embeddings
            return tbc_scores, node_embeddings
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            if tbc_labels is not None:
                return torch.zeros(batch_size, device=device), torch.tensor(0.0, device=device), torch.zeros((batch_size, self.feat_dim), device=device)
            return torch.zeros(batch_size, device=device), torch.zeros((batch_size, self.feat_dim), device=device)

    def compute_total_loss(self, predicted_tbc, true_tbc, contrastive_loss):
        try:
            device = predicted_tbc.device
            if predicted_tbc.dim() == 2 and predicted_tbc.size(1) == 1: predicted_tbc = predicted_tbc.view(-1)
            if true_tbc.dim() == 2 and true_tbc.size(1) == 1: true_tbc = true_tbc.view(-1)
            predicted_tbc = torch.nan_to_num(predicted_tbc, nan=0.0, posinf=1.0, neginf=-1.0)
            true_tbc = torch.nan_to_num(true_tbc, nan=0.0, posinf=1.0, neginf=-1.0)
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                contrastive_loss = torch.tensor(0.0, device=device)
            regression_loss = F.l1_loss(predicted_tbc, torch.log1p(true_tbc))
            total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * regression_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return regression_loss, regression_loss
            return total_loss, regression_loss
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            return torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

    def compute_path_counts(self, src_idx_l, src_ngh_node_batch, src_ngh_t_batch, cut_time_l):
        device = src_idx_l.device
        ngh_nodes_np = src_ngh_node_batch.cpu().numpy().astype(np.int32)
        ngh_times_np = src_ngh_t_batch.cpu().numpy().astype(np.float64)
        counts_np = self.ngh_finder.batch_path_counts(ngh_nodes_np, ngh_times_np)
        return torch.from_numpy(counts_np).float().to(device)


class KContrastNet(torch.nn.Module):
    def __init__(self, embed_dim, tau=0.1, gamma_pos=0.3, gamma_neg=0.7,
                 max_k=10, bootstrap_b=10, sample_ratio=0.4,
                 cluster_sampling='random'):
        super(KContrastNet, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.embed_dim = embed_dim
        self.tau = tau
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.max_k = max_k
        self.bootstrap_b = bootstrap_b
        self.sample_ratio = sample_ratio
        self.cluster_sampling = cluster_sampling
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        for m in self.proj:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def _stratified_subsample(self, num_samples, tbc_np):
        sub_n = max(4, int(num_samples * self.sample_ratio))
        nonzero_tbc = tbc_np[tbc_np > 0]
        tbc_median = np.median(nonzero_tbc) if len(nonzero_tbc) > 0 else 0.0
        g_zero = np.where(tbc_np == 0)[0]
        g_low = np.where((tbc_np > 0) & (tbc_np <= tbc_median))[0]
        g_high = np.where(tbc_np > tbc_median)[0]
        groups = [g_zero, g_low, g_high]
        counts = np.array([len(g) for g in groups])
        total = counts.sum()
        if total == 0:
            return np.random.choice(num_samples, min(sub_n, num_samples), replace=False)
        quotas = np.floor(counts / total * sub_n).astype(int)
        remainder = sub_n - quotas.sum()
        fracs = (counts / total * sub_n) - quotas
        for _ in range(remainder):
            idx = np.argmax(fracs)
            quotas[idx] += 1
            fracs[idx] = -1.0
        chosen = []
        for g, q in zip(groups, quotas):
            q = min(q, len(g))
            if q > 0:
                chosen.append(np.random.choice(g, q, replace=False))
        return np.concatenate(chosen) if chosen else np.arange(min(sub_n, num_samples))

    def stability_based_clustering(self, features_np, tbc_np=None):
        num_samples = features_np.shape[0]
        if num_samples < 4:
            return 2, np.zeros(num_samples, dtype=np.int32)
        if self.cluster_sampling == 'stratified' and tbc_np is not None:
            sub_indices = self._stratified_subsample(num_samples, tbc_np)
        else:
            sub_n = max(4, int(num_samples * self.sample_ratio))
            sub_indices = np.random.choice(num_samples, sub_n, replace=False)
        features_sub = features_np[sub_indices]
        sub_n = len(sub_indices)
        actual_max_k = min(self.max_k, sub_n - 1)
        if actual_max_k < 2:
            return 2, np.zeros(num_samples, dtype=np.int32)
        instability_scores = {}
        for candidate_k in range(2, actual_max_k + 1):
            distances = []
            for _ in range(self.bootstrap_b):
                idx_x = np.random.choice(sub_n, sub_n, replace=True)
                idx_y = np.random.choice(sub_n, sub_n, replace=True)
                try:
                    kmeans_x = KMeans(n_clusters=candidate_k, n_init=3, max_iter=100, random_state=None)
                    kmeans_y = KMeans(n_clusters=candidate_k, n_init=3, max_iter=100, random_state=None)
                    labels_x = kmeans_x.fit_predict(features_sub[idx_x])
                    labels_y = kmeans_y.fit_predict(features_sub[idx_y])
                    set_x = set(idx_x.tolist())
                    set_y = set(idx_y.tolist())
                    common = np.array(sorted(set_x & set_y), dtype=np.int64)
                    if len(common) < 2:
                        distances.append(1.0)
                        continue
                    map_x = np.full(sub_n, -1, dtype=np.int64)
                    for pos, orig in enumerate(idx_x): map_x[orig] = pos
                    map_y = np.full(sub_n, -1, dtype=np.int64)
                    for pos, orig in enumerate(idx_y): map_y[orig] = pos
                    cx = map_x[common]
                    cy = map_y[common]
                    valid = (cx >= 0) & (cy >= 0)
                    cx = cx[valid]
                    cy = cy[valid]
                    if len(cx) < 2:
                        distances.append(1.0)
                        continue
                    lx = labels_x[cx]
                    ly = labels_y[cy]
                    same_x = (lx[:, None] == lx[None, :])
                    same_y = (ly[:, None] == ly[None, :])
                    triu = np.triu_indices(len(cx), k=1)
                    disagree = np.sum(same_x[triu] != same_y[triu])
                    total = len(triu[0])
                    distances.append(disagree / max(total, 1))
                except Exception:
                    distances.append(1.0)
            instability_scores[candidate_k] = np.mean(distances)
        best_k = min(instability_scores, key=instability_scores.get) if instability_scores else 2
        kmeans_final = KMeans(n_clusters=best_k, n_init=10, max_iter=300)
        cluster_labels = kmeans_final.fit_predict(features_np)
        return best_k, cluster_labels

    def forward(self, embeddings, tbc_labels):
        try:
            device = embeddings.device
            batch_size = embeddings.size(0)
            if batch_size < 4:
                return torch.tensor(0.0, device=device, requires_grad=True)
            proj_embeddings = self.proj(embeddings)
            features_np = embeddings.detach().cpu().numpy()
            tbc_np = tbc_labels.detach().cpu().numpy()
            best_k, cluster_labels = self.stability_based_clustering(features_np, tbc_np)
            cluster_labels_tensor = torch.from_numpy(cluster_labels).long().to(device)
            tbc_median = torch.median(tbc_labels[tbc_labels > 0]) if (tbc_labels > 0).any() else torch.tensor(1e-6, device=device)
            tbc_median = tbc_median.clamp(min=1e-8)
            pos_threshold = self.gamma_pos * tbc_median
            neg_threshold = self.gamma_neg * tbc_median
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            num_valid_anchors = 0
            for cluster_id in range(best_k):
                cluster_mask = (cluster_labels_tensor == cluster_id)
                cluster_indices = torch.where(cluster_mask)[0]
                if len(cluster_indices) < 2:
                    continue
                cluster_embeddings = proj_embeddings[cluster_indices]
                cluster_tbc = tbc_labels[cluster_indices]
                cluster_size = len(cluster_indices)
                tbc_diff = torch.abs(cluster_tbc.unsqueeze(0) - cluster_tbc.unsqueeze(1))
                self_mask = torch.eye(cluster_size, device=device).bool()
                pos_mask = (tbc_diff > 0) & (tbc_diff <= pos_threshold) & (~self_mask)
                neg_mask = (tbc_diff >= neg_threshold) & (~self_mask)
                has_pos = pos_mask.any(dim=1)
                has_neg = neg_mask.any(dim=1)
                valid_anchor_mask = has_pos & has_neg
                if not valid_anchor_mask.any():
                    continue
                sim_matrix = torch.matmul(cluster_embeddings, cluster_embeddings.t())
                pos_weights = (pos_threshold / tbc_diff.clamp(min=1e-8)).clamp(max=10.0)
                neg_weights = (tbc_diff / neg_threshold).clamp(max=10.0)
                pos_logits = pos_weights * sim_matrix / self.tau
                neg_logits = neg_weights * sim_matrix / self.tau
                pos_logits = pos_logits.masked_fill(~pos_mask, -1e9)
                neg_logits = neg_logits.masked_fill(~neg_mask, -1e9)
                max_val = torch.max(
                    pos_logits[valid_anchor_mask].max(dim=1).values,
                    neg_logits[valid_anchor_mask].max(dim=1).values
                ).detach()
                pos_exp = torch.exp(pos_logits[valid_anchor_mask] - max_val.unsqueeze(1))
                neg_exp = torch.exp(neg_logits[valid_anchor_mask] - max_val.unsqueeze(1))
                pos_exp = pos_exp * pos_mask[valid_anchor_mask].float()
                neg_exp = neg_exp * neg_mask[valid_anchor_mask].float()
                pos_sum = pos_exp.sum(dim=1)
                neg_sum = neg_exp.sum(dim=1)
                denom = pos_sum + neg_sum
                anchor_losses = -torch.log(pos_sum / denom.clamp(min=1e-8) + 1e-8)
                valid_losses = anchor_losses[torch.isfinite(anchor_losses)]
                if len(valid_losses) > 0:
                    total_loss = total_loss + valid_losses.sum()
                    num_valid_anchors += len(valid_losses)
            if num_valid_anchors > 0:
                total_loss = total_loss / num_valid_anchors
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return torch.tensor(0.0, device=device, requires_grad=True)
            return total_loss
        except Exception as e:
            self.logger.error(f"Error in KContrastNet forward: {str(e)}")
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, drop_out=0.1):
        super(ValueNet, self).__init__()
        self.input_dim = input_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        scores = self.mlp(x)
        if scores.dim() == 2 and scores.size(1) == 1:
            scores = scores.view(-1)
        return scores