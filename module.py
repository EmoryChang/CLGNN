import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
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
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        time_dim = expand_dim
        self.factor = factor

        self.omega = torch.nn.Parameter(torch.randn(time_dim // 2))

    def forward(self, ts):

        try:

            if ts.dim() == 1:
                ts = ts.unsqueeze(0)  # [1, N]
            if ts.dim() == 2:
                batch_size, seq_len = ts.size()
            else:
                raise ValueError(f"Input tensor must be 1D or 2D, got shape {ts.shape}")
            
            time_dim = len(self.omega)

            ts = ts.unsqueeze(-1)
            

            omega_t = ts * self.omega.view(1, 1, -1)
            

            cos_components = torch.cos(omega_t)
            sin_components = torch.sin(omega_t)
            

            harmonic = torch.stack([cos_components, sin_components], dim=-1)
            harmonic = harmonic.view(batch_size, seq_len, -1)

            harmonic = harmonic * math.sqrt(1.0 / (time_dim * 2))
            

            if ts.size(0) == 1:
                harmonic = harmonic.squeeze(0)
            
            return harmonic
            
        except Exception as e:
            logging.error(f"Error in TimeEncode forward: {str(e)}")

            if ts.dim() == 1:
                return torch.zeros(ts.size(0), len(self.omega) * 2, device=ts.device)
            else:
                return torch.zeros(ts.size(0), ts.size(1), len(self.omega) * 2, device=ts.device)


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

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
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

    def __init__(self, feat_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.2):

        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = (feat_dim + time_dim)


        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)


        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
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

    def __init__(self, feat_dim, time_dim, n_head=2, drop_out=0.2, lambda_weight=0.5):
        super(DualMessageAggregator, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.model_dim = feat_dim + time_dim
        self.n_head = n_head
        self.lambda_weight = lambda_weight
        

        self.edge_attention = MultiHeadAttention(
            n_head=n_head, 
            d_model=self.model_dim,
            d_k=self.model_dim // n_head,
            d_v=self.model_dim // n_head,
            dropout=drop_out
        )
        

        self.mean_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model_dim + feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )
        
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model_dim + feat_dim, feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feat_dim, feat_dim)
        )
        
    def forward(self, src_node_feat, src_node_t_embed, ngh_feat, ngh_t_embed, path_weights, mask):
        self.logger.info(f"[DualMessageAggregator] src_node_feat: {src_node_feat.shape}, src_node_t_embed: {src_node_t_embed.shape}, ngh_feat: {ngh_feat.shape}, ngh_t_embed: {ngh_t_embed.shape}, path_weights: {path_weights.shape}, mask: {mask.shape}")
        try:
            device = src_node_feat.device
            batch_size = src_node_feat.size(0)
            

            messages = torch.cat([ngh_feat, ngh_t_embed], dim=2)
            

            valid_mask = ~mask
            path_weights_masked = path_weights * valid_mask.float()
            

            weight_sum = path_weights_masked.sum(dim=1, keepdim=True) + 1e-8
            normalized_weights = path_weights_masked / weight_sum
            

            mean_aggregated = torch.sum(
                messages * normalized_weights.unsqueeze(-1), dim=1
            )
            

            mean_combined = torch.cat([mean_aggregated, src_node_feat], dim=1)
            h_mean = self.mean_mlp(mean_combined)


            src_combined = torch.cat([src_node_feat, src_node_t_embed], dim=1)
            query = src_combined.unsqueeze(1)

            attn_mask = mask.unsqueeze(1)
            attn_output, attn_weights = self.edge_attention(
                q=query, k=messages, v=messages, mask=attn_mask
            )
            

            attn_output = attn_output.squeeze(1)

            attn_combined = torch.cat([attn_output, src_node_feat], dim=1)
            h_attention = self.attention_mlp(attn_combined)
            

            final_embedding = (
                self.lambda_weight * h_mean + 
                (1 - self.lambda_weight) * h_attention
            )
            
            return final_embedding, attn_weights.squeeze(1) if attn_weights is not None else None
            
        except Exception as e:
            self.logger.error(f"Error in DualMessageAggregator forward: {str(e)}")

            return src_node_feat, None


class CLGNN(torch.nn.Module):

    def __init__(self, ngh_finder, n_feat, num_layers=2, use_time='time', 
                 n_head=4, drop_out=0.1, lambda_weight=0.5, alpha=0.3):

        super(CLGNN, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        
        try:

            self.num_layers = num_layers
            self.ngh_finder = ngh_finder
            self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=True)
            self.feat_dim = self.n_feat_th.shape[1]
            self.lambda_weight = lambda_weight
            self.alpha = alpha
            

            if use_time == 'time':
                self.logger.info('Using continuous time encoding (Bochner\'s theorem)')
                self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
            else:
                raise ValueError('Only time encoding is supported')
            

            self.message_agg = torch.nn.ModuleList([
                DualMessageAggregator(
                    feat_dim=self.feat_dim,
                    time_dim=self.feat_dim,
                    n_head=n_head,
                    drop_out=drop_out
                ) for _ in range(num_layers)
            ])
            

            self.path_encoder = PathCountEncode(expand_dim=self.feat_dim)
            

            self.value_net = ValueNet(
                input_dim=self.feat_dim,
                hidden_dim=64,
                drop_out=drop_out
            )
            

            self.k_contrast_net = KContrastNet(
                embed_dim=self.feat_dim,
                tau=0.1,
                gamma_pos=0.5,
                gamma_neg=0.5,
                B=10,
                sample_ratio=0.4
            )
            

            self.init_weights()
            
        except Exception as e:
            self.logger.error(f"Error initializing CLGNN: {str(e)}")
            raise
            
    def init_weights(self):

        try:
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        except Exception as e:
            self.logger.error(f"Error initializing weights: {str(e)}")
            raise
            
    def tem_conv(self, src_idx_l, src_node_features, src_ngh_node_batch, src_ngh_t_batch, src_ngh_feat, cut_time_l, curr_layers):
        self.logger.info(f"[tem_conv] src_idx_l: {src_idx_l.shape}, src_node_features: {src_node_features.shape}, src_ngh_node_batch: {src_ngh_node_batch.shape}, src_ngh_t_batch: {src_ngh_t_batch.shape}, src_ngh_feat: {src_ngh_feat.shape}, cut_time_l: {cut_time_l.shape}, curr_layers: {curr_layers}")
        try:
            device = src_idx_l.device
            batch_size = src_idx_l.size(0) if len(src_idx_l.size()) > 0 else 1
            num_neighbors = src_ngh_node_batch.size(1) if len(src_ngh_node_batch.size()) > 1 else 1

            if len(src_idx_l.size()) == 0:
                src_idx_l = src_idx_l.unsqueeze(0)
            if len(src_node_features.size()) == 1:
                src_node_features = src_node_features.unsqueeze(0)
            if len(src_ngh_node_batch.size()) == 1:
                src_ngh_node_batch = src_ngh_node_batch.unsqueeze(0)
            if len(src_ngh_t_batch.size()) == 1:
                src_ngh_t_batch = src_ngh_t_batch.unsqueeze(0)
            if len(src_ngh_feat.size()) == 2:
                src_ngh_feat = src_ngh_feat.unsqueeze(0)
            if len(cut_time_l.size()) == 0:
                cut_time_l = cut_time_l.unsqueeze(0)
            

            if curr_layers == 0:
                return src_node_features
            

            time_diffs = cut_time_l.unsqueeze(1) - src_ngh_t_batch
            

            time_encodings = self.time_encoder(time_diffs)
            

            path_counts = self.compute_path_counts(src_idx_l, src_ngh_node_batch, src_ngh_t_batch, cut_time_l)


            path_weights = F.softmax(path_counts, dim=1)
            

            mask = (src_ngh_node_batch == 0)
            

            src_node_t_embed = time_encodings[:, 0, :]
            

            curr_embeddings, attn_weights = self.message_agg[curr_layers - 1](
                src_node_features,
                src_node_t_embed,
                src_ngh_feat,
                time_encodings,
                path_weights,
                mask
            )
            
            return curr_embeddings
            
        except Exception as e:
            self.logger.error(f"Error in temporal convolution: {str(e)}")

            return src_node_features

    def forward(self, src_idx_l, cut_time_l, tbc_labels=None, num_neighbors=20):
        self.logger.info(f"[forward] src_idx_l: {type(src_idx_l)}, cut_time_l: {type(cut_time_l)}, tbc_labels: {type(tbc_labels)}, num_neighbors: {num_neighbors}")
        try:
            device = self.n_feat_th.device
            batch_size = len(src_idx_l)
            

            if isinstance(src_idx_l, torch.Tensor):
                src_idx_l = src_idx_l.cpu().numpy()
            elif isinstance(src_idx_l, list):
                src_idx_l = np.array(src_idx_l)
                
            if isinstance(cut_time_l, torch.Tensor):
                cut_time_l = cut_time_l.cpu().numpy()
            elif isinstance(cut_time_l, list):
                cut_time_l = np.array(cut_time_l)
            

            src_ngh_node_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l, 
                cut_time_l, 
                num_neighbors=num_neighbors
            )

            src_idx_l = torch.from_numpy(src_idx_l).long().to(device)
            cut_time_l = torch.from_numpy(cut_time_l).float().to(device)
            src_ngh_node_batch = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_t_batch = torch.from_numpy(src_ngh_t_batch).float().to(device)

            src_node_features = self.n_feat_th[src_idx_l]

            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()

            src_ngh_feat_flat = self.n_feat_th[src_ngh_node_batch_flat]

            ngh_ngh_node_batch, ngh_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_ngh_node_batch_flat.cpu().numpy(),
                src_ngh_t_batch_flat.cpu().numpy(),
                num_neighbors=num_neighbors
            )

            ngh_ngh_node_batch = torch.from_numpy(ngh_ngh_node_batch).long().to(device)
            ngh_ngh_t_batch = torch.from_numpy(ngh_ngh_t_batch).float().to(device)

            src_ngh_feat = self.tem_conv(
                src_ngh_node_batch_flat,
                src_ngh_feat_flat,
                ngh_ngh_node_batch,
                ngh_ngh_t_batch,
                self.n_feat_th[ngh_ngh_node_batch].view(-1, num_neighbors, self.feat_dim),
                src_ngh_t_batch_flat,
                self.num_layers - 1
            )

            node_embeddings = self.tem_conv(
                src_idx_l,
                src_node_features,
                src_ngh_node_batch,
                src_ngh_t_batch,
                src_ngh_feat.view(batch_size, num_neighbors, -1),
                cut_time_l,
                self.num_layers
            )
            

            tbc_scores = self.value_net(node_embeddings)

            if tbc_labels is not None:
                if isinstance(tbc_labels, np.ndarray):
                    tbc_labels = torch.from_numpy(tbc_labels).float().to(device)
                elif isinstance(tbc_labels, list):
                    tbc_labels = torch.tensor(tbc_labels, dtype=torch.float32, device=device)

                if tbc_labels.dim() > 1:
                    tbc_labels = tbc_labels.squeeze()
                if tbc_scores.dim() > 1:
                    tbc_scores = tbc_scores.squeeze()

                contrastive_loss = self.k_contrast_net(node_embeddings, tbc_labels)

                total_loss, regression_loss = self.compute_total_loss(tbc_scores, tbc_labels, contrastive_loss)
                
                self.logger.info(f"[forward] tbc_scores: {tbc_scores.shape}, tbc_labels: {tbc_labels.shape}")
                return total_loss, tbc_scores, node_embeddings
            
            self.logger.info(f"[forward] src_ngh_node_batch: {src_ngh_node_batch.shape}, src_ngh_t_batch: {src_ngh_t_batch.shape}")
            self.logger.info(f"[forward] src_node_features: {src_node_features.shape}, src_ngh_feat_flat: {src_ngh_feat_flat.shape}")
            self.logger.info(f"[forward] ngh_ngh_node_batch: {ngh_ngh_node_batch.shape}, ngh_ngh_t_batch: {ngh_ngh_t_batch.shape}")
            self.logger.info(f"[forward] src_ngh_feat: {src_ngh_feat.shape}")
            self.logger.info(f"[forward] node_embeddings: {node_embeddings.shape}")
            return tbc_scores, node_embeddings
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            if tbc_labels is not None:
                return torch.tensor(0.0, device=device), torch.zeros(batch_size, device=device), torch.zeros((batch_size, self.feat_dim), device=device)
            return torch.zeros(batch_size, device=device), torch.zeros((batch_size, self.feat_dim), device=device)

    def compute_total_loss(self, predicted_tbc, true_tbc, contrastive_loss):

        try:
            device = predicted_tbc.device

            if not isinstance(predicted_tbc, torch.Tensor):
                predicted_tbc = torch.tensor(predicted_tbc, dtype=torch.float32, device=device)
            if not isinstance(true_tbc, torch.Tensor):
                true_tbc = torch.tensor(true_tbc, dtype=torch.float32, device=device)
            if not isinstance(contrastive_loss, torch.Tensor):
                contrastive_loss = torch.tensor(contrastive_loss, dtype=torch.float32, device=device)

            if predicted_tbc.dim() == 2 and predicted_tbc.size(1) == 1:
                predicted_tbc = predicted_tbc.view(-1)
            elif predicted_tbc.dim() > 1:
                predicted_tbc = predicted_tbc.squeeze()
            if true_tbc.dim() == 2 and true_tbc.size(1) == 1:
                true_tbc = true_tbc.view(-1)
            elif true_tbc.dim() > 1:
                true_tbc = true_tbc.squeeze()

            if predicted_tbc.shape != true_tbc.shape:
                raise ValueError(f"Shape mismatch: predicted_tbc {predicted_tbc.shape} != true_tbc {true_tbc.shape}")

            if torch.isnan(predicted_tbc).any() or torch.isinf(predicted_tbc).any():
                self.logger.warning("NaN/Inf values detected in predicted_tbc")
                predicted_tbc = torch.nan_to_num(predicted_tbc, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(true_tbc).any() or torch.isinf(true_tbc).any():
                self.logger.warning("NaN/Inf values detected in true_tbc")
                true_tbc = torch.nan_to_num(true_tbc, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                self.logger.warning("NaN/Inf values detected in contrastive_loss")
                contrastive_loss = torch.tensor(0.0, device=device)

            eps = 1e-6
            log_pred = torch.log(predicted_tbc + eps)
            log_true = torch.log(true_tbc + eps)

            regression_loss = F.l1_loss(log_pred, log_true)

            scaled_reg_loss = torch.clamp(regression_loss, min=0.0, max=100.0)
            scaled_contrast_loss = torch.clamp(contrastive_loss, min=0.0, max=100.0)

            reg_scale = torch.exp(-scaled_reg_loss)
            contrast_scale = torch.exp(-scaled_contrast_loss)
            dynamic_alpha = self.alpha * (contrast_scale / (reg_scale + contrast_scale))

            total_loss = (1 - dynamic_alpha) * scaled_reg_loss + dynamic_alpha * scaled_contrast_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error("NaN/Inf values detected in final loss")
                return torch.tensor(1.0, device=device), regression_loss
            
            return total_loss, regression_loss
            
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            return torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

    def compute_path_counts(self, src_idx_l, src_ngh_node_batch, src_ngh_t_batch, cut_time_l):

        try:
            device = src_idx_l.device
            B, N = src_ngh_node_batch.size()
            

            path_counts = torch.zeros((B, N), device=device)
            

            for b in range(B):
                src = src_idx_l[b]
                cut_time = cut_time_l[b]

                valid_mask = src_ngh_t_batch[b] < cut_time
                valid_neighbors = src_ngh_node_batch[b][valid_mask]
                
                if len(valid_neighbors) > 0:
                    unique_neighbors, counts = torch.unique(valid_neighbors, return_counts=True)
                    for n, c in zip(unique_neighbors, counts):
                        neighbor_mask = src_ngh_node_batch[b] == n
                        path_counts[b][neighbor_mask] = c.float()
            
            return path_counts
            
        except Exception as e:
            self.logger.error(f"Error computing path counts: {str(e)}")
            return torch.zeros((B, N), device=device)


class KContrastNet(torch.nn.Module):
    def __init__(self, embed_dim, tau=0.1, gamma_pos=0.5, gamma_neg=0.5, B=10, sample_ratio=0.4):
        super(KContrastNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        
        try:
            self.embed_dim = embed_dim
            self.tau = tau
            self.gamma_pos = gamma_pos
            self.gamma_neg = gamma_neg
            self.B = B
            self.sample_ratio = sample_ratio

            self.proj = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.BatchNorm1d(embed_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, embed_dim),
                torch.nn.BatchNorm1d(embed_dim)
            )

            self.temperature = torch.nn.Parameter(torch.tensor([tau]))

            self.init_weights()
            
        except Exception as e:
            self.logger.error(f"Error initializing KContrastNet: {str(e)}")
            raise
            
    def init_weights(self):
        try:
            for m in self.proj:
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.BatchNorm1d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                    
        except Exception as e:
            self.logger.error(f"Error initializing weights: {str(e)}")
            raise
            
    def forward(self, embeddings, labels):
        try:
            device = embeddings.device
            batch_size = embeddings.size(0)

            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            if embeddings.dim() != 2:
                raise ValueError(f"Embeddings must be 2D tensor, got shape {embeddings.shape}")
            if embeddings.size(1) != self.embed_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embed_dim}, got {embeddings.size(1)}")
            
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32, device=device)
            if labels.dim() > 1:
                labels = labels.squeeze()
            if labels.size(0) != batch_size:
                raise ValueError(f"Label size mismatch: expected {batch_size}, got {labels.size(0)}")

            proj_embeddings = self.proj(embeddings)

            proj_embeddings = F.normalize(proj_embeddings, p=2, dim=1)

            sim_matrix = torch.matmul(proj_embeddings, proj_embeddings.t())

            sim_matrix = torch.clamp(sim_matrix, min=-1.0, max=1.0)

            label_diff = torch.abs(labels.unsqueeze(0) - labels.unsqueeze(1))
            label_sim = torch.exp(-label_diff / self.temperature)

            mask = torch.eye(batch_size, device=device)
            sim_matrix = sim_matrix * (1 - mask)
            label_sim = label_sim * (1 - mask)

            pos_threshold = torch.quantile(label_sim, 0.75)
            neg_threshold = torch.quantile(label_sim, 0.25)
            
            pos_mask = (label_sim > pos_threshold).float()
            neg_mask = (label_sim < neg_threshold).float() * (1 - mask)

            pos_count = torch.sum(pos_mask, dim=1) + 1e-8
            neg_count = torch.sum(neg_mask, dim=1) + 1e-8

            pos_exp = torch.exp(sim_matrix / self.temperature) * pos_mask
            neg_exp = torch.exp(sim_matrix / self.temperature) * neg_mask
            pos_sum = torch.sum(pos_exp, dim=1)
            neg_sum = torch.sum(neg_exp, dim=1)

            pos_sum = torch.clamp(pos_sum, min=1e-8)
            neg_sum = torch.clamp(neg_sum, min=1e-8, max=1-1e-8)

            pos_weight = torch.exp(-pos_sum)
            neg_weight = torch.exp(neg_sum)
            
            pos_loss = -(pos_weight * torch.log(pos_sum / pos_count)).mean()
            neg_loss = -(neg_weight * torch.log(1 - neg_sum / neg_count)).mean()

            contrastive_loss = pos_loss + neg_loss

            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                self.logger.warning("NaN/Inf values detected in contrastive loss, returning 0.0")
                return torch.tensor(0.0, device=device)
            
            return contrastive_loss
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            return torch.tensor(0.0, device=device)


class ValueNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, drop_out=0.1):
        super(ValueNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        
        try:
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.BatchNorm1d(hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(drop_out),
                torch.nn.Linear(hidden_dim // 2, 1)
            )

            self.init_weights()
            
        except Exception as e:
            self.logger.error(f"Error initializing ValueNet: {str(e)}")
            raise
            
    def init_weights(self):
        try:
            for m in self.mlp:
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.BatchNorm1d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                    
        except Exception as e:
            self.logger.error(f"Error initializing weights: {str(e)}")
            raise
            
    def forward(self, x):
        try:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.dim() != 2:
                raise ValueError(f"Input tensor must be 2D, got shape {x.shape}")
            if x.size(1) != self.input_dim:
                raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.size(1)}")

            x = F.normalize(x, p=2, dim=1)

            scores = self.mlp(x)

            scores = F.softplus(scores) + 1e-6

            if scores.dim() == 2 and scores.size(1) == 1:
                scores = scores.view(-1)

            if torch.isnan(scores).any() or torch.isinf(scores).any():
                self.logger.warning("NaN/Inf values detected in scores")
                scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            if x.dim() == 2:
                return torch.zeros(x.size(0), device=x.device)
            else:
                return torch.zeros(1, device=x.device)
