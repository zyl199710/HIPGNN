import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_



class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)
        # 计算正弦和余弦编码
        sin_cos_encoding = torch.cat((torch.sin(pe), torch.cos(pe)), dim=1)
        # 将初始位置 e 和 sin_cos_encoding 合并
        eeig = torch.cat((e.unsqueeze(1), sin_cos_encoding), dim=1)
        return eeig, sin_cos_encoding


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x




class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none': 
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':    # Arxiv
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(ncombines)
        else:                  # Others
            self.norm = None 

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x



class TransformerLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out




class TransformerModule(nn.Module):
    def __init__(self, embed_size, output_size, heads, dropout, forward_expansion):
        super(TransformerModule, self).__init__()
        self.layer = TransformerLayer(embed_size,heads, dropout, forward_expansion)
        self.linear_tran = nn.Linear(embed_size, output_size)

    def forward(self, x):
        # In the case of graph data, value, key, and query are the same (the features learned by GCN)
        x = self.layer(x, x, x)
        return self.linear_tran(x)


class DotProductRelativePositionEncoding(nn.Module):
    def __init__(self, ):
        super().__init__()
        # self.size = size  # Assuming size is the length of the sequence

    def forward(self, position_embeddings):
        # Calculate the relative positions matrix
        # position_embeddings should be of shape (seq_len, embedding_dim)
        seq_len = position_embeddings.size(0)

        # Compute dot product between all pairs of position embeddings using einsum
        rel_position_scores = torch.einsum('ij,kj->ik', position_embeddings, position_embeddings)
        # Resulting shape will be (seq_len, seq_len)

        return rel_position_scores


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"

        self.rel_pos_encoder = DotProductRelativePositionEncoding()

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)

        # Final linear transformation after concatenating head outputs
        self.output_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.head_dim ** 0.5)

    def split_heads(self, x):
        """ Split the last dimension into (num_heads, head_dim) and remove batch dimension. """
        new_shape = (x.size(0), self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(1, 0, 2)  # (num_heads, seq_length, head_dim)

    def forward(self, query, key, value, sim):
        # Linear transformations
        query = self.queries(query)
        key = self.keys(key)
        value = self.values(value)

        # Split for multihead attention
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Relative position encoding
        relative_positions = self.rel_pos_encoder(sim)  # Shape needs to align with (seq_len, seq_len)
        relative_positions = relative_positions.unsqueeze(0) # Add batch and head dimension

        # Compute Q * K^T for scaled dot product attention
        QK_t = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Compute Q * Score^T for relative positional scores
        QScore_t = relative_positions * self.scale
        # print(QK_t.shape)
        # print(QScore_t.shape)
        # Combine QK^T and QScore^T, then scale
        attn_output_weights = QK_t + QScore_t
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        # Apply the attention to the values
        attn_output = torch.matmul(attn_output_weights, value)

        # Concatenate heads and project back to original embedding dimension
        attn_output = attn_output.permute(1, 0, 2).contiguous()  # Move back seq_length to the first dimension
        new_shape = attn_output.size(0), self.embed_dim
        attn_output = attn_output.reshape(*new_shape)
        attn_output = self.output_linear(attn_output)

        return attn_output, attn_output_weights

class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, hidden_dim1 = 64, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none', forward_expansion = 2):
        super(Specformer, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.hidden_dim1 = hidden_dim1
        
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)


        # 节点分类的Transformer
        self.node_classifier = TransformerModule(hidden_dim, hidden_dim1, nheads, feat_dropout, forward_expansion)

        # 链接预测的Transformer
        self.link_predictor = TransformerModule(hidden_dim, hidden_dim1, nheads, feat_dropout, forward_expansion) # 假设链接预测是二分类问题

        # 链接权重预测的Transformer
        self.weight_predictor = TransformerModule(hidden_dim, hidden_dim1, nheads, feat_dropout, forward_expansion) # 输出单一值表示权重


        self.eig_encoder = SineEncoding(hidden_dim)
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.mha_0 = CustomMultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim * forward_expansion, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(nheads+1, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList([SpecLayer(nheads+1, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])

        self.nodeclass = nn.Linear(hidden_dim1, nclass)

        self.mlp_link_weight = nn.Sequential(
            # nn.Linear(hidden_dim1 * 2, hidden_dim1),  # 输入维度为两个节点的嵌入拼接
            # nn.ReLU(),
            # nn.Dropout(feat_dropout),
            nn.Linear(hidden_dim1, 1)  # 输出维度为1，表示链接权重
        )
        self.c1 = torch.nn.Parameter(torch.Tensor([0.3]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.3]))
        self.c3 = torch.nn.Parameter(torch.Tensor([0.3]))
        self.c4 = torch.nn.Parameter(torch.Tensor([0.1]))
        

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig, sin_cos_encoding = self.eig_encoder(e)   # [N, d]
        eigw = self.eig_w(eig)
        mha_eig = self.mha_norm(eigw)
        # mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        mha_eig, attn = self.mha_0(mha_eig, mha_eig, mha_eig, sin_cos_encoding)
        eig = eigw + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)   # [N, m]

        # torch.save(new_e, 'new_eigen_unw_stringdb.pt')
        # if args.dataset == "CPDB":
        #     torch.save(new_e, 'new_eigen_w_cpdb.pt')
        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
            basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
            h = conv(basic_feats)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            node_classification = self.node_classifier(h)
            link_prediction = self.link_predictor(h)
            weight_prediction = self.weight_predictor(h)

            # weight_prediction = weight_prediction + 0.2 * link_prediction
            return node_classification, link_prediction, weight_prediction, self.c1, self.c2, self.c3, self.c4

    def predict_node(self, node_classification):


        # 通过MLP预测链接权重
        node_output = self.nodeclass(node_classification)

        return node_output

    def predict_link_weight(self, weight_prediction, edge_index):
        # 获取连接节点的嵌入
        src, dest = edge_index
        src_embedding, dest_embedding = weight_prediction[src], weight_prediction[dest]

        # 拼接节点嵌入
        concat_embedding = (src_embedding + dest_embedding) / 2


        # 通过MLP预测链接权重
        link_weight = self.mlp_link_weight(concat_embedding).squeeze(-1)

        return link_weight
