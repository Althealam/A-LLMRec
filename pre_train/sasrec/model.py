# -*- coding: utf-8 -*-

import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # 1x1卷积层，相当于全连接层
        self.dropout1 = torch.nn.Dropout(p=dropout_rate) # Dropout层
        self.relu = torch.nn.ReLU() # ReLU激活函数，引入非线性
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1) # 1x1卷积层，相当于全连接层
        self.dropout2 = torch.nn.Dropout(p=dropout_rate) # Dropout层

    def forward(self, inputs):
        """前向传播"""
        # 对输入进行卷积, Dropout, 激活等操作，最后将结果与输入相加（残差连接）
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs
    
class SASRec(torch.nn.Module):
    """Self-Attentive Sequential Recommendation，是一个基于自注意力机制的序列推荐模型"""
    def __init__(self, user_num, item_num, args):
        """
        :param user_num: 用户数量
        :param item_num: 物品数量
        :param args: 包含各种超参数的对象
        """
        super(SASRec, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # 物品嵌入层，将物品ID转换为固定长度的向量
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        # 位置嵌入向量：为输入序列中的每个位置生成对应的位置嵌入
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        # 嵌入层的Dropout层
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 多头自注意力层前的LN模块列表
        self.attention_layernorms = torch.nn.ModuleList()
        # 多头自注意力层模块列表，计算序列中元素之间的注意力分数
        self.attention_layers = torch.nn.ModuleList()
        # 前馈网络前的层归一化模块列表
        self.forward_layernorms = torch.nn.ModuleList()
        # 最后一层的层归一化层
        self.forward_layers = torch.nn.ModuleList()
        # 最后一层的层归一化层
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args =args
        
        
        # Transformer块
        for _ in range(args.num_blocks):
            # 层归一化
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            # 多头自注意力层：计算序列中元素之间的注意力分数
            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            # 层归一化
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            # 前馈网络层
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        """前向传播方法：将输入的用户行为序列转换为特征表示"""
        # 嵌入相加：将物品嵌入和位置嵌入相加，并进行缩放和Dropout操作
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # 掩码操作：对填充位置进行掩码
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # 多头自注意力计算：通过多个Transformer块，依次进行多头自注意力计算和前馈网络计算，每块中包含残差连接和层归一化
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default'):
        """
        调用log2feats方法得到序列特征
        mode的不同处理逻辑：
        * log_only：返回序列的最后一个特征向量
        * item: 返回重塑后的序列特征、正样本embedding和负样本embedding
        * default：计算正样本和负样本的logits
        """
        log_feats = self.log2feats(log_seqs)
        if mode == 'log_only':
            log_feats = log_feats[:, -1, :]
            return log_feats
            
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        if mode == 'item':
            return log_feats.reshape(-1, log_feats.shape[2]), pos_embs.reshape(-1, log_feats.shape[2]), neg_embs.reshape(-1, log_feats.shape[2])
        else:
            return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """
        调用log2feats方法得到序列特征，取最后一个特征向量
        计算该特征向量与候选物品嵌入的点积，得到预测的logits
        """
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
