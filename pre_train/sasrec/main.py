# -*- coding: utf-8 -*-

import os
import time
import torch
import argparse
import numpy as np
from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':
    
    # ======== 数据预处理和统计 ===============
    # global dataset
    preprocess(args.dataset) # 预处理数据，将用户-物品交互记录转换为一对一的序列数据
    dataset = data_partition(args.dataset) # 数据划分为训练集、验证集、测试集

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # ============== 数据采样和模型初始化 =================
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3) # 使用WarpSampler进行数据采样    
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device) # 初始化SASRec模型并且移动到指定设备上
    
    # 对模型参数进行Xavier正态初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data) 
        except: 
            pass
    
    # 将模型设置为训练模式
    model.train()
    
    # 加载预训练模型
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    # 推理模式
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # 训练配置
    bce_criterion = torch.nn.BCEWithLogitsLoss() # 二元交叉熵损失函数
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98)) # Adam优化器
    
    T = 0.0
    t0 = time.time()
    
    # 训练循环
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)): # 遍历训练批次
        if args.inference_only: break
        for step in range(num_batch): # 遍历每个批次
            # 从采样器获取批量数据
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            # 输入模型得到正样本和负样本的logits
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            # 计算损失，包括二元交叉熵损失和L2正则化损失
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            # 反向传播更新模型参数
            loss.backward()
            adam_optimizer.step()
            if step % 100 == 0: # 每次间隔100步打印一次损失值
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        # 每20个轮次将模型设置为评估模式，进行验证和测试，并打印评估结果
        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    print("Done")