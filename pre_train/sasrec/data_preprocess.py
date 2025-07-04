# -*- coding: utf-8 -*-

import os
import os.path
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    g = gzip.open(path, 'rb')
    for l in tqdm(g):
        yield json.loads(l)
        
def preprocess(fname):
    """预处理数据"""
    countU = defaultdict(lambda: 0) # 每个用户的交互次数
    countP = defaultdict(lambda: 0) # 每个商品的交互次数
    line = 0 # 处理的行数

    file_path = f'/root/repo/A-LLMRec/data/Amazon/{fname}.json.gz'

    # 统计用户和商品的交互次数
    # counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        if ('Beauty' in fname) or ('Toys' in fname):
            if l['overall'] < 3:
                continue
        asin = l['asin'] # 商品的唯一标识符
        rev = l['reviewerID'] # 用户的唯一标识符
        time = l['unixReviewTime'] # 评论的时间戳

        countU[rev] += 1
        countP[asin] += 1
    
    # 初始化映射字典和存储结构
    usermap = dict() # 用户ID到整数索引的映射
    usernum = 0 # 用户数量 
    itemmap = dict() # 商品ID到整数索引的映射
    itemnum = 0 # 商品数量
    User = dict() # 存储每个用户的交互记录
    review_dict = {} # 存储每个商品的用户评论和总结
    name_dict = {'title':{}, 'description':{}}
    
    f = open(f'/root/repo/A-LLMRec/data/Amazon/meta_{fname}.json', 'r')
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data]
    meta_dict = {}
    
    for l in data_list:
        meta_dict[l['asin']] = l
    
    # 构建用户-商品交互序列
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        
        threshold = 5
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 4
            
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []
        
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])
        
        
        if itemmap[asin] in review_dict:
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                a = 0
        else:
            review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                a = 0
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                a = 0
        try:
            if len(meta_dict[asin]['description']) ==0:
                name_dict['description'][itemmap[asin]] = 'Empty description'
            else:
                name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0]
            name_dict['title'][itemmap[asin]] = meta_dict[asin]['title']
        except:
            a =0
    
    # 保存商品文本信息
    with open(f'/root/repo/A-LLMRec/data/Amazon/{fname}_text_name_dict.json.gz', 'wb') as tf:
        pickle.dump(name_dict, tf)
    
    # 按照时间排序并保存交互序列
    # 1. 对每个用户的交互记录按照时间戳排序
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
    
    # 2. 打印用户数量和商品数量
    print(usernum, itemnum)
    
    # 3. 将用户-商品交互序列保存为文本文件
    f = open(f'/root/repo/A-LLMRec/data/Amazon/{fname}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1])) # 用户-商品交互序列，每行包含用户ID和商品ID，用空格分隔
    f.close()
