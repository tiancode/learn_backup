import math
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

# k-Nearest Neighbor算法
def k_nearest_neighbors(data, predict, k=5):
    
    if len(data) >= k:
        warnings.warn("k is too small")
    
    # 计算predict点到各点的距离
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    sorted_distances =[i[1]  for i in sorted(distances)]
    top_nearest = sorted_distances[:k]
    
    group_res = Counter(top_nearest).most_common(1)[0][0]
    confidence = Counter(top_nearest).most_common(1)[0][1]*1.0/k
    
    return group_res, confidence

if __name__=='__main__':
    df = pd.read_csv('breast-cancer-wisconsin.data')  # 加载数据
    #print(df.head()
    #print(df.shape)
    df.replace('?', np.nan, inplace=True)  # -99999
    df.dropna(inplace=True)  # 去掉无效数据
    #print(df.shape)
    df.drop(['id'], 1, inplace=True)
    
    # 把数据分成两部分，训练数据和测试数据
    full_data = df.astype(float).values.tolist()
    
    random.shuffle(full_data)
    
    test_size= 0.2   # 测试数据占20%
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]
    
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    
    correct = 0
    total = 0
    
    for group in test_set:
        for data in test_set[group]:
            res,confidence = k_nearest_neighbors(train_set, data, k = 5)
            if group == res:
                correct += 1
            else:
                print(confidence)
            total += 1

    print(correct/total)
    
    print(k_nearest_neighbors(train_set, [4,2,1,1,1,2,3,2,1], k = 5))
