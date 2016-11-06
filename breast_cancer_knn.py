import math
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn("k is too small")
    
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    sorted_distances =[i[1]  for i in sorted(distances)]
    top_nearest = sorted_distances[:k]
    #print(top_nearest)  ['r','k','r']
    return Counter(top_nearest).most_common(1)[0][0]


df = pd.read_csv('breast-cancer-wisconsin.data')
#print(df.head())

#print(df.shape)
df.replace('?', np.nan, inplace=True)  # -99999
df.dropna(inplace=True)
#print(df.shape)

df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size= 0.2
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
        res = k_nearest_neighbors(test_set, data, k = 5)
        if group == res:
            correct += 1
        total += 1

print(correct/total)

