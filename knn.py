import math
import numpy as np
from matplotlib import pyplot
from collections import Counter
import warnings


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
    
    group_res = Counter(top_nearest).most_common(1)[0][0]
    confidence = Counter(top_nearest).most_common(1)[0][1]*1.0/k

    return group_res, confidence

if __name__=='__main__':
    """
        p1 = [1, 3]
        p2 = [2, 5]
        
        d = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        print(d)
        """
    
    dataset = {'black':[ [1,2], [2,3], [3,1] ], 'red':[ [6,5], [7,7], [8,6] ]}
    new_features = [3.5,5.2]
    for i in dataset:
        for ii in dataset[i]:
            pyplot.scatter(ii[0], ii[1], s=50, color=i)

    which_group,confidence = k_nearest_neighbors(dataset, new_features, k=3)
    print(which_group, confidence)
    pyplot.scatter(new_features[0], new_features[1], s=100, color=which_group)
    
    pyplot.show()
