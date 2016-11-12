import numpy as np
from sklearn.cluster import MeanShift
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

centers = [[2,1,3], [6,6,6], [10,12,9]]
x,_ = make_blobs(n_samples=200, centers=centers, cluster_std=1)

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
#print(labels)
print(cluster_centers)

colors = ['r', 'g', 'b']
for i in range(len(x)):
    ax.scatter(x[i][0], x[i][1], x[i][2], c=colors[labels[i]])

ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='*', c='k', s=200, zorder=10)

pyplot.show()

