###################### HEIRARCHICAL CLUSTERING ALGO ######################


import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift# as ms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

 #an accuracy measurement to compare the cluster_centers with the actual cluster centers you started with
 #centers = [[],[]]
 # # binary = pd.read_csv('D:/University Selection System/Datasets/Dataset of 2000 students/dataset.csv')
 # binary.drop('IsEntranceTestRequired', axis=1, inplace=True)
binary = pd.read_csv("D:/University Selection System/Datasets/TrialDataset_1_CSV.csv")
X= binary.values[:, 1:4]

 #
 # fig = plt.figure()
 # ax = fig.add_subplot(111,projection='3d')
 # ax.scatter(X[:,0],X[:,1],X[:,2])
 # plt.show()

 # initialize MeanShift, then we fit according to the dataset, "X"
ms = MeanShift()
ms.fit(X)
 # the labels are the ones the machine has chosen, these are not the same labels as the unpacked-to "y" variable
labels = ms.labels_
cluster_centers = ms.cluster_centers_

 #the cluster centers and grabing the total number of clusters
print(cluster_centers)
n_clusters_ = len(np.unique(labels))

print("Number of estimated clusters:", n_clusters_)

 #to graph the results, we want to have a nice list of colors to choose from.

 #This allows for 70 clusters, so that should be good enough.
colors = 10*['r','g','b','c','k','y','m']

print(colors)
print(labels)

 #This code is purely for graphing only.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):

    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
            marker="x",color='k', s=150, linewidths = 5, zorder=10)
ax.text(cluster_centers[0,0],cluster_centers[0,1],cluster_centers[0,2],
            '%s,%s,%s'  %(str(int(cluster_centers[0,0])),
                         str(int(cluster_centers[0,1])),
                         str(round(cluster_centers[0,2], 2))),
            size=7,color='b')
# ax.text(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
#             '%s' % (str(cluster_centers[:,1])),size=10,color='b')
# ax.text(cluster_centers[:,0],cluster_centers[:,1],cluster_centers[:,2],
#             '%s' % (str(cluster_centers[:,2])),size=10,color='b')
plt.show()

