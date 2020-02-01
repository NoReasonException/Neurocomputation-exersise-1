from collections import Counter

import numpy as np
from sklearn import datasets

def distance(instance1,instance2):
    instance1=np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1-instance2)

def compose_labels_data(labels,data):
    compose = []
    for i in range(len(data)):
        compose.append([data[i],labels[i]])
    return compose


def k_neighboors(labels,data,center,n):
    composed_data = compose_labels_data(labels,data)
    for i in range(len(composed_data)):
        composed_data[i]=[composed_data[i][0],composed_data[i][1],distance(composed_data[i][0],center)]
    composed_data.sort(key=lambda x:x[2])
    return composed_data[:n]


def vote(neighboors):
    cnt=Counter()
    for i in range(len(neighboors)):
        cnt[neighboors[i][1]]+=1/(neighboors[i][2]+1)
    return cnt.most_common(1)[0][0]


iris = datasets.load_iris()
data = iris.data
labels=iris.target

np.random.seed(84)
print(len(data))
indixes=np.random.permutation(len(data))
training_samples=100

learn_data = data[indixes[:-training_samples]]
learn_labels = labels[indixes[:-training_samples]] # auto edw einai magiko!

test_data = data[indixes[-training_samples:]]
test_labels = labels[indixes[-training_samples:]]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colours=("r","b")
X=[]
for iclass in range(3):
    X.append([[],[],[]])
    for i in range(len(learn_data)):
        if(learn_labels[i]==iclass):
            X[iclass][0].append(learn_data[i][0])
            X[iclass][1].append(learn_data[i][1])
            X[iclass][2].append(sum(learn_data[i][2:]))


colours = ("r","g","y")
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

for iclass in range(3):
    ax.scatter(X[iclass][0],X[iclass][1],X[iclass][2],c=colours[iclass])
plt.show()



print(distance(learn_data[0],learn_data[1]))
print(k_neighboors(learn_labels,learn_data,learn_data[2],5))
print(str(learn_data[2])+" result "+str(learn_labels[2]))
print(vote(k_neighboors(learn_labels,learn_data,test_data[2],10)))

suc=0
for i in range(len(test_labels)):

    result = vote(k_neighboors(learn_labels,learn_data,test_data[i],20))
    if(result==test_labels[i]):
        suc+=1

print(suc/len(test_labels))

