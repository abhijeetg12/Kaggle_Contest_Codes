from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from numpy import genfromtxt
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')
pca = PCA(n_components=50)
#t=pca.fit(testa)
t=10
datatr=pca.fit_transform(X)
neigh = KNeighborsClassifier(n_neighbors=t)
clf= KNeighborsClassifier(n_neighbors=t).fit(datatr,Y)
#neigh.fit(X, y)
y2=clf.predict(datatr)
scores = cross_val_score(neigh  , datatr, Y)
print f1_score(Y, y2, average='macro')
print scores
print scores.mean()
