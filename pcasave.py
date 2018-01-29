from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.metrics import f1_score
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
import os
from numpy import genfromtxt

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')
pca = PCA(n_components=100)
datatr=pca.fit_transform(X)
print datatr .shape
#t=pca.fit(testa)
'''
datatr=pca.fit_transform(X)
print datatr .shape
#datate=pca.fit_transform(testa)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.10,max_depth=5, random_state=0)
scores = cross_val_score(clf, datatr, Y)
print scores.mean()
clf2=GradientBoostingClassifier(n_estimators=100, learning_rate=.1,max_depth=5, random_state=0).fit(datatr,Y)
y2=clf2.predict(datatr)
print f1_score(Y, y2, average='macro')'''

np.savetxt('PCA.csv',datatr,delimiter=',')
