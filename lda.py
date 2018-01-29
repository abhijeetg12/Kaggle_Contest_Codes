from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from numpy import genfromtxt
from sklearn.metrics import f1_score

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/KNN_train.csv")
path2 = os.path.join(cwd, "../Dataset/KNN_train_labels.csv")
X = np.loadtxt(path1)#genfromtxt(path1 ,delimiter=',')
Y = np.loadtxt(path2)#genfromtxt(path2 ,delimiter=',')
pca = PCA(n_components=39)
#t=pca.fit(testa)
datatr=pca.fit_transform(X)
clf = QuadraticDiscriminantAnalysis(priors=None, reg_param=.38, store_covariance=False, store_covariances=None, tol=0.001)
clf2= QuadraticDiscriminantAnalysis(priors=None, reg_param=.38, store_covariance=False, store_covariances=None, tol=0.001).fit(datatr,Y)
y2=clf2.predict(datatr)
scores = cross_val_score(clf  , datatr, Y)
print f1_score(Y, y2, average='macro')
print scores
print scores.mean()
