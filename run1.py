import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import Imputer
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#The dataset is stored in the Dataset Folder, in train.csv file

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/mean_imp.csv")
test = genfromtxt(path1 ,delimiter=',')

# remove the data labels from the dataset

print test.shape
# the size of the train dataset is



Yt=np.transpose(test)
X=Yt[:-1]
test1=np.transpose(X)
print Yt[2600] # these are the labels for the given data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X1, y = test1, Yt[2600]
X_new = SelectKBest(chi2, k=2).fit_transform(X1, y)
print X_new.shape
X_new=X_new[:1000]
T=np.transpose(X_new)


fig = pyplot.figure()
ax = fig.add_subplot(111)
for i in range(len(T[0])):

    if (Yt[2600][i]==0):
        pyplot.scatter(T[0][i],T[1][i],c='r')
    if (Yt[2600][i]==1):
        pyplot.scatter(T[0][i],T[1][i],c='g')
    if (Yt[2600][i]==2):
        pyplot.scatter(T[0][i],T[1][i],c='b')
'''for i,j in zip(T[0],T[1]):
    ax.annotate(Yt[2600][i],xy=(i,j))
'''
'''for i in range(len(T[0])):
    ax.annotate(Yt[2600][i],xy=(T[0][i],T[1][i]))
'''
pyplot.show()
