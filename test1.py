import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import Imputer
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

#The dataset is stored in the Dataset Folder, in train.csv file

cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/train.csv")
test = genfromtxt(path1 ,delimiter=',')

# remove the data labels from the dataset
testm=test[1:,1:]
print testm.shape
# the size of the train dataset is
'''Part 1: Filling missing Values'''
#using the imputer function to replace the missing values, we are using the mean of the columns to fill out the data
imp = Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)
Y=imp.fit_transform(testm)
path2 = os.path.join(cwd, "../Dataset/mean_imp.csv")
np.savetxt(path2, Y, delimiter=",")

'''
Yt=np.transpose(Y)

#print Yt[2600] # these are the labels for the given data

fig = pyplot.figure()
ax = fig.add_subplot(111)

pyplot.scatter(Y[0],Y[1])
for i,j in zip(Y[0],Y[1]):
    ax.annotate(Y[2600],xy=(i,j))

pyplot.show()
'''
