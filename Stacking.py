'''
--------
STACKING
--------
'''

from mlxtend.classifier import StackingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC # using a high level implementation of libsvm
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree 
from sklearn.preprocessing import Imputer
from fancyimpute import KNN,MICE,IterativeSVD
from sklearn.ensemble import BaggingClassifier 

Xtrain=np.loadtxt('KNN_train.csv')
ytrain=np.loadtxt('KNN_train_labels.csv')
Xtest=np.loadtxt('KNN_test1.csv')
ytest=np.loadtxt('KNN_test1_labels.csv')
test=np.loadtxt('KNN_test_actual.csv')

'''X=np.loadtxt('train_features.csv')
y=np.loadtxt('train_labels.csv')
t=np.loadtxt('test_features.csv')

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.35,train_size=0.65)'''

'''imp=Imputer(missing_values='NaN',strategy='median',axis=0) 
imp.fit(Xtrain)
Xtrain_1=imp.transform(Xtrain)
Xtest_1=imp.transform(Xtest)
t_1=imp.transform(t)'''

'''Xtrain_1=KNN(k=3,orientation='columns').complete(Xtrain)
Xtest_1=KNN(k=3,orientation='columns').complete(Xtest)
test_1=KNN(k=3,orientation='columns').complete(t)

Xtrain_1=MICE(n_imputations=4,n_burn_in=1).complete(Xtrain)
Xtest_1=MICE(n_imputations=4,n_burn_in=1).complete(Xtest)
test_1=MICE(n_imputations=4,n_burn_in=1).complete(t)

Xtrain_1=IterativeSVD().complete(Xtrain)
Xtest_1=IterativeSVD().complete(Xtest)
test_1=IterativeSVD().complete(t)'''


scaler=preprocessing.StandardScaler().fit(Xtrain)
Xtrain1=scaler.transform(Xtrain)
Xtest1=scaler.transform(Xtest)
test1=scaler.transform(test)

pca=PCA(n_components=50)
pca.fit(Xtrain1)
Xtrain2=pca.transform(Xtrain1)
Xtest2=pca.transform(Xtest1)
test2=pca.transform(test1)

gaus=SVC(gamma=0.001,C=0.005,kernel='rbf',probability=True)
lr=LogisticRegression(C=5)
neigh=KNeighborsClassifier(n_neighbors=20) #11
grad=GradientBoostingClassifier(n_estimators=10,max_depth=3,min_samples_split=7,verbose=0,learning_rate=0.05)
gb=GaussianNB()
qda=QuadraticDiscriminantAnalysis(reg_param=0.965) #0.965
qda1=QuadraticDiscriminantAnalysis(reg_param=0.9)
lda=LinearDiscriminantAnalysis()
mlp=MLPClassifier(hidden_layer_sizes=5)
dt=tree.DecisionTreeClassifier(max_depth=25,min_samples_leaf=15,presort=True)
bag=BaggingClassifier(qda1,n_estimators=5)
rf=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=10)
stack=StackingClassifier(classifiers=[gaus,lr,neigh,grad,gb,qda,lda,bag,rf],use_probas=True,verbose=2,meta_classifier=lr,use_features_in_secondary=False)
#no lda
stack.fit(Xtrain2,ytrain)
ypred=stack.predict(Xtrain2)
ypred1=stack.predict(Xtest2)
ypred2=stack.predict(test2)
print accuracy_score(ytrain,ypred)
print accuracy_score(ytest,ypred1)
print f1_score(ytrain,ypred,average='micro')
print f1_score(ytest,ypred1,average='micro')
print f1_score(ytrain,ypred,average='macro')
print f1_score(ytest,ypred1,average='macro')
 


ypred=stack.predict(Xtrain2)
ypred1=stack.predict(Xtest2)
ypred2=stack.predict(test2)


print accuracy_score(ytrain,ypred)
print accuracy_score(ytest,ypred1)
print f1_score(ytrain,ypred,average='micro')
print f1_score(ytest,ypred1,average='micro')
print f1_score(ytrain,ypred,average='macro')
print f1_score(ytest,ypred1,average='macro')




