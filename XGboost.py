'''
-------
XGBOOST
-------
'''

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

X=np.loadtxt('train_features.csv')
y=np.loadtxt('train_labels.csv')
test=np.loadtxt('test_features.csv')

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.35,train_size=0.65)
xg=XGBClassifier(colsample_bytree=0.1,max_delta_step=0,colsample_bylevel=0.1,max_depth=15,learning_rate=0.1,silent=True,reg_alpha=0.5,subsample=0.9,gamma=1,min_child_weight=7)
depth=[10,15,25]
estim=[30,100,200]
gam=[0,0.05,0.5]
alph=[0.1,0.5,1]
learn_rate=[0.01,0.1,0.3]
sub_sample=[0.6,0.75,0.9]
bytree=[0.1,0.5,0.9]
bylevel=[0.1,0.5,0.9]
#lamb=[0.01,0.1,1]
lamb=[0,0.000001,0.00001]
delt=[0,0.1,1]
lamb=[0.0001,0.1,1,10]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
xg=XGBClassifier(n_estimators=20,reg_lambda=0.0001,colsample_bytree=0.1,max_delta_step=0,colsample_bylevel=0.1,max_depth=5,learning_rate=0.1,silent=True,reg_alpha=0.5,subsample=0.9,gamma=1,min_child_weight=9)
estim=[30,100,200]
for i in lamb : 
	xg.reg_lambda=i
	score1=cross_val_score(xg,Xtrain,ytrain,cv=3)
	print(i,np.mean(score1))
'''	for j in delt : 
		xg.max_delta_step=j
		score1=cross_val_score(xg,Xtrain,ytrain,cv=3)
		print(i,j,np.mean(score1))'''

#xg=XGBClassifier(n_estimators=100,reg_lambda=0.01,colsample_bytree=0.1,max_delta_step=0,colsample_bylevel=0.1,max_depth=5,learning_rate=0.1,silent=True,reg_alpha=1,subsample=0.9,gamma=1,min_child_weight=9)


scaler=preprocessing.StandardScaler().fit(Xtrain)
Xtrain1=scaler.transform(Xtrain)
Xtest1=scaler.transform(Xtest)
test1=scaler.transform(test)

pca=PCA(n_components=100)
pca.fit(Xtrain1)
Xtrain2=pca.transform(Xtrain1)
Xtest2=pca.transform(Xtest1)
test2=pca.transform(test1)


xg=XGBClassifier(n_estimators=100,reg_lambda=0.01,colsample_bytree=0.1,max_delta_step=0,colsample_bylevel=0.1,max_depth=10,learning_rate=0.05,silent=True,reg_alpha=1,subsample=0.9,gamma=10,min_child_weight=9)

xg.fit(Xtrain2,ytrain)

ypred=xg.predict(Xtrain2)
ypred1=xg.predict(Xtest2)
ypred2=xg.predict(test2)

print accuracy_score(ytrain,ypred)
print accuracy_score(ytest,ypred1)
print f1_score(ytrain,ypred,average='micro')
print f1_score(ytest,ypred1,average='micro')
print f1_score(ytrain,ypred,average='macro')
print f1_score(ytest,ypred1,average='macro')

''' 5,100
1.0
0.308177991581
1.0
0.308177991581
1.0
0.282350274192
'''

# max_depth, n_estimators
'''
(2, 30, 0.2424078606368168)
(2, 60, 0.26783877029923359)
(2, 100, 0.27721492885150822)
(6, 30, 0.27610127250322003)
(6, 60, 0.28569762619974026)
(6, 100, 0.29766267174169891)
(8, 30, 0.26658229718111248)
'''

#max_Depth, learning_rate
'''
(10, 0.01, 0.20777669289073186)
(10, 0.1, 0.24713473447714759)
(10, 0.3, 0.24341639265963025)
(15, 0.01, 0.20566250406989839)
(15, 0.1, 0.25054377960316737)
(15, 0.3, 0.24972028479444197)
(25, 0.01, 0.2068075641135915)
(25, 0.1, 0.24471487665635391)
(25, 0.3, 0.24875599525827696)
'''

# sub sample, min child weight
'''
(0.6, 3, 0.24923078484029448)
(0.6, 5, 0.25392267566390891)
(0.6, 7, 0.25716964314168506)
(0.75, 3, 0.25423154050010771)
(0.75, 5, 0.25456870550436356)
(0.75, 7, 0.25376035292306698)
(0.9, 3, 0.26055835896046026)
(0.9, 5, 0.26232976384192325)
(0.9, 7, 0.26444221935505802)
'''

# gamma, alpha
'''
(0.01, 0.1, 0.25783349434840375)
(0.01, 0.5, 0.25425862674031929)
(0.01, 1, 0.25635454470635305)
(0.1, 0.1, 0.25683866469040528)
(0.1, 0.5, 0.25601752233065428)
(0.1, 1, 0.25893240341052642)
(1, 0.1, 0.26445047973779506)
(1, 0.5, 0.2600699171420075)
(1, 1, 0.25312914998152986)
'''

#gamma , min child weight
'''
(0, 1, 0.25617342574592566)
(0, 6, 0.25699692055465101)
(0, 10, 0.25489375019800892)
(0.05, 1, 0.25536745077688755)
(0.05, 6, 0.2582876530965999)
(0.05, 10, 0.25343779554469897)
(0.5, 1, 0.26185020619210847)
(0.5, 6, 0.26039177706680144)
(0.5, 10, 0.25376571392230302)

'''

# by tree, by level
'''
(0.1, 0.1, 0.27418338577072715)
(0.1, 0.5, 0.27205123342648491)
(0.1, 0.9, 0.26865389512122889)
(0.5, 0.1, 0.27239972516745892)
(0.5, 0.5, 0.26770769530595584)
(0.5, 0.9, 0.26910880903595424)
(0.9, 0.1, 0.26800466882038798)
(0.9, 0.5, 0.27158675578259134)
(0.9, 0.9, 0.26962712015449403)
'''

#lambda, delta
'''
(0.01, 0, 0.26492916777752679)
(0.01, 0.1, 0.21067338291024137)
(0.01, 1, 0.2587723978262304)
(0.1, 0, 0.25439533861276969)
(0.1, 0.1, 0.21083245148494692)
(0.1, 1, 0.2615174742600897)
(1, 0, 0.25538115616233709)
(1, 0.1, 0.21181263725611946)
(1, 1, 0.25406647946488653)
'''

# alpha
'''
(0.1, 0.27028536296489181)
(0.5, 0.27432092125533447)
(1, 0.26379907419104148)
'''

# n_estim, min_cild_weight=7
'''
(30, 0.28404077772889141)
(60, 0.30250900936055464)
(100, 0.3128841117800239)
(150, 0.30867990602750223)
(200, 0.31077634208620242)
'''

# min_child_weight, n_estimators=100
'''
(7, 0.3128841117800239)
(8, 0.31110271298723668)
(9, 0.31531272115145398)
(10, 0.31207517544497465)
'''

#n_estim
'''
(30, 0.28161213443368277)
(100, 0.30850427069479069)
(200, 0.31627622802693472)
(210, 0.31644150428056816)
'''