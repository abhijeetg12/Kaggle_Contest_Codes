import numpy as np
import theano.tensor as T
from theano import function
from theano import pp
import theano

a=T.dscalar('a')
b=T.dscalar('b')

c=a*b
f=function([a,b],c)
q=f(1.5,3)
print q
'''declaration of the function sigmoid'''
x=T.dmatrix('x')
s= 1/ (1+ T.exp(-x))
sigmoid=function([x],s)
