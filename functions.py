# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:31:09 2016

@author: stelios
"""
import numpy as np
import scipy
from scipy import stats


def checkWrongNumberFormat(x):
    if((any(np.isnan(x))) or (any(np.isinf(x)))):
        return True
    else:
        return False

def add(x,y):
    return x+y
    
def sub(x,y):
    return x-y
 
def mul(x,y):
    return (x*y)   
    
def div(left, right):
    res=np.array(left * 1.0 / right)
    k=np.isfinite(res)
    res[np.logical_not(k)]=0
    return res

def cube(x):
    return x**3

def recipr(x):
    if(not any(x==0)):
        return 1/x
    else:
        return np.zeros(len(x))+np.max(x)

    
def squareroot(x):
    res=np.sqrt(x)
    k=np.isfinite(res)
    res[np.logical_not(k)]=0    
    return np.sqrt(res)        

    
def makelog(x):
    if(all(x>=0)):
        return np.log(x+1)
    else:
        return np.zeros(len(x))+np.min(x)


singlefunctions=[np.square,cube,recipr,makelog,squareroot,np.cos,np.sin,np.abs]
twopartfunctions=[add,sub,mul,div]
#aggregationfunctions=[np.min,np.max,scipy.mean,scipy.stats.hmean,np.std,sum,scipy.stats.kurtosis,scipy.stats.skew]
aggregationfunctions=[np.min,np.max,scipy.mean,np.std,sum,scipy.stats.kurtosis,scipy.stats.skew]