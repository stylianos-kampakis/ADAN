# -*- coding: utf-8 -*-
#from minepy import MINE
import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.feature_selection import f_classif
from adan.aidc.feature_selection import f_classifNumba
from numba.decorators import jit
from adan.metrics.regression import corrNumba


def evalSymbRegCV(individual, targets,toolbox,cv=3):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,

    model=linear_model.LinearRegression()
    scores=cross_validation.cross_val_score(estimator=model, X=np.expand_dims(func,1), y=targets, cv=cv,scoring="r2")
    
    return np.mean(scores),
 
        
def evalPearsonCor(individual, targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,

    score=np.corrcoef(func,targets )[0][1]
    if np.isnan(score):
        score=0.0
    final=score

    return final,


def evalPearsonCorNumba(individual, targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return -2.0,
    
    if any(abs(func)>3.4028235e+38):
        return -2.0,
        
    score=corrNumba(func,targets)

    return abs(score),   



def evalANOVA(individual,targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,   
    #this returns the p-value but we use 1-x so that greater values are better
    #we have to use reshape(-1,1) because scikit learn needs arrays in the form [[0],[1.34],..etc.]
    score=1-f_classif(func.reshape(-1,1),targets)[1][0]
    if np.isnan(score):
        score=0.0

    return score,

def evalANOVANumba(individual,targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,
    
    #this returns the p-value but we use 1-x so that greater values are better
    #we have to use reshape(-1,1) because scikit learn needs arrays in the form [[0],[1.34],..etc.]
    score=1-f_classifNumba(func.reshape(-1,1),targets)[1][0]
    if np.isnan(score):
        score=0.0

    return score,


