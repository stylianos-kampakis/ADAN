# -*- coding: utf-8 -*-
from scipy import special
import numpy as np
from minepy import MINE 
import multiprocessing
import pandas as pd


import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)
from adan.metrics.regression import *
from adan.metrics.classification import *


import warnings
from sklearn.utils import (as_float_array, check_X_y, safe_sqr,safe_mask)



    
def f_classifNumba(X, y):
    """Compute the ANOVA F-value for the provided sample.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} shape = [n_samples, n_features]
        The set of regressors that will tested sequentially.

    y : array of shape(n_samples)
        The data matrix.

    Returns
    -------
    F : array, shape = [n_features,]
        The set of F values.

    pval : array, shape = [n_features,]
        The set of p-values.

    See also
    --------
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    """
    X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    return f_onewayNumba(*args)
    

@jit 
def f_onewayNumba(*args):
    """Performs a 1-way ANOVA.

    The one-way ANOVA tests the null hypothesis that 2 or more groups have
    the same population mean. The test is applied to samples from two or
    more groups, possibly with differing sizes.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    sample1, sample2, ... : array_like, sparse matrices
        The sample measurements should be given as arguments.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.

    Notes
    -----
    The ANOVA test has important assumptions that must be satisfied in order
    for the associated p-value to be valid.

    1. The samples are independent
    2. Each sample is from a normally distributed population
    3. The population standard deviations of the groups are all equal. This
       property is known as homoscedasticity.

    If these assumptions are not true for a given set of data, it may still be
    possible to use the Kruskal-Wallis H-test (`scipy.stats.kruskal`_) although
    with some loss of power.

    The algorithm is from Heiman[2], pp.394-7.

    See ``scipy.stats.f_oneway`` that should give the same results while
    being less efficient.

    References
    ----------

    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 14.
           http://faculty.vassar.edu/lowry/ch14pt1.html

    .. [2] Heiman, G.W.  Research Methods in Statistics. 2002.

    """
    n_classes = len(args)
    n_samples_per_class=np.array([])
    ss_alldata=0
    sums_args=np.array([])
    for a in args:
        n_samples_per_class=np.append(n_samples_per_class,a.shape[0])
        ss_alldata=ss_alldata+(a**2).sum(axis=0)
        sums_args=np.append(sums_args,a.sum(axis=0))
        
    #n_samples_per_class = np.array([a.shape[0] for a in args])
    n_samples = np.sum(n_samples_per_class)
    #ss_alldata = sum(safe_sqr(a).sum(axis=0) for a in args)
    #sums_args = [np.asarray(a.sum(axis=0)) for a in args]
    square_of_sums_alldata = sum(sums_args) ** 2
    
    square_of_sums_args=np.array([])
    for s in sums_args:
        square_of_sums_args=np.append(square_of_sums_args,s**2)
        
    #square_of_sums_args = [s ** 2 for s in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.
    for k, _ in enumerate(args):
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.)[0]
    if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
        warnings.warn("Features %s are constant." % constant_features_idx,
                      UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob        
        
def calcMICReg(df,target,col):
    """
    
    """
    m=MINE()
    if df[col].dtype.name=="category":
        g=df.groupby(by=[col])['_target_variable_'].mean()
        g=g.to_dict()
        X=df[col].values
        X=[g[x] for x in X]    
    else:
        X=df[col].values
    m.compute_score(X, target)
    
    return {col:m.mic()} 
  
 
 
def calcFReg(df,col):
    
    score=(1-f_regression(X,tar)[1])[0]
    
    return {col:score}
  
  
   
def calcMetricParallel(df,target,processes=1,metric=calcMICReg):
    columns=df.columns
    pool = multiprocessing.Pool(processes=processes)
    df2=df.copy()
    df2['_target_variable_']=target
    results=[pool.apply_async(metric,args=(df2,target,col)) for col in columns]
    
    scores={}
    for res in results:
        scores.update(res.get())
    scores=pd.Series(scores)
    scores=pd.DataFrame({'variable':scores.index,'value':scores})
    return scores
    
    
def _chooseQuantileBestHelper(df,scores,quant=0.9):
    """
    Chooses a percentage of the best features.
    
    params:
    @df:a dataframe produced by calcMetridForDf or calcMetricParallel
    """
    df2=df.copy()
    k=scores.ix[scores['value']>=scores.value.quantile(quant)]
    df2=df2[k['variable']]
    return(df2)
    
def _chooseNBestHelper(df,scores,n=100):
    """

    :param df:
    :param scores:
    :param n:
    :return:
    """
    if n>scores.shape[1]:
        n=scores.shape[1]

    df2 = df.copy()
    k = scores.sort_values(by=['value'],ascending=False)

    k=k[0:(n-1)]
    df2 = df2[k['variable']]
    return (df2)


def chooseBest(df,target,processes=6,metric=calcMICReg,quant=0.9,num_features=100,limit_n=None,verbose=False,method="quantile"):
    """

    :param df:
    :param target:
    :param processes:
    :param metric:
    :param quant:
    :param num_features:
    :param limit_n:
    :param verbose:
    :param method:
    :return:
    """
    if(limit_n is None or df.shape[0]<=limit_n):
        scores=calcMetricParallel(df,target,processes,metric)
    else:
        indices=np.random.randint(0,(df.shape[0]-1),limit_n)
        scores=calcMetricParallel(df.ix[indices,:],target[indices],processes,metric)
    if method=="quantile":
        return _chooseQuantileBestHelper(df,scores,quant),scores
    elif method=="num":
        return _chooseNBestHelper(df,scores,num_features),scores
    
  
#def cbfSelection(features,target,task):
#    k=len(features)
#    scores=[]
#    cormat=np.corrcoef(features)
#    i=0
#    for i in range(0,len(features)):
#        feat=features[i]
#        #we square as an addition to the original paper to make sure that 
#        #higher scores in absolute value are always better
#        if task=="regression":
#            cor=(k*np.corrcoef(feat,target)[0][1])**2  
#        elif task=="classification":
#            score=1-f_classif(feat.reshape(-1,1),target)[1][0]
#            if np.isnan(score):
#                score=0.0
#            cor=(k*score)**2
#        corfeat=cormat[i]
#        if len(corfeat[corfeat==1.0])==1:
#            corfeat=corfeat[~np.isnan(corfeat)]
#            #we square (addition to the original paper) so that cor is always >0
#            #otherwise, if negative it cannot be calculated. Plus, having a large cor
#            #either negative or positive is a bad thing.
#            corfeatmean=np.mean(corfeat)**2
#            denom=k+k*(k-1)*corfeatmean
#            cbf=(k*cor)/(np.sqrt(denom))
#            scores.append(cbf)
#        else:
#            scores.append(0.0)
#        
#    return scores


def cbfSelectionNumba(features,target,task=None,metric=None):

    if task is None and metric is None:
        raise Exception('You should either specify a task or a metric.')

    k=len(features)
    scores=np.zeros(k)
    cormat=multiple_corrNumba(np.array(features))

    for i in range(0,k):
        feat=features[i]
        #we square as an addition to the original paper to make sure that 
        #higher scores in absolute value are always better
        if metric!=None:
            cor=metric(target,feat)            
        if metric is None and task=="regression":
            cor=(k*corrNumba(feat,target))**2              
        elif metric is None and task=="classification":
            score=1-f_classifNumba(feat.reshape(-1,1),target)[1][0]
            if np.isnan(score):
                score=0.0
            cor=(k*score)**2
        corfeat=cormat[i]
        locations=np.where(corfeat==1)[0]
        #if a feature has a perfect correlation with some other feature
        #then this feature gets a score of -Inf unless this is the last feature in the set
        #This is used so that one of the features survives.
        if len(corfeat[corfeat==1.0])>1 and i<max(locations):
            scores[i]=-np.inf         
        else:
            #we square (addition to the original paper) so that cor is always >0
            #otherwise, if negative it cannot be calculated. Plus, having a large cor
            #either negative or positive is a bad thing anyway.
            corfeatmean=np.mean(corfeat)**2
            denom=k+k*(k-1)*corfeatmean
            cbf=(k*cor)/(np.sqrt(denom))
            scores[i]=cbf
                       
    return scores


def deleteCorrelatedFeatures(features,threshold=0.9):
    """
    :param features: a dataframe or a numpy array containing the input datac
    :param threshold:
    :return:
    """
    array=False
    try:
        #if dataframe
        features2=features.copy()
    except:
        #if numpy array
        features2=np.copy(features)
    to_remove=[]
    for i in range(0,features2.shape[1]-1):
        for j in range(i+1,features2.shape[1]-1):
            
            cor = corrNumba(features2.ix[:,i].values, features2.ix[:,j].values)
            if cor>threshold:
                to_remove.append(i)
                break

    features2.drop(features.columns[to_remove], axis=1,inplace=True)
    return features2,to_remove


def deleteUselessFeatures(df,target,limit_n=None,threshold=0.01,metric=calcMICReg,n_jobs=2):
    if (limit_n is None or df.shape[0] <= limit_n):
        scores = calcMetricParallel(df, target, n_jobs, metric)
    else:
        indices = np.random.randint(0, (df.shape[0] - 1), limit_n)
        scores = calcMetricParallel(df.ix[indices, :], target[indices], n_jobs, metric)

    df2=df.copy()

    to_remove=[]
    for score in scores.itertuples():
        if score.value<threshold:
            to_remove.append(score.variable)
           
    df2.drop(to_remove,axis=1,inplace=True)
    return df2,to_remove,scores
