# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from feature_preprocess import *

#import os, sys
#lib_path = os.path.abspath(os.path.join('..','..'))
#sys.path.append(lib_path)

from adan.functions import *



def applySingleInputFunction(funcs,df): 
    df=df.copy()    
    columns=df.select_dtypes(include=[np.number]).columns
    for column in columns:       
        for f in funcs:
            try:
                df[column+f.__name__]=f(df[column])
            except:
                pass

    return df


def applyDoubleInputFunction(funcs,df):
    df=df.copy()
    columns=df.select_dtypes(include=[np.number]).columns   
    
    for f in funcs:
        cols_to_avoid=[]
        for column1 in columns:
            cols_to_avoid.append(column1)
            for column2 in columns.drop(cols_to_avoid):
                    k=df[column1]
                    l=df[column2]
                    try:
                         df[column1+"_"+f.__name__+"_"+column2]=(f(k,l))
                    except:
                        pass
    return df
        

def aggregatePerColumn(functions,df):
    df=df.copy()
    columns=df.select_dtypes(include=['category']).columns
    groups=[]
    for column in columns:
        try:
            g=df.groupby(by=column,as_index=False).agg(functions)
            g.columns = ['_'.join(col).strip()+"_over_"+column for col in g.columns.values]
            g[column]=g.index
            groups.append(g)
        except:
            print("error for column:"+column+" in aggregatePerColumn")
            pass
    for g,column in zip(groups,columns):
        df=pd.merge(df,g,on=column,how="left")
    return df
    

def createFeatures(df,singlefunctions=singlefunctions,
                   twopartfunctions=twopartfunctions, aggregationfunctions=aggregationfunctions):    

    dfs=[]
    if len(singlefunctions)>0:
        print("doing colssingle")
        colssingle=applySingleInputFunction(singlefunctions,df)
        dfs.append(colssingle)
    if len(twopartfunctions)>0:
        print("doing colsdouble")
        colsdouble=applyDoubleInputFunction(twopartfunctions,df)
        dfs.append(colsdouble)
    if len(aggregationfunctions)>0:
        print("doing colsagg")
        colsagg=aggregatePerColumn(aggregationfunctions,df)
        dfs.append(colsagg)
    
    #for i in range(0,len(dfs)):
    #    dfs[i]=generalPreprocess(dfs[i])
        
    dfnew=pd.concat(dfs,axis=1)
    return dfnew
