# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from feature_preprocess import *

#import os, sys
#lib_path = os.path.abspath(os.path.join('..','..'))
#sys.path.append(lib_path)

from adan.functions import *

        

def aggregatePerColumn(functions,df):
    df=df.copy()
    columns=df.select_dtypes(include=['category']).columns
    groups=[]
    for column in columns:
#        try:
#            g=df.groupby(by=column,as_index=False).agg(functions)
#            g.columns = ['_'.join(col).strip()+"_over_"+column for col in g.columns.values]
#            g[column]=g.index
#            groups.append(g)
#        except:
#            print("error for column:"+column+" in aggregatePerColumn")
#            pass
        g=df.groupby(by=column,as_index=False).agg(functions)
        g.columns = ['_'.join(col).strip()+"_over_"+column for col in g.columns.values]
        g[column]=g.index
        groups.append(g)
        
    for g,column in zip(groups,columns):
        df=pd.merge(df,g,on=column,how="left")
    return df
    

def createFeatures(df,aggregationfunctions=aggregationfunctions):    

    colsagg=aggregatePerColumn(aggregationfunctions,df)

    return colsagg
