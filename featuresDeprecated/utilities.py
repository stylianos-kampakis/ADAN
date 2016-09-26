# -*- coding: utf-8 -*-
import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)

from adan.features.feature_preprocess import *
from adan.features.feature_creation import *
import pandas as pd

#I think this function here is useless   
#def identity(x):
#    return x
    
def readDataCSV(path,sep=None,header='infer',na_values=["?","na","NA","N/A"," ","NULL","NAN","nan","null"]):
    if sep==None:
        f=open(path,'r')
        t=f.read()
        comma=t.count(',')
        semicolon=t.count(';')
        whitespace=t.count(' ')
        separators=[comma,semicolon,whitespace]
        if max(separators)==comma:
            sep=','
        if max(separators)==semicolon:
            sep=';'
        if max(separators)==whitespace:
            sep="\s+"
        
    
    df=pd.read_csv(path,sep=sep,na_values=na_values,header=header)
    
    num_floats_in_columns=0
    for col in df.columns:
        try:
            float(col)  
            num_floats_in_columns=num_floats_in_columns+1
        except:
            pass
        
    num_floats_in_first_row=0
    for item in df.ix[0,:].values:
        try:
            float(item)  
            num_floats_in_first_row=num_floats_in_first_row+1
        except:
            pass
    
    if num_floats_in_columns==num_floats_in_first_row:
        df=pd.read_csv(path,sep=sep,na_values=na_values,header=None)
    
    return df
    
    

def readDataSQL():
    return True
    
    

def prepareData(dataframe,target_name=None,scaler=[],del_zero_var=True,match_columns=[]):
    df=dataframe.copy()
    
    #drop the target variable, if the dataframe is to be used as input
    if target_name!=None:
        target=df[target_name].values
        target=convertTargetToNumerical(target)
        df,target=preprocessTarget(df,target)
        df.drop([target_name],inplace=True,axis=1)
    else:
        target=[]
        
    df.columns=fixColumns(df.columns)  

    #we do not delete zero variance columns in this case, because some columns might be zero variance in the test set,
    #but might not be in the tarining set, so in the original preparation of the dataset we wouldn't have deleted them!
    df=generalPreprocess(df,add_var_name=False,del_zero_var=False)
    df=createFeatures(df)
    df=pd.get_dummies(df,prefix_sep='_categoryname_')
    
    if len(match_columns)>0:
        for col in df.columns:
            if col not in match_columns:
                df.drop(col,axis=1,inplace=True)    
    
    df.columns=fixColumns(df.columns)
    df=generalPreprocess(df,add_var_name=False,del_zero_var=del_zero_var)
       
    df=df._get_numeric_data()
    if len(scaler)==0:
        scaler=df.var()
    #FIX SCALER
    for col in df.columns:
        if col in scaler.index:
            df.ix[:,col]=df.ix[:,col]/scaler[col]
    
    df=df.astype(np.float32)
    return df, target,scaler
    
    
def sample(df,targets,fraction=0.5):
    n=df.shape[0]-1
    samples=np.random.permutation(n)
    samples=samples[0:np.round(fraction*n)]
    return df.ix[samples,:],targets[samples]