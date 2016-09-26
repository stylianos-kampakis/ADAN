# -*- coding: utf-8 -*-
import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)

from adan.aidc.feature_preprocess import *
from adan.aidc.feature_creation import *
import pandas as pd

#I think this function here is useless   
#def identity(x):
#    return x


    
def readData(path,sep=None,header='infer',na_values=["?","na","NA","N/A"," ","NULL","NAN","nan","null"]):
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
    
def createLags(dataframe,lagged_variables,lags=[1,2,3]):
    df=dataframe.copy()
    for var in lagged_variables:
        for lag in lags:
            col_name=var+'_lag'+str(lag)
            if col_name in df.columns:
                col_name=col_name+'adan_made'
            df[col_name]=df[var].shift(lag)
    
    df=df.dropna() 
    
    return df
        
    

def prepareTrainData(dataframe,target_name=None,del_zero_var=True,copy=True,fillNaN=True):
    
    filler={}    
    
    if copy:
        df=dataframe.copy()
    else:
        df=dataframe

           
    #drop the target variable, if the dataframe is to be used as input
    if target_name!=None:
        target=df[target_name].values
        target=convertTargetToNumerical(target)
        df,target=preprocessTarget(df,target)
        df.drop([target_name],inplace=True,axis=1)
    else:
        target=[]
    
    deleteIdColumn(df)
    df.columns=fixColumns(df.columns)    
    
    df.columns=["var_"+str(k) for k in df.columns]

    convertDates(df)
    categorical_vars=makeObjectToCategorical(df)
    
    if del_zero_var:
        deleteZeroVariance(df)
    
    if fillNaN:
        fillOutNaN(df)

    df=createFeatures(df)
    
    df=pd.get_dummies(df,prefix_sep='_categoryname_',columns=categorical_vars)    
    df=df._get_numeric_data()  
    
    df=df.astype(np.float32)
    
    if del_zero_var:
        deleteZeroVariance(df)
        
    if fillNaN:
        filler=fillOutNaN(df)
    
    scaler={}
    for column in df.columns:
        if column.find('_categoryname_')==-1:
            if df[column].var()>0:
                df[column]=df[column]/df[column].var()
                scaler[column]=df[column].var()
            #elif df[column].var()==0:
            #    df.drop(column,1)

    

    return df, target,scaler,categorical_vars,filler


def prepareTestData(dataframe,scaler=None,categorical_vars_match=[],match_columns=[],filler={}):
    
    df=dataframe.copy() 
          
    deleteIdColumn(df)
    df.columns=fixColumns(df.columns)
    df.columns=["var_"+str(k) for k in df.columns]
    
    convertDates(df)
    
    if len(categorical_vars_match)>0:
        for cat in categorical_vars_match:
            df[cat]=df[cat].astype("category")
            
    df=createFeatures(df)
    
    df=pd.get_dummies(df,prefix_sep='_categoryname_',columns=categorical_vars_match)    
    df=df._get_numeric_data()  
    
    df=df.astype(np.float32)

    #remove columns that do not exist in the original data, before creating features, excluding dummy columns
    if len(match_columns)>0:
        for col in df.columns:
                if col not in match_columns:
                    df.drop(col,axis=1,inplace=True)
    
        diff=set(match_columns)-set(df.columns)
        for col in diff:
            df[col]=0
                    
    
    fillOutNaN(df,filler=filler)
    
    for col,v in scaler.items():
        df[col]=df[col]/v
            
    return df
    
    
def sample(df,targets,fraction=0.5):
    n=df.shape[0]-1
    samples=np.random.permutation(n)
    samples=samples[0:int(np.round(fraction*n))]
    return df.ix[samples,:],targets[samples]