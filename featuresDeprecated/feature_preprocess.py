# -*- coding: utf-8 -*-
import numpy as np
from dateutil.parser import parse
import pandas as pd
import warnings
import scipy as sp


def fixColumns(columns):
    """
    fixColumns fixes problems that might exist with the names of the columns in the dataframe.
    """
    cols=[]
    for col in columns:
        if type(col)!=str:
            col="var"+str(col)
        col=col.strip().replace(" ","_").replace(".","_dot_").replace("-","_").replace("@","at").replace("(","").replace(")","")
        col=col.strip().replace("+","_plus_").replace("/","_div_").replace("*","_star_").replace("\'","")        
        cols.append(col)
    return cols
    
def convertTargetToNumerical(x):
    if type(x[0])==type('a'):
        elements=np.unique(x)
        for i in range(0,len(elements)):
            x[x==elements[i]]=i
        x=x.astype('int32')
    return x

def deleteZeroVariance(df):
    toremove=[]
    for column in df.columns:
        if df[column].dtype.name=="category":      
            if len(df[column].cat.categories)==1:
                toremove.append(column)
        else:
            if not df[column].std()>0.0:
                toremove.append(column)       
    df.drop(toremove,axis=1,inplace=True)
    
def fillOutNaN(df,numerical_filler=np.nanmedian):
    for col in df.columns:
        if df[col].dtype.name=="category":
            df[col].fillna(value=df[col].mode()[0],inplace=True)
        else:
            df[col].replace([np.inf, -np.inf], np.nan,inplace=True)
            df[col].fillna(value=numerical_filler(df[col]),inplace=True)
            

def makeObjectToCategorical(df):
    columns=df.select_dtypes(include=[object]).columns
    for column in columns:
        try:
            df[column]=df[column].astype('category')
        except:
            pass
      
      
def checkIfDate(column):
    """
    Functions that uses heuristics to determine if a column is a date column.
    """
    num_dates=0
    for element in column:
        try:
            #an element must contain at least 6 characters to be a date, e.g. 010116
            if(len(element)>=6):
                parse(element)
                num_dates=num_dates+1
        except:
            pass
    if num_dates>(len(column)/2):
        return True
    else:
        return False
    
    
def convertDates(df,day=True,month=True,year=True,hour=True):
    columns=df.select_dtypes(include=[object]).columns
    
    for column in columns:
        if checkIfDate(df[column]):
            date=pd.to_datetime(df[column])
            if(day):
                df['day']=[x.day for x in date]
                df['day']=df['day'].astype('category')
            if(month):
                df['month']=[x.month for x in date]
                df['month']=df['month'].astype('category')
            if(year):
                df['year']=[x.year for x in date]
                df['year']=df['year'].astype('category')
            if(hour):
                df['hour']=[x.hour for x in date]
                df['hour']=df['hour'].astype('category')
            df.drop(column,axis=1,inplace=True)



def deleteIdColumn(df):
    for col in df.columns:
        if col.lower().strip()=='id':
            df.drop(col,axis=1,inplace=True)
            break
    return df


def generalPreprocess(newd,add_var_name=True,del_zero_var=True):
    convertDates(newd)
    makeObjectToCategorical(newd)
    fillOutNaN(newd)
    if del_zero_var:
        deleteZeroVariance(newd)
    deleteIdColumn(newd)
    if add_var_name:
        newd.columns=["var_"+str(k) for k in newd.columns]    
    return newd
    
def preprocessTarget(df,target):
    #delete rows with lots of missing values
    df2=df[np.isfinite(target)]
    target2=target[np.isfinite(target)]
    rows_original=df.shape[0]
    rows_new=df2.shape[0]
    if rows_new<=rows_original/2:
        warnings.warn("the new dataset with removed rows has fewer rows than the original")
    
    #transform the target if it is too skewed
    if np.abs(sp.stats.skew(target))>1:
        if(all(target>0)):
            target2=np.log(target+1)
        else:
            target2=np.log(target+abs(min(target))+1)
    
    return df2,target2
    

    
