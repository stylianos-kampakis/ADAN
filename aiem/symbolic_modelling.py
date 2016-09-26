# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn import cross_validation
import numpy as np
from adan.aiem.symbolic_conversion import convertIndividualsToEqs
from sympy import sympify,simplify,together


def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints
    
    
def dominates(row, candidateRow):
    return sum([row[x] > candidateRow[x] for x in range(len(row))]) == len(row)   

def trailing_zeros(num):
    num = str(num)
    numzeroes=0
    
    for i in range(0,len(num)):
        if num[i]=='0':
            numzeroes=numzeroes+1
        if num[i]!='.' and num[i]!='0':
            break
    
    return numzeroes


def findAcceptable(sols):
    acceptable=[]
    for s in sols:
        #if the mean is higher than 1 standard deviation then it is acceptable
        if (s[1]-s[2])>=0:
            acceptable.append(s)
    if len(acceptable)==0:
        acceptable.append(sols[0])
        for s in sols:
            if s[1]>acceptable[0][1]:
                acceptable[0]=s
    return acceptable
    
def findAcceptableL1(sols):
    new_sols=[]
    scores=[k[1] for k in sols]
    lenargs=[]
    new_scores=[]
    if (max(scores))>0:
        for sol in sols:
            if sol[1]>0:
                new_sols.append(sol)
                if(type(sol[0])==type([1,2,3])):
                    lenargs.append(-1*len(sol[0][0].expand().args))
                else:
                    lenargs.append(-1*len(sol[0].expand().args))
                new_scores.append(sol[1])
    
    paretoPoints, dominatedPoints = simple_cull(list(zip(lenargs,new_scores)), dominates)
   
    new_sols=[]
    for sol in sols:
        for point in paretoPoints:
            if(type(sol[0])==type([1,2,3])):
                length=(-1*len(sol[0][0].expand().args))
            else:
                length=(-1*len(sol[0].expand().args))
            if length==point[0] and sol[1]==point[1]:
                new_sols.append(sol)
    
    #remove duplicates
    toremove=[]
    for i in range(0,len(new_sols)):
        for j in range(i,len(new_sols)):
            if i!=j:
                if(new_sols[i][1]==new_sols[j][1]):
                    if(type(new_sols[j][0])==type([1,2,3])):
                        length1=(len(new_sols[i][0][0].expand().args))
                        length2=(len(new_sols[j][0][0].expand().args))
                    else:
                        length1=(len(new_sols[i][0].expand().args))
                        length2=(len(new_sols[j][0].expand().args))
                    if length1==length2:
                        toremove.append(i)
    
    toremove=list(set(toremove))  
    
    for rem in sorted(toremove,reverse=True):
        del new_sols[rem]

    
    return new_sols



def findSymbolicExpression(df,target,result_object,scaler,task="regression",logged_target=False,acceptable_only=True):
    columns=df.columns.values
    X=result_object['best_features']
    individuals=result_object['best_individuals_object']

    
    final=[]
    eqs=convertIndividualsToEqs(individuals,columns)
    for i in range(1,len(X)):
        x=np.column_stack(X[0:i])        
        if task=='regression':
            model=linear_model.LinearRegression()
            scoring='r2'
        elif task=='classification':
            model=linear_model.LogisticRegression()  
            scoring='f1_weighted'
        
        scores=cross_validation.cross_val_score(estimator=model, X=x, y=target, cv=10,scoring=scoring)
        model.fit(x,target)  
        
        m=np.mean(scores)
        sd=np.std(scores)
        if(acceptable_only and m>sd) or (not acceptable_only):
            if(len(model.coef_.shape)==1):                              
                eq=""
                #category represents a class. In logistic regression with scikit learn                
                for j in range(0,i):
                        eq=eq+"+("+str(model.coef_[j])+"*("+str(sympify(eqs[j]))+"))"
                                    
                final.append((together(sympify(eq)),m,sd))
            else:
                dummy=[]
                for category_coefs in model.coef_:               
                    eq=""
                    #category represents a class. In logistic regression with scikit learn
                    
                    for j in range(0,i):    
                        eq=eq+"+("+str(category_coefs[j])+"*("+str(sympify(eqs[j]))+"))"
                        
                    dummy.append(together(sympify(eq)))
                final.append((dummy,m,sd))
    return final   
    
    
    
    


def findSymbolicExpressionL1(df,target,result_object,scaler,task="regression",logged_target=False,find_acceptable=True,features_type='best_features',max_length=-1):
    columns=df.columns.values
    X=result_object[features_type]
    individuals=result_object['best_individuals_equations']

    
    final=[]
    eqs=convertIndividualsToEqs(individuals,columns)
    eqs2=[]
    if max_length>0:
        for eq in eqs:
            if len(eq)<max_length:
                eqs2.append(eq)           
        eqs=eqs2
    
    x=np.column_stack(X)        
    if task=='regression':
        models=findSymbolicExpressionL1_regression_helper(x,target)
    elif task=='classification':
        models=findSymbolicExpressionL1_classification_helper(x,target)  
        
    
    
    
    for model,score,coefs in models:
        if(len(model.coef_.shape)==1):                              
            eq=""
            #category represents a class. In logistic regression with scikit learn                
            for j in range(0,len(eqs)):
                    eq=eq+"+("+str(coefs[j])+"*("+str(sympify(eqs[j]))+"))"
                                  
            final.append((together(sympify(eq)),np.around(score,3)))
        else:
            dummy=[]
            for category_coefs in model.coef_:               
                eq=""
                #category represents a class. In logistic regression with scikit learn
                
                for j in range(0,len(eqs)):    
                    eq=eq+"+("+str(category_coefs[j])+"*("+str(sympify(eqs[j]))+"))"
                    
                dummy.append(together(sympify(eq)))
            final.append((dummy,np.around(score,3)))
    
    
    if find_acceptable:
        final=findAcceptableL1(final)
    
    return final     
    
    
#def round_coefs(coefs):
#    newcoefs=[]
#    if(max(np.abs(coefs))>=1):
#        return np.round(coefs)
#    else:
#        for coef in coefs:
#            newcoefs.append(np.round(coef,trailing_zeros(coef)))
#        return np.array(newcoefs)
    
def round_coefs(coefs):
    
    if(np.max(np.abs(coefs))>=1):
        return np.round(coefs)
    else:
        dummy=np.max(np.abs(coefs))
        zeroes=trailing_zeros(dummy)
        return np.around(coefs,zeroes+1)
    
    
def findSymbolicExpressionL1_regression_helper(x,target):
    models=[linear_model.ElasticNetCV(l1_ratio=1),linear_model.ElasticNetCV(l1_ratio=0.5),linear_model.LinearRegression()]
    results=[]
    for model in models:
        model.fit(x,target)
        res=(model,model.score(x,target),model.coef_)
        results.append(res)
        
        model.coef_=round_coefs(model.coef_)
        res=(model,model.score(x,target),model.coef_)
        results.append(res)
    return results
    

def findSymbolicExpressionL1_classification_helper(x,target):
    models=[linear_model.SGDClassifier(l1_ratio=1),linear_model.SGDClassifier(l1_ratio=0.5),linear_model.LogisticRegression()]
    results=[]
    for model in models:
        model.fit(x,target)
        res=(model,model.score(x,target),model.coef_)
        results.append(res)
        
        model.coef_=round_coefs(model.coef_)
        res=(model,model.score(x,target),model.coef_)
        results.append(res)
    return results    
    

    
    