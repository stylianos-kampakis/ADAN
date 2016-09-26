# -*- coding: utf-8 -*-
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from evaluators import *

#import pathos

import pathos
import operator

#from adan import functions
from adan.functions import *
from adan.aidc.feature_selection import *


def calcNewFeatures(result_set,df,features='best'):
    """
    returns the best features alongside the variables participating in the complex variables
    """
    all_features=[]
    complex_features=[]
    pset=setPset(df)
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset) 
    
    complex_columns=[]  
    all_columns=[]
    simple_columns=[]
    
    if features=='best':
        dummy='best_individuals_object'
    elif features=='all':
        dummy='all_features_individuals_object'
    
    for feat in result_set[dummy]:
        complex_features.append(toolbox.compile(feat))
        all_features.append(toolbox.compile(feat))
        complex_columns.append(str(feat))
        all_columns.append(str(feat))
    
    simple_features=[]
    
    for feat in result_set['variables']:
        simple_features.append(df[feat])
        simple_columns.append(str(feat))
        all_features.append(df[feat])
        all_columns.append(str(feat))
        
    
    return pd.DataFrame(np.column_stack(all_features),columns=all_columns),pd.DataFrame(np.column_stack(complex_features),columns=complex_columns),pd.DataFrame(np.column_stack(simple_features),columns=simple_columns)
        
    

def setPset(df):
    pset = gp.PrimitiveSet("MAIN", 0,prefix="coef")
    pset.addPrimitive(add,2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(div, 2)
    
    for fun in singlefunctions:
        pset.addPrimitive(fun,1)
    
    for col in df.columns.values:
        #we must use strings for column names otherwise the functions interpret the
    #column names as numbers
        pset.addTerminal(df[col].values,name=col)
        
    return pset


def findFeaturesGP(df,targets,population=300,ngen=50,cxpb=0.9,features=10,
                   max_tree=3,evaluator=evalPearsonCorNumba,task="regression",n_processes=1):
                       
    """
    This function calculates complex features that correlate with the response variable.
    Output:
    
    A dictionary with the following fields:
    
    best_features: a list of lists, where every element is a feature selected by the best n features as defined by the cbf method
    best_features_plus_cols: a list of lists, where every element is a feature selected by the best n features as defined by the cbf method plus
    any original features participating in the creation of the individuals 
    best_individuals_equations: the equations used to compute the best_features (this is the string version of best_individuals_object)
    best_individuals_object: the programs used to compute the best_features
    
    scores: the score of each individual produced during the genetic programming
    scores_cbf: the cbf score of each feature (all features not just the best ones)
    variables: the names of the original variables that participate in the creation of the features in the best_features
    all_features: a list of lists with all the features produced by the genetic algorithm
    all_features_individuals: the programs used to compute all_features
    
    """
    
    mutpb=1-cxpb    
    
    pset=setPset(df)
            
    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset) 
    
    #need to do that because multithreading does not support functions with more than one arguments
    def evaluate(x):
        return evaluator(x,toolbox=toolbox,targets=targets)
    
    #toolbox.register("evaluate", evaluator,toolbox=toolbox, targets=targets)
    toolbox.register("evaluate", evaluate)

    #toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selDoubleTournament,fitness_size=3,parsimony_size=1.4,fitness_first=True)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree)) 
    
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(population)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    if n_processes>1:
        pool = pathos.multiprocessing.ProcessingPool(n_processes)
        toolbox.register("map", pool.map)
    
#    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu,lamb, cxpb,mutpb,ngen=ngen, stats=mstats,
#                                   halloffame=hof, verbose=True)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb,mutpb, ngen=ngen, stats=mstats,
                                   halloffame=hof, verbose=True)
                         
    allfeatures=[]
    allfeatures_individuals_object=[]
    scores=[]
    
    for i in range(0,len(hof.items)):
        feature=toolbox.compile(hof.items[i])     
        
        if not np.isnan(feature).any():
            #need to guard against zero variance features
            if np.var(feature)>0.0:
                allfeatures.append(feature)
                allfeatures_individuals_object.append(hof.items[i])
                #for some reason in DEAP the key in the hall-of-fame is the score
                
    cbfscores=cbfSelectionNumba(allfeatures,targets,task=task)
    bestindices=sorted(range(len(cbfscores)), key=lambda x: cbfscores[x],reverse=True)

    bestfeatures=[]
    bestindividuals=[]
    scorescbf=[]
    best_features_plus_cols=[]
    bestindividuals_object=[]
    for i  in range(0,features):
        index=bestindices[i]
        bestfeatures.append(allfeatures[index])
        best_features_plus_cols.append(allfeatures[index])
        
        bestindividuals.append(str(hof.items[index]))
        bestindividuals_object.append(hof.items[index])
        
        scores.append(eval(str(hof.keys[index])))
        scorescbf.append(cbfscores[index])
               
       
    #all features includes the best variables, plus any single variables which might participate in the creation of the complex variables
    final_vars=[] 
    str_individuals=str(bestindividuals)
    for col in df.columns:
        if str_individuals.find(col)>-1:
            final_vars.append(col)
            #append the original variable to bestfeatures if it exists in a complex feature
            best_features_plus_cols.append(df[col])
                           
    
    return {'best_features':bestfeatures,'best_features_plus_cols':best_features_plus_cols,'best_individuals_equations':bestindividuals,'best_individuals_object':bestindividuals_object,
    'scores':scores,'scores_cbf':scorescbf,'variables':final_vars,'all_features':allfeatures,'all_features_individuals_object':allfeatures_individuals_object}
