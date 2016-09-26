import os, sys
lib_path = os.path.abspath('/Users/stelios/Dropbox/ADAN')
sys.path.append(lib_path)
import random

#27 is interesting
#32 is interesting
random.seed(27)


from adan.features.utilities import *
from adan.aipm.optimizers import *
from adan.aiem.genetics.genetic_programming import *
from adan.modelling.estimation_utilities import *
from adan.aiem.symbolic_modelling import *


target_name=13
ngen=10
df=readData("/Users/stelios/Dropbox/ADAN/datatests/housing/housing.csv")

df2,target,scaler=prepareData(df,target_name)
df2=chooseQuantileBest(df2,target,limit_n=1000,quant=0.9)[0]

g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=4,population=300,features=3,task="regression",evaluator=evalPearsonCorNumba,n_processes=1)


k=findSymbolicExpressionL1(df2,target,g,scaler,task='regression',features_type='best_features_plus_cols',max_length=-1)

from IPython.display import display, Math, Latex

for x,y in k:   
    print("\n")
    print("Explained variance of "+str(y*100)+"%")
    display(Math(latex(x)))