import numpy as np
from xgboost_wrapper import *
from collections import OrderedDict
from optimizers import *
from adan.metrics.metrics_utilities import regressionMix,classificationMix
import xgboost as xgb

class xgbOptimTree(Optimizer):
    def __init__(self,train,target,task,ratio_evals=20,metric=[],n_jobs=1,randomize=False,objective="reg:linear"):
        self.ratio_evals=ratio_evals

        if task=="regression":
            self.model=XGBRegressor(nthread=n_jobs,objective=objective)
        elif task=="classification":
            self.model=XGBClassifier(nthread=n_jobs,objective=objective)
        
        n_features=train.shape[1]*1.0

        #do not let the estimators go below 100
        dummy=int(n_features**1.75)
        if dummy<100:
            dummy=100

        n_estimators=np.arange(max(np.round(train.shape[1]/4.0),1),min(5000,dummy),max(int(n_features/ratio_evals),1))
        learning_rate=np.arange(0.0001,0.5,0.5/ratio_evals)  
        max_depths=np.arange(np.round(train.shape[1]/4.0),n_features,max(int(n_features/ratio_evals),1))
        colsample_bytrees=np.arange(0.1,1.0,1.0/ratio_evals)  
        subsamples=np.arange(0.1,1.0,1.0/ratio_evals)  
    
        grid={'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depths,'colsample_bytree':colsample_bytrees,'subsample':subsamples}
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_depth':int}
        self.constraints={'n_estimators':(1,n_features),'learning_rate':(0.001,1.0),'max_depth':(1,n_features),'colsample_bytree':(0.1,1.0),'subsample':(0.1,1.0)}
        
        super(xgbOptimTree,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)
            
    def plot_importance(self):
        xgb.plot_importance(self.model._Booster)
        
#to do
class xgbOptimLinear(Optimizer):
    def __init__(self,train,target,task,ratio_evals=20,metric=[],n_jobs=1,randomize=False,objective="reg:linear"):
        self.ratio_evals=ratio_evals

        if task=="regression":
            self.model=XGBRegressor(nthread=n_jobs,objective=objective)
            self.model.booster='gblinear'
        elif task=="classification":
            self.model=XGBClassifier(nthread=n_jobs,objective=objective)
            self.model.booster='gblinear'
        
        n_features=train.shape[1]*1.0

        #do not let the estimators go below 100
        dummy=int(n_features**1.75)
        if dummy<100:
            dummy=100

        n_estimators=np.arange(max(np.round(train.shape[1]/4.0),1),min(5000,dummy),max(int(n_features/ratio_evals),1))
        learning_rate=np.arange(0.0001,0.5,0.5/ratio_evals)  
        reg_alpha=np.arange(0.1,1.0,1.0/ratio_evals)
        reg_lambda=np.arange(0.1,1.0,1.0/ratio_evals)  
    
        grid={'n_estimators':n_estimators,'learning_rate':learning_rate,'reg_alpha':reg_alpha,'reg_lambda':reg_lambda}
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_depth':int}
        self.constraints={'n_estimators':(1,n_features),'learning_rate':(0.001,1.0),'reg_alpha':(0.0,1.0),'reg_lambda':(0.0,1.0)}
        
        super(xgbOptimTree,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)
            
    def plot_importance(self):
        xgb.plot_importance(self.model._Booster)