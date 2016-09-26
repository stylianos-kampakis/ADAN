from adan.aidc.feature_selection import *
from sklearn.cross_validation import StratifiedKFold
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from adan.aipm.optimizers.xgboost_wrapper import *
from adan.aipm.optimizers.xgbOptimizer import *
from adan.aipm.optimizers.keras_nn import *
from adan.aipm.optimizers.rfOptimizer import *
from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *
from adan.aipm.optimizers.elasticNetOptimizer import *
from adan.aipm.optimizers.keras_nn import *


def get_prediction_correlations(models,train,target,probability=False,task=None,metric=None,n_folds=3):
    results=OrderedDict()
    for m in models:
        results[m]=[]
    
    skf=StratifiedKFold(y=target, n_folds=n_folds,shuffle=False, random_state=None)
                                         
    for train_index, test_index in skf:
        X_train =  train.iloc[test_index,:]   
        for m in models:
            if probability:
                preds=m.predict_proba(X_train)
            else:
                preds=m.predict(X_train)
            results[m]=np.append(results[m],preds)
            
    cors=cbfSelectionNumba(features=results.values(),target=target,task=task,metric=metric)
    
    return cors,zip(models,cors)
    
def choose_models(cor_model_results,percentile):
    """
    feed into the funciton the results of a get_prediction_correlations call
    """
    cors=cor_model_results[0]
    model_cors=cor_model_results[1]
    perc=np.percentile(cors,percentile)    
    final=[]
    for model,cor in model_cors:
        if cor>=perc:
            final.append(model)
    return final

class predictor_main(object):
    def __init__(self,train,target,test_train=[],test_target=[],task="regression"):
        """
        power:the power determines how much computational power is going to be used (from 1 to 5)

        """

        self.train=train
        self.target=target

        self.test_train=test_train
        self.test_target=test_target

        self.df=[]

        self.task=task
        self.results={}

    def _cleanup(self,n_rows,n_columns,task):
        df2=self.df.copy()
        df2.sample(n=n_rows,inplace=True)

        if task=="regression":
            df2 = chooseQuantileBest(df2, self.target, limit_n=n_rows, num_features=n_columns, processes=2,method="num")[0]
        elif task=="classification":
            df2 = chooseQuantileBest(df2, self.target, limit_n=n_rows, num_features=n_columns, processes=2,method="num")[0]
        return df2



    def train_evaluate_models(self,models=None,generations=10,population=10,probability=False,metric=None,ratio_evals=10):
        if models is None:
            models=[elasticNetOptimizer(task=self.task, train=self.train, target=self.target,metric=metric,ratio_evals=ratio_evals),
                    deepNNOptim(task=self.task, train=self.train, target=self.target,metric=metric,ratio_evals=ratio_evals),
                    xgbOptimTree(task=self.task, train=self.train, target=self.target,metric=metric,ratio_evals=ratio_evals),
                    randomForestOptimizer(task=self.task, train=self.train, target=self.target,metric=metric,ratio_evals=ratio_evals)]
                
        results=[]                
        
        for m in models:
            res = m.optimizeModelGA()
            [results.append(k) for k in res['estimators']]
            
        scores=get_prediction_correlations(results,self.train,self.target,task=self.task,metric=metric)
        
        return scores
            
    

    def optimize(self,power=5,n_max_rows=-1,n_max_columns=-1):
        #if the data has not been prepared before, then do so now
        if len(df)==0:
            self.df, self.target, self.scaler, self.categorical_vars, self.filler = prepareTrainData(df, target_name,del_zero_var=True)

        self.n_columns=df.shape[1]
        self.n_rows=df.shape[0]

        if n_max_rows>0 and n_max_columns>0:
            df2=self._cleanup(n_max_rows,n_max_columns,task)

        power_list=[[1,1000,100],[2,10000,1000],[3,50000,5000],[4,100000,10000],[5,np.inf,np.inf]]

        for el in power_list:
            if el[0]==power:
                break
            self._cleanup(el[1], el[2], task)

        n_columns=df2.shape[1]
        n_rows=df2.shape[0]

        self.run_enet_protocol()
        self.run_rf_protocol()

        if self.task == "regression":
            self.run_linear_regression_protocol()
        
        if self.task == "classification":
            self.run_logistic_regression_protocol()

        if n_rows>10000 and n_columns>20:
            self.run_deep_net_protocol()

        if n_rows>1000 and n_columns>10:
            self.run_gbm_protocol()
            
        


    def run_enet_protocol(self):
        model = elasticNetOptimizer(task=self.task, train=df_processed, target=target_processed)
        res = model.optimizeModelProtocol()
        model = res['model']
        m = res['metrics']
        self.results['enet'] = {'metrics': m, 'model': model}

    def run_deep_net_protocol(self):
        model = deepNNOptim(task=self.task, train=df_processed, target=target_processed)
        res = model.optimizeModelProtocol()
        model = res['model']
        m = res['metrics']
        self.results['dnn'] = {'metrics': m, 'model': model}

    def run_gbm_protocol(self):
        model = xgbOptimTree(task=self.task, train=df_processed, target=target_processed)
        res = model.optimizeModelGA()
        model = res['model']
        m = res['metrics']
        self.results['gbm']={'metrics':m,'model':model}

    def run_rf_protocol(self):
        model = randomForestOptimizer(task=self.task, train=df_processed, target=target_processed)
        res = model.optimizeModelProtocol()
        model = res['model']
        m = res['metrics']
        self.results['rf'] = {'metrics': m, 'model': model}

    def run_linear_regression_protocol(self):
        model = LinearRegression()
        model.fit(df_processed,target_processed)
        model = model
        
        if self.task=="regression":
            m=calcMetricsRegression(self.model,self.train,self.target,n_folds=n_folds)
        elif self.task=="classification":
            m=calcMetricsClassification(self.model,self.train,self.target,n_folds=n_folds)
            
        self.results['linear'] = {'metrics': m, 'model': model}

    def run_logistic_regression_protocol(self):
        
        model = LogisticRegression()
        model.fit(df_processed,target_processed)
        model = model
        
        if self.task=="regression":
            m=calcMetricsRegression(self.model,self.train,self.target,n_folds=n_folds)
        elif self.task=="classification":
            m=calcMetricsClassification(self.model,self.train,self.target,n_folds=n_folds)
            
        self.results['logistic'] = {'metrics': m, 'model': model}


