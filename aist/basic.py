import numpy as np
import skfuzzy as fuzz
from scipy import stats


quant=['highest','higher','high','average','low','lower','lowest']
qual=['excellent','good','decent','average','mediocre','poor','dismal']


class labelMapper():    
    def __init__(self,mapping={},labels=[],lower=-1,upper=1):
        if len({})>0:
            self.mapping=mapping
        else:
            self.mapping=dict()
            split=np.linspace(lower,upper,len(labels)+1)
            for i,lab in enumerate(labels):
                self.mapping[lab]=[split[i],split[i+1]]
            
            self.mapping['min']=lower
            self.mapping['max']=upper

    
    def read_map_function(self,x):    
        for label,value in self.mapping.items():
            if not label=='min' or label=='max':
                if value[0]<=x<=value[1]:
                    return label    
        return None


def threshold(x,lower=0,upper=1):
    if x>upper:
        x=upper  
    elif x<lower:
        x=lower
    return x

class sentenceRealizer(object):
    def __init__(self):
        self.mapper=dict()
        self.results={}
    
    def realize(self,term,value):
        dummy=self.mapper[term]
        dummy=dummy.replace('<WORD>',value)
        return dummy
        
    def realizeAll(self):
        dummy=""
        for key,val in self.results.items():
            dummy+=self.realize(key,val)+" "
            
        return dummy
    


class sentenceRealizerExpert(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerExpert, self).__init__()
        self.mapper['fit_quality_regression']="The quality of the fit is <WORD>."
        self.mapper['bias']="The model has <WORD> bias."
        self.mapper['kappa']="The kappa statistic is <WORD>"
        
    
    def interpretRegression(self,correlation,concordance):
        cor_conc_diff=threshold(correlation-concordance,lower=0,upper=1)  
        bias_labels=['no','low','average','above average','high','very high','extremely high']
        if cor_conc_diff<0.05:
            bias=bias_labels[0]
        else:
            bias_mapper=labelMapper(labels=bias_labels,lower=-0.1,upper=1)
            bias=bias_mapper.read_map_function(cor_conc_diff)        
        
        qual=['dismal', 'poor', 'mediocre', 'average', 'decent', 'good', 'excellent']      
        if correlation<=0 or concordance<=0:
            fitness_quality=qual[0]
        else:
            m=labelMapper(labels=qual,lower=-0.1,upper=1)
            fitness_quality=m.read_map_function(concordance)
              
        self.results={'fit_quality_regression':fitness_quality,'bias':bias}
        
        

        
class sentenceRealizerSymbolic(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerExpert, self).__init__()
        self.mapper['complexity']="The average complexity of the equations is <WORD>."
        self.mapper['atoms_performance']="Increasing the complexity of the terms seems to have <WORD> effect on performance."
        self.mapper['num_terms_performance']="Increasing the number of the terms seems to have <WORD> effect on performance."     
        self.mapper['average_atoms_size']="The equations are of <WORD> size on average."
        self.mapper['average_eq_size']="The the atoms are <WORD>."
        
    def interpretSymbolic(self,res_object):
        num_sols=len(res_object)   
        length_sols=np.array([len(x[0].expand().args) for x in res_object])
        performances=np.array([x[1] for x in res_object])
        length_atoms=np.array([len(x[0].atoms()) for x in res_object])
        
        complexity_mapper={}
        complexity_mapper['very low']=[0,8]
        complexity_mapper['low']=[8,15]
        complexity_mapper['average']=[15,30]
        complexity_mapper['high']=[30,70]
        complexity_mapper['very high']=[70,200]
        
        complexity=(length_atoms**2)/length_sols
        complexity_map=labelMapper(mapping=complexity_mapper)
        complexity_mean_read=complexity_map.read_map_function(complexity)
        
        slope_labels=['a strong negative','a negative','a small negative','little','a small positive','a positive','a strong positive']
        slope_mapper=labelMapper(labels=slope_labels,lower=-0.3,upper=0.3)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(length_atoms,performances)
        atoms_performance=read_map_function(slope,slope_mapper)
                        
        slope, intercept, r_value, p_value, std_err = stats.linregress(num_sols,performances)
        num_terms_performance=slope_mapper.read_map_function(slope)
                           
            
        average_length_sols=np.mean(length_sols)
        average_length_atoms=np.mean(atoms)
        
        equation_length=['very short','short','normal','long','very long']
        equation_length_mapper=labelMapper(labels=equation_length,lower=1,upper=10)
        average_eq_size=equation_length_mapper.read_map_function(average_length_sols)
        
        
        atoms_length=['simple','average','complicated','very complicated']
        atoms_length_mapper=labelMapper(labels=equation_length,lower=1,upper=7)
        average_atoms_length=atoms_length_mapper.read_map_function(average_length_atoms)
        
        self.results={'complexity':complexity_mean_read,'atoms_performance':atoms_performance,
        'num_terms_performance':num_terms_performance,'average_eq_size':average_eq_size,'average_atom_size':average_atoms_length}


realizer=sentenceRealizerExpert()

for i in (np.arange(0,1000)*0.1):
    cor=np.random.rand(1)[0]
    conc=cor-np.random.rand(1)[0]*np.random.rand(1)[0]*np.random.rand(1)[0]
    if conc<0:
        conc=cor
    
    print('pear_sub_cor: '+str(cor-conc))
    print('concordance: '+str(conc))
    print('pearson: '+str(cor))                              

    realizer.interpretRegression(cor,conc)    
    res=realizer.realizeAll()
    
    print(res)
