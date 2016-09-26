def runAiPD(df,high_cor_threshold):
    high_cors=findHighCorrelations(df,high_cor_threshold)
    association_results=associationRuleMining(df)
    aiem_results=smartAiEM(df,population=100,generations=50)