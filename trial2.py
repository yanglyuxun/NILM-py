#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonintrusive Load Monitoring
==========[[[house1]]]=============

"""
#%% import and setup
import os
os.chdir('/home/igen/桌面/Work/3.rules_of_electricity_usage-PY')

from trial2_fun import *
from collections import Counter
#import matplotlib.pyplot as plt

#%% prepare data file list 
datadir = "./Data_origin/uk-power/house_1/"
flist = dealfnames(datadir).join(pd.read_csv(datadir+'labels.dat',sep=' ',header=None,index_col=0))
flist.columns = ['dir','label']
flist['varname'] = [str(i) for i in flist.index]
flist.drop(3,inplace=True) # because it is solar powered

## see the time situation
get_time(flist).describe()
# decide the starttime
sorted(get_time(flist)['first'])
get_time(flist)['first']

sorted(get_time(flist)['last'])
get_time(flist)['last']


#%% convert all data to pickle
convert_data(flist)
des = describe(flist.index)

#%% get the index and adjust all others
df1 = fillto6(load(1))
Counter(np.diff(df1.index))
#Counter({numpy.timedelta64(5000000000,'ns'): 230927,
#         numpy.timedelta64(6000000000,'ns'): 19828201,
#         numpy.timedelta64(7000000000,'ns'): 2943424,
#         numpy.timedelta64(8000000000,'ns'): 7})
store(df1, 1)
store(df1.index,'index')
missing = adjtime(flist.index)

# check the stats
missing['rate'] = missing.missing/missing.len
des2 = describe(flist.index)
des['newcount'] = des2['count']
des['newcount']-des['count']

#%% imputation
impute(flist.index)

#%% on/off detection
on_off_all(flist.index) # save pickle files: n+'onoff.pc'
#plot_on_off_all(flist)
lenth = plot_on_off_all(flist,save=20)
lenth = update_lenth(flist,lenth)
# lenth2 = update_lenth(flist)
flist['lenth']=pd.Series(list(lenth.values()),index=lenth.keys())
flist_bak = flist.copy()
del lenth
flist = flist.loc[flist.lenth>=70] # only use ones whose samples >70 

#%% make the data that will be used
# store: 1event 1noevent
trans_data_1()




#%% MC


# MC

para = trans_para(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0.02, learning_rate=0.01, max_delta_step=0, max_depth=7,
       min_child_weight=1, missing=None, n_estimators=704, nthread=-1,
       objective='binary:logistic', reg_alpha=0.5, reg_lambda=1,
       scale_pos_weight=1, seed=27, silent=True, subsample=0.8)
result = MC_all(flist,para,selection=[])
# multi_result=[]
MC_all_multi(dfall,beginmat,endmat,para,multi_result,selection=[],rep=100)
table_all = make_table(multi_result,flist)
table_all.iloc[:,1:].apply(np.mean,reduce=['label'])
table_descri = dfall.describe()

# reduce the train set
# multi_result_train2=[]
para = {'num_boost_round':10, 
        'params':{'max_depth':2, 'eta':0.1,
                     'booster':'gbtree',
                     'objective':'binary:logistic'}}
result = MC_all(dfall,beginmat,endmat,para,selection=['p13'],lag=10,pre=10,trainrate=0.7,train_n=2)
MC_all_multi(dfall,beginmat,endmat,para,multi_result_train2,selection=[],
             lag=10,pre=10,trainrate=0.7,train_n=2,rep=1)

# different trainrates
para = {'num_boost_round':50, 
        'params':{'max_depth':12, 'eta':0.3,
                     'booster':'gbtree',
                     'objective':'binary:logistic'}}
# result_trainrate = {}
MC_trainrate(flist,para,result_trainrate,selection=[],
                 trainrate=0.5,train_n=range(1,21),rep=10)

table_s = make_table_s(result_trainrate, flist)
table_s_1 = make_error_table(table_s)
tables1_plot(table_s_1.loc[[i for i in table_s_1.index if i not in [('p9','begin'),('p9','end')]]],flist)
#plot_on_off(dfall, beginmat, endmat, 'p9',lines=False)
