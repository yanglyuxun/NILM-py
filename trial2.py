#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonintrusive Load Monitoring
==========[[[house1]]]=============
@author: Lyuxun Yang
"""

from trial2_fun import *
from collections import Counter
#import matplotlib.pyplot as plt

# read data -----------------
datadir = "/media/igen/DATA/_Onedrive/Data/uk-power/house_1/"
flist = dealfnames(datadir).join(pd.read_csv(datadir+'labels.dat',sep=' ',header=None,index_col=0))
flist.columns = ['dir','label']
flist['varname'] = ['p'+str(i) for i in flist.index]

# read all
dflist = read_all_data(flist)
#########################################
#########################################
To be continued...
########################################
# remove some useless machine which has too less samples
flist = flist.loc[[i not in ['p11','p25'] for i in flist.varname],:]
del dflist['p11'],dflist['p25']

# cut the times of each data
starttime = sorted([dflist[i].index[0] for i in dflist])
endtime = sorted([dflist[i].index[-1] for i in dflist])
starttime = max(starttime)
endtime = min(endtime)

# cut data
dflist = cut_data_bytime(dflist, starttime, endtime)
Counter(np.diff(dflist['p1'].index))

# fill in index and combine all data
dfall = fillto6(dflist['p1'])
Counter(np.diff(dfall.index))

# adjust time
dfall = adjtime(dfall, dflist)

#imputation
dfall = impute(dfall)
dfall.apply(lambda x: x.isnull().sum())

# on/off detection
beginmat,endmat = on_off_all(dfall)

plot_on_off(dfall, beginmat, endmat, 'p3')
plot_on_off(dfall, beginmat, endmat, 'p7')
plot_on_off(dfall, beginmat, endmat, 'p13')
plot_on_off_p1(dfall, beginmat, endmat)

plot_on_off_all(dfall,beginmat, endmat, var=[], bound=0.05)

# MC
para = {'num_boost_round':50, 
        'params':{'max_depth':12, 'eta':0.3,
                     'booster':'gbtree',
                     'objective':'binary:logistic'}}
result = MC_all(dfall,beginmat,endmat,para,selection=['p13'])
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
MC_trainrate(dfall,beginmat,endmat,para,result_trainrate,selection=[],
                 lag=10,pre=10,trainrate=0.7,train_n=range(1,21),rep=10)

