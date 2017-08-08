#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonintrusive Load Monitoring
=================[[[house5]]]=============

"""

from trial1_fun import * #调用写好的各种函数。凡是spyder标感叹号的函数都来自这里。
from collections import Counter
import matplotlib.pyplot as plt

# 读数据 -----------------
datadir = "/media/igen/DATA/_Onedrive/OneDrive/Work/3.rules_of_electricity_usage-PY/Data_origin/uk-power/house_5" #源数据所在目录
flist = dealfnames(datadir).join(pd.read_csv(datadir+'labels.dat',sep=' ',header=None,index_col=0)) #统计目录内的文件信息，形成一个列表
flist.columns = ['dir','label']
flist['varname'] = ['p'+str(i) for i in flist.index]

# 全部数据读入内存
dflist = read_all_data(flist)

# 把后面实验中发现的不合适的变量移除
flist = flist.loc[[i not in ['p11','p25'] for i in flist.varname],:]
del dflist['p11'],dflist['p25']

# 找到公共的起始时间和结束时间
starttime = sorted([dflist[i].index[0] for i in dflist])
endtime = sorted([dflist[i].index[-1] for i in dflist])
starttime = max(starttime)
endtime = min(endtime)

# 删掉时间范围以外的数据
dflist = cut_data_bytime(dflist, starttime, endtime)
Counter(np.diff(dflist['p1'].index))

# 有些时间间隔显著大于6s，在中间填充一些时间点，使得所有的时间间隔大概都是6s
dfall = fillto6(dflist['p1'])
Counter(np.diff(dfall.index))

# 源数据中不同的电表的时间戳是不统一的，不太好对应起来。这里使所有数据的时间点统一：采用就近原则，也就是待修改的时间点离哪一个已有时间点更近，就改成哪一个
dfall = adjtime(dfall, dflist)

# 填充缺失值：分表缺失值如果旁边有非缺失值，则填充旁边的值；如果没有（连续长缺失），则填充最小值（一般是0）. 总表的缺失值填充每个分表的和。
dfall = impute(dfall)
dfall.apply(lambda x: x.isnull().sum())

# 检测分表数据中该用电器的打开和关闭：先描绘波形轮廓，然后记录轮廓经过阀值的时候的时间点。
beginmat,endmat = on_off_all(dfall)
# 作图看看检测的效果怎么样
plot_on_off(dfall, beginmat, endmat, 'p3')
plot_on_off(dfall, beginmat, endmat, 'p7')
plot_on_off(dfall, beginmat, endmat, 'p13')
plot_on_off_p1(dfall, beginmat, endmat)

plot_on_off_all(dfall,beginmat, endmat, var=[], bound=0.05)

# 识别实验：
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
#MC_all_multi(dfall,beginmat,endmat,para,multi_result_train2,selection=[],
#             lag=10,pre=10,trainrate=0.7,train_n=2,rep=1)
#table_all_2 = make_table(multi_result_train2,flist)
#table_all_2.iloc[:,1:].apply(np.mean,reduce=['label'])

# 在不同的训练集次数的情况下的识别实验
para = {'num_boost_round':50, 
        'params':{'max_depth':12, 'eta':0.3,
                     'booster':'gbtree',
                     'objective':'binary:logistic'}}
# result_trainrate = {}
MC_trainrate(dfall,beginmat,endmat,para,result_trainrate,selection=[],
                 lag=10,pre=10,trainrate=0.7,train_n=range(19,21),rep=10)

table_s = make_table_s(result_trainrate, flist)
table_s_1 = make_error_table(table_s)
tables1_plot(table_s_1.loc[[i for i in table_s_1.index if i not in [('p9','begin'),('p9','end')]]],flist)
plot_on_off(dfall, beginmat, endmat, 'p9',lines=False)
