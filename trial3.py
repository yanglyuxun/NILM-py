#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对house1中的1s1个的数据做处理

整体思路：把1s1个的数据拿进来，把之前识别好的用电器事件的时间拿进来，对每一个时间，截取前后10s的20个数据，作为XGBoost模型的输入样本。
    换句话说就是之前的模型的每一个样本的输入是前后6s，12s，...，的数据，现在换成前后1s,2s,...,的数据，然后同样用XGBoost跑，看识别率是不是会高一些。

"""

#%% import and setup
import os
os.chdir('/home/igen/桌面/Work/3.rules_of_electricity_usage-PY')

from trial2_fun import *
from collections import Counter
#import matplotlib.pyplot as plt

datadir = "./Data_origin/uk-power/house_1/" #数据目录

#%% 读数据
# 整个读不进来（内存不足），用chunk iterator，并且只取第一列
dfm_chunk = pd.read_csv(datadir+'mains.dat', sep=' ', header=None,
      index_col=0, iterator=True, chunksize = 10000000)
dfmall= []
i=0
for dfm in dfm_chunk:
    dfm.columns = ['p','other1','other2']
    dfmall.append(dfm.p)
    i+=1
    print(i,'/13 finished.')
del dfm,i
gc.collect()
dfm = pd.concat(dfmall)
del dfmall
gc.collect()

# 按时间排序，并四舍五入到秒
dfm.sort_index(inplace=True)
dfm.index = np.round(dfm.index)

# 看看时间间隔的分布
Counter(np.diff(dfm.index)) # 结果很多是0（即使没有四舍五入，也有很多0！），还有很多很大的数值，因此缺失值和重复值都很多
start = dfm.index[0] # 起始时间戳： 1363547563.0
end = dfm.index[-1] # 结束时间戳： 1493228158.0

# 载入之前已经存好的有事件发生的部分。这些时间点的前后20s的数据是现在想要获取的。
ev = load('1event')
evi = [dt.datetime.timestamp(i) for i in ev.index] # 把需要的时间都转成时间戳
evi[0]  # 起始时间戳：1352472382.0
evi[-1] # 结束时间戳：1493228158.0

# 然而这里起始时间戳的范围居然超出了前面的start...这样造成的缺失值就更多了。可能需要看看舍弃所有的缺失样本后，还能剩下多少，还够不够做识别的实验....

