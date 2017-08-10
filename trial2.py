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

#%% 数据预处理
# 生成一个数据目录：
datadir = "./Data_origin/uk-power/house_1/" #数据目录
flist = dealfnames(datadir).join(pd.read_csv(datadir+'labels.dat',sep=' ',header=None,index_col=0)) # 读取labels的数据
flist.columns = ['dir','label']
flist['varname'] = [str(i) for i in flist.index]
flist.drop(3,inplace=True) # because it is solar powered

## 看看这些数据的时间index的分布情况
get_time(flist).describe()
# 看看起始时间有哪些，并排序
sorted(get_time(flist)['first'])
get_time(flist)['first']
# 看看结束时间有那些，并排序
sorted(get_time(flist)['last'])
get_time(flist)['last']
# 这次没有像trial1那样截取一段时间了，因为那样会损失一些样本。这次直接按照总表的时间跨度来，分表无数据的部分按照缺失值来处理。

#%% 用pickle包将所有原始数据转存到data文件夹下。每个表的数据存一个文件，文件名和flist.index对应。对象是dataframe，其中index为时间，p为数值。
convert_data(flist)
des = describe(flist.index) # 这个函数对所有的变量做基本的描述统计

#%% 总表中的时间戳大致是6s一个，但是会有很多很大的时间间隔，比如20s，40s。这一步的作用是在这些长距离中插入时间点，使得处理后的时间间隔大致是6s。
# Counter(np.diff(load(1).index)) # 统计原有的时间间隔
df1 = fillto6(load(1))
Counter(np.diff(df1.index)) #统计一下处理后的时间间隔有哪些：（默认单位ns）
#Counter({numpy.timedelta64(5000000000,'ns'): 230927,
#         numpy.timedelta64(6000000000,'ns'): 19828201,
#         numpy.timedelta64(7000000000,'ns'): 2943424,
#         numpy.timedelta64(8000000000,'ns'): 7})
# 可见大致可以看做是6s一个点
store(df1, 1) #保存总表数据到data文件夹
store(df1.index,'index') #保存index备用 
del df1
# store(df,n)和load(n)是读写data文件夹下缓存文件的函数（具体看trial2_fun.py line70-77)。比如说调试的时候想查看哪个变量就用load(n)取回查看即可。

#%% 分表的时间戳和总表并不是一一对应的，这里以总表的时间戳（已经调好到约6s一个的）为标准，调整每个分表的时间戳，调完以后还是保存到缓存文件里面。
missing = adjtime(flist.index) # 这个函数调完时间戳、保存后，返回详细的每个变量在调整中产生了多少missing values的个数。

# 检查一下处理完时间以后的变量形态
missing['rate'] = missing.missing/missing.len #缺失率
des2 = describe(flist.index) # 再次对所有变量做描述统计分析，与之前的des变量对比
des['newcount'] = des2['count']
des['newcount']-des['count'] # 看看新产生了多少缺失值

#%% imputation 填充缺失值 
impute(flist.index)
# 这个函数填充缺失值的步骤：
# 首先填充分表的缺失值。如果是零星的缺失值，就用附近的值填充。如果附近还是缺失，就用最小值填充（一般是0）。
# 然后填充总表数据，如果缺失，直接用所有分表的和填充。
# 最后还是覆盖保存到data文件夹里面缓存文件

#%% 对分表数据检测，判断用电器的开关
on_off_all(flist.index) # 这个函数会对每一个分表的数据进行开关检测，检测结果以True，False的形式的dataframe保存到data文件夹下的 n+'onoff.pc' 文件名。
# 关于检测的原理，之前尝试了很多方法：方差变化、熵值变化等，但是不是太慢就是识别准确性不高。
# 后来采用的这么一种方法：
# 首先描绘波形轮廓：对于任意一个点，取左边10个点的最大值，右边10个点的最大值，然后取这两个值中小的那个。这样的好处是：就能把上下起伏的波形的上界轮廓描绘出来，同时又不丢失用电器打开、关闭位置的突变。
# 然后识别波形轮廓穿过阀值的位置：功率增加穿过阀值，识别为“打开”动作；功率减少穿过阀值，识别为“关闭”动作。阀值采用最大值与最小值之间1%的位置，即(max-min)*0.01+min
# 最后移除太近的标记点：对于关闭后60s（10个点）以内又打开的情况，认为是没有关闭（移除这样的标记点）。
# 这一步涉及的函数on_off_detect()和P_profile()的代码都不是很直观地对应上面的步骤。这是为了避免使用大循环（这个数据量的循环速度太慢了），拐着弯利用pd.eval()，np.where()等c编译好的函数。最后结果检查过和循环跑出来是一样的。

# 把上面识别出的打开关闭的情况作图：
lenth = plot_on_off_all(flist,save=5)
# 这个函数会对每一个表的识别结果，save=5表示每个表随机取5个位置（有变化的位置）画图并保存起来。其中1号是总表，里面的线不是识别出来的，而是各个分表的识别结果的加总。
# 结果在img文件夹里面。文件名含义例如4_161026093428.png表示电表4：在16年10月26日9:34:28处画的波形图。
# 通过画图的结果，可以调整前面的on/off识别算法和参数。
# 返回值是一个统计表，统计每个表识别出了多少的打开关闭事件。
lenth = update_lenth(flist,lenth) # 这个函数是针对上面返回的lenth不全的情况（比如有时检测到某表画图已足够，会跳过这个表），进一步补全所有的电表的on/off次数。如果想重新统计一遍，用下面的命令：
# lenth2 = update_lenth(flist)

# 下面是删除可用样本太少的用电器：
flist['lenth']=pd.Series(list(lenth.values()),index=lenth.keys())
flist_bak = flist.copy()
del lenth
flist = flist.loc[flist.lenth>=70] # 仅使用样本数多于70的用电器数据 

#%% 实验数据准备：
trans_data_1()
# 这个命令在data文件夹下生成2个文件： 1event 和 1noevent，都是dataframe
# 1event里面的index是所有发生过on/off事件的时间点，列p是该时间点的数据，lag1是前1步（6s）的数据，pre1是后1步（6s）的数据，以此类推。所有的数据都是差分值（后一步功率-当前功率）。
# 用差分值作为训练集的input的原因：在不同的时候其他用电器的功率可能不同，但是某用电器打开、关闭造成的功率变化是相似的。
# 1noevent 是在其他没有发生过on/off事件的时间点中抽样得到的。列名和1event一样。样本量采用1event样本量的10倍。
# 这些是为了加速后面的实验，所以预先处理好2个数据集。每次实验中只抽取其中的一部分数据。


#%% MC 蒙特卡洛模拟实验

para = trans_para(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.75,
       gamma=0.3833333333333333, learning_rate=0.1, max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=100,
       nthread=-1, objective='binary:logistic',
       reg_alpha=6.1111111111111107e-05, reg_lambda=1, scale_pos_weight=1,
       seed=27, silent=True, subsample=0.76000000000000001)
# 这是要传递给xgboost模型的参数值，通过trans_para()转化为一个dict。这些值是通过多次的CVsearch调参得到的。调参的步骤和参考资料见trial2_para_tuning.py

result_trainrate = {}
MC_trainrate(flist,para,result_trainrate,selection=[],
                 trainrate=0.5,train_n=range(100,101),rep=1)
# flist传递需要实验的哪些用电器，para传递上面设好的参数，result_trainrate存放实验结果，
# selection指定flist.index中的数，比如[3,4,5]代表只算3,4,5这三个电表，而[]代表全部算一遍
# trainrate是划分测试集和训练集大小时训练集的比例，
# train_n必须是一个iterabel的量，其中每一个数是在划分以后在训练集中允许训练集中出现的发生事件的次数，这个次数模拟的是用户需要人工参与的次数
# rep是以上所有实验重复的次数
# 这里因为时间不够，所以我只实验了train_n=[20,100]的情形
table_s = make_table_s(result_trainrate, flist)
# 这个命令是对dict中的每一个元素中的实验结果进行转换，转换为一个可以阅读的表
table_s_1 = make_error_table(table_s)
# 这个命令是对上面得到的多个表的dict合并为一个表，只留下TF的误差率
# 这个TF错误率表示当真实情况是该用电器确实发生了on/off事件时，识别结果为没有发生的比例。与之相对的FT错误率是实际没发生事件，而识别出事件发生的错误率。在所有结果中FT错误率都非常小，所以终点考察TF错误率。这也和实际需求相符合。
tables1_plot(table_s_1.loc[[i for i in table_s_1.index]],flist)
# 这个命令对上面的结果作图。横坐标是train_n，纵坐标是TF错误率，颜色和图示是识别的用电器以及是打开还是关闭。

# 下面这几行和上面几行重复
#result_trainrate20 = {}
#MC_trainrate(flist,para,result_trainrate20,selection=[],
#                 trainrate=0.5,train_n=range(20,21),rep=1)
#table_s_20 = make_table_s(result_trainrate20, flist)
#table_s_1_20 = make_error_table(table_s)
#tables1_plot(table_s_1.loc[[i for i in table_s_1.index]],flist)


