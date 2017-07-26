#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
functions for trial2.py
"""

import os
import time
import pandas as pd
import datetime as dt
import numpy as np
from scipy.stats import mode
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
#from sklearn.metrics import confusion_matrix



def dealfnames(datadir):
    'get the info of data'
    flist = os.listdir(datadir)
    flist = [i for i in flist if i.startswith('channel')]
    dir0 = [datadir + i for i in flist]
    var = [i.replace('channel_','').replace('.dat','') for i in flist]
    n = [int(i.split('_')[0]) for i in var]
    button = [i.find('button')>=0 for i in var]
    df = pd.DataFrame({'dir':dir0, 'var':var, 'n':n, 'button':button})
    df = df.loc[~df.button,:]
    df.index = df.pop('n')
    return df.sort_index().drop(['button','var'],axis=1)

#def get_time(flist):
#    first=[]
#    last = []
#    def strip2(s):
#        return dt.datetime.utcfromtimestamp(int(s.strip().split(b' ')[0]))
#    for i,di in enumerate(flist['dir']):
#        with open(di, 'rb') as f:  
#            first.append(strip2(f.readline()))
#            f.seek(-50, 2) # point to the last 50 chars
#            lines = f.readlines() 
#            if len(lines)>=2: 
#                last.append(strip2(lines[-1]))
#            else:
#                raise Exception('Error!')
#    return pd.DataFrame({'first':first,'last':last},index = flist.varname)
#            
def convert_data(flist,d = './data/'):
    'read all csv files from files and convert them in pickle files'
    
    if not os.path.exists(d):
        os.mkdir(d)
    for i,dir0 in enumerate(flist.dir):
        varname = flist['varname'].iloc[i]
        print(varname, 'reading...')
        df = pd.read_csv(dir0, sep=' ', header=None,
              index_col=0, dtype=np.int64)
        df.columns = [varname]
        df = df.sort_index()
        df.index = [dt.datetime.utcfromtimestamp(int(i2))
                                for i2 in df.index.to_native_types()]
        print(varname, 'writing...')
        with open(d+varname+'.pc','wb') as f:
            pickle.dump(df,f)
        print(varname, 'finished.\n')

def load(n,d = './data/'):
    with open(d+str(n)+'.pc','rb') as f:
        df = pickle.load(f)
    return df

def store(df,n,d = './data/'):
    with open(d+str(n)+'.pc','wb') as f:
        pickle.dump(df,f)
    

#def read_all_data(flist):
#    'read all data from files'
#    dflist = {}
#    for i,dir0 in enumerate(flist.dir):
#        varname = flist['varname'].iloc[i]
#        print(varname, 'working...')
#        dflist[varname] = pd.read_csv(dir0, sep=' ', header=None,
#              index_col=0, dtype=np.int64)
#        dflist[varname].columns = [varname]
#        dflist[varname] = dflist[varname].sort_index()
#        dflist[varname].index = [dt.datetime.utcfromtimestamp(int(i2))
#                                for i2 in dflist[varname].index.to_native_types()]
#        print(varname, 'finished.\n')
#    return dflist

#def cut_data_bytime(dflist,starttime, endtime):
#    for i in dflist:
#        dflist[i] = dflist[i].loc[[(i>=starttime) & (i<=endtime) for i in dflist[i].index],:]
#        print(i, ': finished. Shape:',dflist[i].shape)
#    return dflist

def fillto6(df):
    ints = pd.to_timedelta(np.diff(df.index)).astype('timedelta64[s]')
    nfill = (ints-3) // 6
    nall = (nfill>0).sum()
    print('all intervals needed to be changed:',nall)
    n=0
    inttimes = []
    for i in np.where(nfill>0)[0]:
        nints = nfill[i] +1
        for j in ((ints[i]/nints) * np.array(range(1,nfill[i]+1))).round().astype('timedelta64[s]'):
            inttimes.append(df.index[i] + j)
        n+=1
        if n%10000==0:
            print(n/nall)
    df = pd.concat([df,pd.DataFrame(index=inttimes)]).sort_index()
    return df

def adjtime(df1, varnames):
    'adjust the time of all meters'
    df1 = df1.sort_index()
    skip = {}
    for varname in varnames:
        if varname==1:
            continue
        print(varname,'------------------------')
        print('Reading...')
        df = load(varname)
        skip[varname] = 0
        j=0
        t1 = np.int64(np.array(df.index).astype('datetime64[s]'))
        n1 = np.array(df.)
        t0s = np.int64(np.array(df1.index).astype('datetime64[s]'))
        n1 = t0s.copy()
        ns[:] = np.nan
        jmax = len(t1) - 1
        imax = len(t0s) - 1
        print('Looping...')
        for i,t0 in enumerate(t0s):
            if j>=jmax:
                break
            while np.abs(t0-t1[j])>np.abs(t0-t1[j+1]):
                j += 1
                if j>=jmax:
                    break
            if np.abs(t1[j] - t0) <= 4:
                ns[i] = df.iloc[j,0]
            else:
                skip[varname]+=1
            if i % 1000 ==0:
                print(i/imax, 'has been finished.')
                print('match index:',i,j)
        print('Writing...')
        store(temp,varname)
        print(varname,': Done.')
    return skip

def impute(df0):
    'imputation'
    df=df0.copy()
    for varname in df:
        if varname=='p1':
            continue
        nalist = np.where(df[varname].isnull())[0]
        maxi = len(df[varname])-1
        print(varname,'step1')
        for i in nalist:
            if i==0 or i==maxi:
                continue
            if i-1 not in nalist and i+1 not in nalist:
                df[varname].iloc[i] = min([df[varname].iloc[i-1],df[varname].iloc[i+1]])
        print(varname,'step2')
        df[varname].fillna(mode(df[varname])[0][0],inplace=True)
    temp = df['p1'].copy()
    for i in np.where(temp.isnull())[0]:
        temp.iloc[i] = df.iloc[i,1:].sum()
    df['p1'] = temp
    return df

def P_profile(p00, closestep = 10):
    'Get the profile of the line'
    p=np.array(p00.copy())
    p1=p.copy()
    n=len(p)
    imax = len(p)-1
    for i,p0 in enumerate(p):
        if i<closestep:
            p1[i] = np.min(p[:(i+1)])
        elif i>imax-closestep:
            p1[i] = np.min(p[i:])
        else:
            p1[i] = np.min([np.max(p[(i-closestep):i]),np.max(p[i:(i+closestep)])])
        if i % 100000==0:
            print(i/n)
    return p1

def on_off_detect(p0, bound=0.05, closestep=50):
    'detect in a single meter'
    p=p0.copy()
    bound=(p.max())*bound + (p.min())*(1-bound)
    p2 = P_profile(P_profile(p))
    onoff = (p2>=bound).astype(np.int)
    diffr = np.diff(onoff)
    n=len(p)
    markbegin = np.zeros(n,dtype=np.bool)
    markend = np.zeros(n,dtype=np.bool)
    for i in np.where(diffr!=0)[0]:
        if diffr[i]>0:
            if i>=2:
                markbegin[i-2]=True
        else:
            markend[i]=True
    # remove close ones
#    for i in np.where(markend)[0]:
#        if i+closestep>n-1:
#            continue
#        begins = np.where(markbegin[(i+1):(i+closestep)])[0]
#        if len(begins)>0:
#            markend[i] = False
#            markbegin[begins[0]+i]=False
    return markbegin,markend

def on_off_all(dfall,bound=0.05, closestep=50):
    begin = pd.DataFrame(index=dfall.index)
    end = pd.DataFrame(index=dfall.index)
    for var in dfall:
        if var=='p1':
            continue
        print('detect:',var)
        begin[var],end[var] = on_off_detect(dfall[var],bound,closestep)
    begin['any']=begin.apply(np.any,axis=1)
    end['any'] = end.apply(np.any,axis=1)
    return begin,end

def plot_on_off(dfall,begin, end, var,bound=0.05):
    starttime = dfall.sample().index[0]
    endtime = starttime + pd.Timedelta('1d')
    ind = dfall.index[(dfall.index>=starttime) & (dfall.index<endtime)]
    yma = dfall[var].max()
    ymi = dfall[var].min()
    bound=(yma)*bound + (ymi)*(1-bound)
    pdf = dfall.loc[ind,var]
    markbegin = begin.loc[ind,var]
    markend = end.loc[ind,var]
    n=1
    while markbegin.sum()+markend.sum()==0:
        starttime = dfall.sample().index[0]
        endtime = starttime + pd.Timedelta('1d')
        ind = dfall.index[(dfall.index>=starttime) & (dfall.index<endtime)]
        pdf = dfall.loc[ind,var]
        markbegin = begin.loc[ind,var]
        markend = end.loc[ind,var]
        n+=1
        if n>100:
            return('error!')
    plt.figure()
    plt.ylim(ymi,yma)
    plt.plot(pdf,alpha=0.5)
    plt.vlines(x=ind[markbegin],ymin=ymi,ymax=yma,color='r')
    plt.vlines(x=ind[markend],ymin=ymi,ymax=yma,color='g')
    plt.hlines(y=bound,xmin=np.min(ind),xmax=np.max(ind))
    plt.title(var+': '+flist.label[flist.varname==var].iloc[0])
   # return ax

def plot_on_off_p1(dfall,begin, end,bound=0.05):
    var='p1'
    starttime = dfall.sample().index[0]
    endtime = starttime + pd.Timedelta('1d')
    ind = dfall.index[(dfall.index>=starttime) & (dfall.index<endtime)]
    yma = dfall[var].max()
    ymi = dfall[var].min()
    bound=(yma)*bound + (ymi)*(1-bound)
    pdf = dfall.loc[ind,var]
    markbegin = begin.loc[ind,'any']
    markend = end.loc[ind,'any']
    n=1
    while markbegin.sum()+markend.sum()==0:
        starttime = dfall.sample().index[0]
        endtime = starttime + pd.Timedelta('1d')
        ind = dfall.index[(dfall.index>=starttime) & (dfall.index<endtime)]
        pdf = dfall.loc[ind,var]
        markbegin = begin.loc[ind,'any']
        markend = end.loc[ind,'any']
        n+=1
        if n>100:
            return('error!')
    plt.figure()
    plt.ylim(ymi,yma)
    plt.plot(pdf,alpha=0.5)
    plt.vlines(x=ind[markbegin],ymin=ymi,ymax=yma,color='r')
    plt.vlines(x=ind[markend],ymin=ymi,ymax=yma,color='g')
    plt.hlines(y=bound,xmin=np.min(ind),xmax=np.max(ind))
    plt.title(var+': '+flist.label[flist.varname==var].iloc[0])

def plot_on_off_all(dfall,begin, end, var=[],bound=0.05):
    if len(var)==0:
        var = [i for i in begin.columns if i!='any']
    for v in var:
        plot_on_off(dfall,begin, end, v,bound)

def MakeData(target0, mark0,markall0, lag=10,pre=10,trainrate=0.7,train_n=0):
    '''
   target: the p data
   mark: the T/F vactor for prediction
   markall: the T/F indicating if any meter has an event
   train_n: if=0, all train set will be kept
            if>0, it means that only such events happen in the train period
    '''
    target = np.diff(target0).tolist().copy()
    target.append(0)
    mark = mark0.tolist().copy()
    markall =markall0.tolist().copy()
    markother = ((~np.array(mark)) & np.array(markall)).tolist()
    dt=pd.DataFrame({'label':mark, 'p':target,'other':markother})
    for i in range(1,lag+1):
        dt['lag'+str(i)] = dt.p.shift(i)
    for i in range(1,pre+1):
        dt['pre'+str(i)] = dt.p.shift(-i)
    n=np.sum(mark)
    dt = dt.dropna()
    dt = pd.concat([dt.loc[dt.label],
                    dt.loc[(~dt.label) & (~dt.other)].sample(n),
                    dt.loc[dt.other].sample(np.min([n,dt.other.sum()]))]).sample(frac=1)
    del dt['other']
    bound = round(dt.shape[0]*trainrate)
    trainx = dt.iloc[:bound,:]
    if train_n>0 & train_n<trainx.label.sum():
        onlytrain = trainx.loc[trainx.label].sample(train_n).copy()
        T_n = trainx.label.sum()
        trainx = trainx.loc[trainx.label==False]
        all_train = [trainx]
        j=0
        for i in range(T_n):
            all_train.append(onlytrain.iloc[[j]])
            j+=1
            if j>=train_n:
                j=0
        trainx = pd.concat(all_train)
        trainx = trainx.sample(frac=1)
    trainy = trainx.pop('label')
    testx = dt.iloc[bound:,:]
    testy = testx.pop('label')   
    return trainx,trainy,testx,testy

def Err(truevalue0,prediction0):
    '''
    calculate the RMSE, MAPE for some prediction
    '''
    truevalue = (np.array(truevalue0)>0)
    prediction = (np.array(prediction0)>0.5)
    TT = np.sum(truevalue & prediction)
    TF = np.sum(truevalue & (~prediction))
    FT = np.sum((~truevalue) & prediction)
    FF = np.sum((~truevalue) & (~prediction))
    con = pd.DataFrame({'F':[FF,TF],'T':[FT,TT],'error':[FT/(FF+FT),TF/(TF+TT)]},
                        index=['F','T'])
    return con
def Errmodel(model,x,y,xtest,ytest, **other):
    '''
    calculate the error indexes for a model which has a .predict() method
    '''
    yp = model.predict(x, **other)
    time0 = time.time()
    ytestp = model.predict(xtest, **other)
    test_time = time.time()-time0
    return pd.concat({"train":Err(y,yp),
            "test":Err(ytest,ytestp)}), test_time
def xgb_result(x,y,testx,testy,para):
    print("----- Working on 'xgb' method...")
    dtrain = xgb.DMatrix(x, label=y)
    dtest = xgb.DMatrix(testx, label=testy)
    time0 = time.time()
    bst = xgb.train(dtrain=dtrain,**para)
    train_time = time.time() - time0
    confusion, test_time = Errmodel(bst,dtrain,dtrain.get_label(),dtest,dtest.get_label(),
             ntree_limit=bst.best_iteration)
    print(confusion,'\n', train_time,'\n', test_time)
    importance = sorted(bst.get_score().items(), key=lambda x: x[1])
    result= {'model': bst,
            'confusion': confusion,
            'train_time': train_time,
            'test_time': test_time,
            'importance': importance,
            'best_iter': bst.best_iteration}
    print("best_iter", bst.best_iteration)
    return result


def MC_all(dfall,begin,end,para,selection=[],lag=10,pre=10,trainrate=0.7,train_n=0):
    result={}
    if len(selection)==0:
        selection = begin.columns[begin.apply(np.sum)>30]
    for var in selection:
        if var=='any':
            continue
        print('\n',var,'------------------------')
        x,y,tx,ty = MakeData(dfall[var],begin[var], begin['any'],lag,pre,trainrate,train_n)
        result[var]={}
        result[var]['begin'] = xgb_result(x,y,tx,ty,para)
        x,y,tx,ty = MakeData(dfall[var],end[var],end['any'],lag,pre,trainrate,train_n)
        result[var]['end'] = xgb_result(x,y,tx,ty,para)
    return result

def MC_all_multi(dfall,begin,end,para,*result,selection=[],
                 lag=10,pre=10,trainrate=0.7,train_n=0,rep=100):
    for i in range(rep):
        result[0].append(MC_all(dfall,begin,end,para,selection,lag,pre,trainrate,train_n))
        print('===========================',
              'Now we have '+str(len(result[0]))+' results',
              '===========================',sep='\n')

def MC_trainrate(dfall,begin,end,para,*result,selection=[],
                 lag=10,pre=10,trainrate=0.7,train_n=range(1,21),rep=10):
    for i in train_n:
        if i in result[0]:
            print('pass',i)
            continue
        result1=[]
        MC_all_multi(dfall,begin,end,para,result1,selection,
                 lag,pre,trainrate,train_n=i,rep=rep)
        result[0][i]=result1.copy()
        print('***************************',
              'Now we have '+str(i)+' results',
              '***************************',sep='\n')

def make_table(result,flist):
    varnames = result[0].keys()
    mindex = pd.MultiIndex.from_product([varnames,['begin','end']],names=['var','side'])
    table = pd.DataFrame(index=mindex,
                         columns=['label','TF','TT','error','error_std','F_error'])
    for var,side in table.index:
        TFl = [i[var][side]['confusion']['F'][('test','T')] for i in result]
        TTl = [i[var][side]['confusion']['T'][('test','T')] for i in result]
        errorl = [i[var][side]['confusion']['error'][('test','T')] for i in result]
        F_el = [i[var][side]['confusion']['error'][('test','F')] for i in result]
        table.loc[(var,side),'label'] = flist.label[flist.varname==var].iloc[0]
        table.loc[(var,side),'TF'] = np.mean(TFl)
        table.loc[(var,side),'TT'] = np.mean(TTl)
        table.loc[(var,side),'error'] = np.mean(errorl)
        table.loc[(var,side),'error_std'] = np.std(errorl)
        table.loc[(var,side),'F_error'] = np.mean(F_el)
    return table