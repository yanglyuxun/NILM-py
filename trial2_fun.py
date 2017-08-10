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
#from scipy.stats import mode
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import gc #garbage collector
#from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier



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

def get_time(flist):
    first=[]
    last = []
    def strip2(s):
        return dt.datetime.utcfromtimestamp(int(s.strip().split(b' ')[0]))
    for i,di in enumerate(flist['dir']):
        with open(di, 'rb') as f:  
            first.append(strip2(f.readline()))
            f.seek(-50, 2) # point to the last 50 chars
            lines = f.readlines() 
            if len(lines)>=2: 
                last.append(strip2(lines[-1]))
            else:
                raise Exception('Error!')
    return pd.DataFrame({'first':first,'last':last},index = flist.varname)
            
def convert_data(flist,d = './data/'):
    'read all csv files from files and convert them in pickle files'
    
    if not os.path.exists(d):
        os.mkdir(d)
    for i,dir0 in enumerate(flist.dir):
        varname = flist['varname'].iloc[i]
        print(varname, 'reading...')
        df = pd.read_csv(dir0, sep=' ', header=None,
              index_col=0, dtype=np.int64)
        df.columns = ['p']
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

def describe(varns,d = './data/'):
    ans = []
    for n in varns:
        df=load(n,d)
        des = df.describe().T
        des.index = [str(n)]
        ans.append(des)
        del df,des
        gc.collect()
        print(n)
    return pd.concat(ans)

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

def adjtime(varnames):
    'adjust the time of all meters'
    print('Preparing index...')
    df0 = load(1)
    ind0 = df0.index
    ind = np.int64(np.array(ind0).astype('datetime64[s]'))
    dfnew0 = [pd.DataFrame({'keep':True},index=ind),pd.DataFrame({'keep':False},index=(ind+3)[:-1])]
    dfnew1 = pd.concat(dfnew0)
    dfnew = dfnew1.sort_index()
    missing=[]
    del df0,ind,dfnew0,dfnew1
    gc.collect()
    def deal(n,dfnew,ind0):
        print(n,'------------------------')
        print('Reading...')
        df1 = load(n)
        print('Indexing...')
        indnew = np.int64(np.array(df1.index).astype('datetime64[s]'))
        df1.index = indnew
        del indnew
        df2 = pd.concat([dfnew,df1],axis=1)
        df3 = df2.sort_index()
        del df1,df2
        print('Filling...')
        p1 = df3.p.fillna(method='bfill',limit=1)
        p2 = df3.p.fillna(method='ffill',limit=1)
        del df3['p']
        df3['p'] = p1.fillna(p2)
        del p1,p2
        df4 = df3.query('keep==True')
        del df4['keep'],df3
        df5 = df4.sort_index()
        del df4
        df5.index = ind0
        print('Saving...')
        store(df5,n)
        print(varname,': Done.')
        return {'n':n,'len':len(df5),'missing':np.sum(df5.p.isnull())}
    for varname in varnames:
        if varname==1:
            continue
        missing.append(deal(varname,dfnew,ind0))
        gc.collect()
    return pd.DataFrame(missing)

def impute(varns):
    'imputation'
    print('Reading 1...')
    df1 = load(1)
    nanind = np.where(df1.p.isnull())[0]
    nans = df1.iloc[nanind].copy()
    nans['p']=0
    for n in varns:
        if n==1:
            continue
        print('Reading',n,'...')
        df = load(n)
        print('Filling...')
        p1 = df.p.fillna(method='bfill',limit=1)
        p2 = df.p.fillna(method='ffill',limit=1)
        df['p'] = p1.fillna(p2).fillna(np.min(df.p))
        del p1,p2
        print('Saving...')
        store(df,n)
        nans+=df.iloc[nanind]
        del df
        print('Done.')
        gc.collect()
    print('Saving 1...')
    df1['p'].iloc[nanind] = nans.p
    store(df1,1)
    print('Done.')

#def P_profile(p00, closestep = 10):
#    'Get the profile of the line'
#    p=np.array(p00)
#    p1=p.copy()
#    n=len(p)
#    imax = len(p)-1
#    for i,p0 in enumerate(p):
#        if i<closestep:
#            p1[i] = np.min(p[:(i+1)])
#        elif i>imax-closestep:
#            p1[i] = np.min(p[i:])
#        else:
#            p1[i] = np.min([np.max(p[(i-closestep):i]),np.max(p[i:(i+closestep)])])
#        if i % 100000==0:
#            print(i/n)
#    return p1

def P_profile(p0, closestep = 10):
    'Get the profile of the line'
    # to let p[i] = min(max(p[(i-10):i]),max(p[i:(i+10)]))
    print('Runing P_profile...')
    p1 = p0.shift(1)
    for i in range(2,closestep+1):
        p2 = p0.shift(i)
        p1 = np.where(pd.eval('p1>p2'),p1,p2)
    del p2
    p_0 = p0.copy()
    for i in range(1,closestep):
        p_1 = p0.shift(-i)
        p_0 = np.where(pd.eval('p_0>p_1'),p_0,p_1)
    del p_1
    p1 = np.where(pd.eval('p1>p_0'),p_0,p1)
    p1[:closestep] = p_0[:closestep] # fill the nan
    return pd.Series(p1,index = p0.index)

def on_off_detect(p, bound, closestep=10):
    'detect on/off in a single meter'
    bound=(p.max())*bound + (p.min())*(1-bound)
    p2 = P_profile(P_profile(p)) #closestep = 10
    gc.collect()
    print('Detecting...')
    onoff = pd.eval('p2>bound')
    onoff1 = onoff.shift(-1)
    markbegin = pd.Series(np.where(pd.eval('onoff<onoff1'),True,False),index=p.index)
    markbegin = markbegin.shift(-2) # because the profile can shift the point
    markend = pd.Series(np.where(pd.eval('onoff>onoff1'),True,False),index=p.index)
    markbegin = markbegin.fillna(False)
    markend = markend.fillna(False)
    print('Removing close ones...')
    n=len(markend)
    for i in np.where(markend)[0]:
        if i+closestep>=n:
            break
        begins = np.where(markbegin[i:(i+closestep+1)])[0]
        if len(begins)>0:
            markend[i] = False
            markbegin[begins[0]+i]=False
    return pd.DataFrame({'begin':markbegin,'end':markend})

def on_off_union(varns):
    ans = None
    for i in varns:
        if i==1:
            continue
        if not (ans is None):
            ans = ans | load(str(i)+'onoff')
        else:
            ans = load(str(i)+'onoff')
        print(i,'done.')
    return ans

def on_off_all(varns,bound=0.01, closestep=10):
    ans=None
    for n in varns:
        if n==1:
            continue
        print('Reading',n,'...')
        df = load(n)
        print('Dealing with',n,'...')
        df = on_off_detect(df.p,bound,closestep)
        print('Saving...')
        store(df,str(n)+'onoff')
        print('Combining...')
        if not (ans is None):
            ans = ans | df
        else:
            ans = df
        del df
        gc.collect()
    print('Saving combination...')
    store(ans,'1onoff')
    print('Done.')

def plot_on_off(varn,flist,bound=0.01,save=False):
    if save:
        saven = save - len([i for i in os.listdir('./img/') if i.startswith(str(varn)+'_')])
        if saven<=0:
            return
    print('-------------------------')
    print(varn,'reading...')
    df = load(varn)
    onoff = load(str(varn)+'onoff')
    inds = np.where(onoff.begin | onoff.end)[0]
    if save:
        if len(inds)<save:
            print('Not enough samples.')
            saven = len(inds) - len([i for i in os.listdir('./img/') if i.startswith(str(varn)+'_')])
            if saven<=0:
                return 
    n=0
    yma = df.p.max()
    ymi = df.p.min()
    bound=(yma)*bound + (ymi)*(1-bound)
    print('looping...')
    while True:
        if save and len(inds)<saven:
            starttime = df.index[inds[n]] - pd.Timedelta('12h')
        else:
            starttime = df.index[np.random.choice(inds)] - pd.Timedelta('12h')
        endtime = starttime + pd.Timedelta('1d')
        pdf = df.loc[starttime:endtime,'p']
        markbegin = onoff.begin.loc[starttime:endtime]
        markend = onoff.end.loc[starttime:endtime]
        n+=1
        print('No.',n)
        ind = df.loc[starttime:endtime].index
        print('ploting...')
        plt.close()
        plt.figure(figsize=(24,10))
        plt.ylim(ymi,yma)
        plt.plot(pdf,alpha=0.8)
        plt.vlines(x=ind[markbegin],ymin=ymi,ymax=yma,color='r',alpha=0.5)
        plt.vlines(x=ind[markend],ymin=ymi,ymax=yma,color='g',alpha=0.5)
        plt.hlines(y=bound,xmin=np.min(ind),xmax=np.max(ind))
        plt.title(str(varn)+': '+flist.label.loc[varn])
        if save:
            plt.savefig('./img/'+str(varn)+'_'+starttime.strftime('%y%m%d%H%M%S')+'.png')
        else:
            plt.show()
        print('Done.')
        if save:
            saven-=1
            if saven<=0:
                break
            else:
                continue
        elif input('Input n to the next:')=='n':
            break
    return {varn:len(inds)}

def plot_on_off_all(flist,bound=0.01,save=False):
    if save:
        if not os.path.exists('./img/'):
            os.mkdir('./img/')
    lenth = {}
    for varn in flist.index:
        l1 = plot_on_off(varn,flist,bound,save)
        gc.collect()
        if l1:
            lenth.update(l1)
    return lenth

def update_lenth(flist,lenth={}):
    for varn in flist.index:
        if varn not in lenth:
            print(varn,'...')
            onoff = load(str(varn)+'onoff')
            lenth[varn] = np.sum(onoff.begin | onoff.end)
    return lenth

def trans_data_1(lag=10,pre=10):
    df = pd.concat([load(1),load('1onoff')],axis=1)
    df['p'] = np.concatenate((np.diff(df.p),[0]))
    df1 = df.loc[df.begin | df.end]
    df2 = df.loc[~(df.begin | df.end)].sample(10*len(df1))
    for i in range(1,lag+1):
        arr = df.p.shift(i)
        df1['lag'+str(i)] = arr.loc[df1.index].copy()
        df2['lag'+str(i)] = arr.loc[df2.index].copy()
    for i in range(1,pre+1):
        arr = df.p.shift(-i)
        df1['pre'+str(i)] = arr.loc[df1.index].copy()
        df2['pre'+str(i)] = arr.loc[df2.index].copy()
    del df1['begin'],df1['end'],df2['begin'],df2['end']
    df2 = df2.sort_index()
    store(df1.dropna(),'1event')
    store(df2.dropna(),'1noevent')



def MakeData(markind, trainrate=0.5,train_n=0):
    '''
   target: the p data
   mark: the T/F vactor for prediction
   markall: the T/F indicating if any meter has an event
   train_n: if=0, all train set will be kept
            if>0, it means that only such events happen in the train period
    '''
    def sample(df,n):
        try:
            return df.sample(n,replace=False)
        except:
            return df.sample(n,replace=True)    
    ev1 = load('1event') # event data
    noev1 = load('1noevent') # no event data
    mk1 = ev1.loc[markind] # marked data
    ev1.drop(markind,inplace = True)
    mk1['mk']=True
    ev1['mk']=False
    noev1['mk']=False
    # split
    mk0 = mk1.sample(frac=trainrate)
    mk1.drop(mk0.index,inplace=True)
    ev0 = ev1.sample(frac=trainrate)
    ev1.drop(ev0.index,inplace=True)
    noev0 = noev1.sample(frac=trainrate)
    noev1.drop(noev0.index,inplace=True)
    # sample
    n0 = len(mk0)
    n1 = len(mk1)
    if train_n:
        mk0 = sample(mk0,train_n)
    mk0 = sample(mk0,n0)
    ev0 = sample(ev0,n0)
    noev0 = sample(noev0,n0)
    mk1 = sample(mk1,n1)
    ev1 = sample(ev1,n1)
    noev1 = sample(noev1,n1)
    trainx = pd.concat([mk0,ev0,noev0]).sample(frac=1)
    trainy = trainx.pop('mk')
    testx = pd.concat([mk1,ev1,noev1]).sample(frac=1)
    testy = testx.pop('mk')   
    return trainx,trainy,testx,testy

def Err(truevalue0,prediction0):
    '''
    calculate the RMSE, MAPE for some prediction
    '''
    truevalue = (np.array(truevalue0)>0.5)
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
    #dtrain = xgb.DMatrix(x, label=y)
    #dtest = xgb.DMatrix(testx, label=testy)
    xgb0=XGBClassifier(**para)
#    with open('xgb.pickle','rb') as f:
#        xgb0 = pickle.load(f)
    time0 = time.time()
    #bst = xgb.train(dtrain=dtrain,**para)
    xgb0.fit(x,y)
    train_time = time.time() - time0
    confusion, test_time = Errmodel(xgb0,x,y,testx,testy,
             ntree_limit=xgb0.booster().best_iteration)
    print(confusion,'\n', train_time,'\n', test_time)
    importance = sorted(xgb0.booster().get_score().items(), key=lambda x: x[1])
    result= {'model': xgb0,
            'confusion': confusion,
            'train_time': train_time,
            'test_time': test_time,
            'importance': importance,
            'best_iter': xgb0.booster().best_iteration}
    print("best_iter", xgb0.booster().best_iteration)
    return result


def MC_all(flist,para,selection=[],trainrate=0.5,train_n=0):
    result={}
    if not selection:
        selection = flist.index
        #selection = begin.columns[begin.apply(np.sum)>30]
    for var in selection:
        if var==1:
            continue
        print('\n',var,'------------------------')
        print(var, 'Reading...')
        onoff = load(str(var)+'onoff')
        print('making data...')
        x,y,tx,ty = MakeData(onoff.index[onoff.begin],trainrate,train_n)
        result[var]={}
        result[var]['begin'] = xgb_result(x,y,tx,ty,para)
        x,y,tx,ty = MakeData(onoff.index[onoff.end],trainrate,train_n)
        result[var]['end'] = xgb_result(x,y,tx,ty,para)
    return result

def MC_all_multi(flist,para,*result,selection=[],
                 trainrate=0.5,train_n=0,rep=100):
    for i in range(rep):
        result[0].append(MC_all(flist,para,selection,trainrate,train_n))
        print('===========================',
              'Now we have '+str(len(result[0]))+' results',
              '===========================',sep='\n')

def MC_trainrate(flist,para,*result,selection=[],
                 trainrate=0.5,train_n=range(1,21),rep=10):
    for i in train_n:
        if i in result[0]:
            print('pass',i)
            continue
        result1=[]
        MC_all_multi(flist,para,result1,selection,
                 trainrate,train_n=i,rep=rep)
        result[0][i]=result1.copy()
        del result1
        print('***************************',
              'Now train_n='+str(i),
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
        table.loc[(var,side),'label'] = flist.label[flist.index==var].iloc[0]
        table.loc[(var,side),'TF'] = np.mean(TFl)
        table.loc[(var,side),'TT'] = np.mean(TTl)
        table.loc[(var,side),'error'] = np.mean(errorl)
        table.loc[(var,side),'error_std'] = np.std(errorl)
        table.loc[(var,side),'F_error'] = np.mean(F_el)
    return table

def trans_para(**par):
    return par


def make_table_s(result_s, flist):
    tables={}
    for n in result_s:
#        if n in flist.index:
        tables[n] = make_table(result_s[n],flist)
        print(n,'done.')
    return tables
def make_error_table(tables):
    first = True
    for n in tables:
        if first:
            etable = [tables[n][['label']]]
            first = False
        t = tables[n][['error']]
        t.columns = [n]
        etable.append(t)
    return(pd.concat(etable,axis=1))
def tables1_plot(tb,flist):
    t = tb.T.iloc[1:].copy()
    tc = t.columns
    tcl = tc.levels
    tcl0 = tcl[0].tolist()
    tcl00 = [flist.label[flist.index==i].iloc[0] for i in tcl0]
    tc.set_levels([tcl00,tcl[1].tolist()],inplace=True)
    t.columns = tc
    # sort by amount
    sind = pd.DataFrame(t.apply(np.sum)).sort_values(by=0,ascending=False).index
    t=t.reindex_axis(sind,axis=1)
    t.plot(xlim=[0,t.index.max()])
    plt.show()
