#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameter tuning

reference:
    https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
"""
#%%
#2. XGBoost Parameters
#
#The overall parameters have been divided into 3 categories by XGBoost authors:
#
#    General Parameters: Guide the overall functioning
#    Booster Parameters: Guide the individual booster (tree/regression) at each step
#    Learning Task Parameters: Guide the optimization performed
#
#I will give analogies to GBM here and highly recommend to read this article to learn from the very basics.
#General Parameters
#
#These define the overall functionality of XGBoost.
#
#    booster [default=gbtree]
#        Select the type of model to run at each iteration. It has 2 options:
#            gbtree: tree-based models
#            gblinear: linear models
#    silent [default=0]:
#        Silent mode is activated is set to 1, i.e. no running messages will be printed.
#        It’s generally good to keep it 0 as the messages might help in understanding the model.
#    nthread [default to maximum number of threads available if not set]
#        This is used for parallel processing and number of cores in the system should be entered
#        If you wish to run on all cores, value should not be entered and algorithm will detect automatically
#
#There are 2 more parameters which are set automatically by XGBoost and you need not worry about them. Lets move on to Booster parameters.
#
# 
#Booster Parameters: see the following...
#
#Though there are 2 types of boosters, I’ll consider only tree booster here because it always outperforms the linear booster and thus the later is rarely used.
#

#    max_delta_step [default=0]
#        In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
#        Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
#        This is generally not used but you can explore further if you wish.
#    scale_pos_weight [default=1]
#        A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.


#%%
import os
os.chdir('/home/igen/桌面/Work/3.rules_of_electricity_usage-PY/new_house5')

from trial2_fun import *
from collections import Counter
from numpy import linspace

#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import pickle

import matplotlib.pylab as plt
##%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#train = pd.read_csv('train_modified.csv')
#target = 'Disbursed'
#IDcol = 'ID'

def modelfit(alg, x, y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    print("\nModel Report")
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x.values, label=y.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print('Best iteration:',cvresult.shape[0])
        print(cvresult.iloc[[-1]])
    #Fit the algorithm on the data
    alg.fit(x, y,eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(x)
    dtrain_predprob = alg.predict_proba(x)[:,1]
    #Print model report:
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    if useTrainCV:
        return cvresult.shape[0]

def tune(xgb, param):
    gsearch1 = GridSearchCV(estimator = xgb, 
                            param_grid = param, 
                            scoring='roc_auc',
                            n_jobs=-1,
                            iid=False, 
                            cv=5,
                            verbose=1)
    gsearch1.fit(x,y) # x,y are global
    # gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    modelfit(gsearch1.best_estimator_, x, y)
    print(gsearch1.best_params_)
    return gsearch1,gsearch1.best_params_

#%% Data
varn=20
onoff = load(str(varn)+'onoff')
x,y,tx,ty = MakeData(onoff.index[onoff.begin],trainrate=0.5,train_n=0)
x = pd.concat([x,tx])
y = pd.concat([y,ty])
del tx,ty

#%% Fix learning rate and number of estimators for tuning tree-based parameters
#    eta [default=0.3] –> learning_rate
#        Analogous to learning rate in GBM
#        Makes the model more robust by shrinking the weights on each step
#        Typical final values to be used: 0.01-0.2
xgb1 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=-1,
     scale_pos_weight=1,
     seed=27)
best_iter = modelfit(xgb1, x, y)
xgb1.set_params(n_estimators=best_iter)

#%% Tune max_depth and min_child_weight
gsearch=[]
#    min_child_weight [default=1]
#        Defines the minimum sum of weights of all observations required in a child.
#        This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#        Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#        Too high values can lead to under-fitting hence, it should be tuned using CV.
#    max_depth [default=6]
#        The maximum depth of a tree, same as GBM.
#        Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#        Should be tuned using CV.
#        Typical values: 3-10
#    max_leaf_nodes
#        The maximum number of terminal nodes or leaves in a tree.
#        Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
#        If this is defined, GBM will ignore max_depth.
gsearch.append(tune(xgb1,{
 'max_depth':list(range(6,15)),
 'min_child_weight':list(range(0,4))
}))
# {'max_depth': 6, 'min_child_weight': 0}

#%% Tune gamma
#    gamma [default=0]
#        A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
#        Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
gsearch.append(tune(gsearch[-1][0].best_estimator_,{
 'gamma':linspace(0.05,0.15,50)
}))
# {'gamma': 0.11326530612244898}

#%% Tune subsample and colsample_bytree
#    subsample [default=1]
#        Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
#        Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
#        Typical values: 0.5-1
#    colsample_bytree [default=1]
#        Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
#        Typical values: 0.5-1
#    colsample_bylevel [default=1]
#        Denotes the subsample ratio of columns for each split, in each level.
#        I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.
gsearch.append(tune(gsearch[-1][0].best_estimator_,{
 'subsample':linspace(0.3,0.8,10),
 'colsample_bytree':linspace(0.8,1,10)
}))
# {'colsample_bytree': 1.0, 'subsample': 0.52222222222222214}

#%% Tuning Regularization Parameters
#    alpha [default=0] –> reg_alpha
#        L1 regularization term on weight (analogous to Lasso regression)
#        Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
#    lambda [default=1]–> reg_lambda
#        L2 regularization term on weights (analogous to Ridge regression)
#        This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
gsearch.append(tune(gsearch[-1][0].best_estimator_,{
 'reg_alpha':[10**i for i in range(-7,3)]
}))
# {'reg_alpha': 0.01}
gsearch.append(tune(gsearch[-1][0].best_estimator_,{
 'reg_alpha':linspace(0.005,0.015)
}))
# {'reg_alpha': 0.0060204081632653063}

#%% Reducing Learning Rate
xgb0 = gsearch[-1][0].best_estimator_
xgb0.set_params(learning_rate=0.01,n_estimators=1000)
xgb0.set_params(n_estimators=modelfit(xgb0,x,y))
xgb0
with open('xgb.pickle','wb') as f:
    pickle.dump(xgb0,f)
