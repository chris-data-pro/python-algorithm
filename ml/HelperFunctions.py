"""
    This is an importable collection of functions that are used by other scripts in the repository.
"""
import os
import pandas as pd
import numpy as np
import random
from sklearn import linear_model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from collections import Counter
from matplotlib import style
from matplotlib import cm as cm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif as MIfsmeth
from sklearn.feature_selection import mutual_info_regression as MIfsmethr
import xgboost as xgb
import itertools
import mutual_info as lmi
import copy
from ast import literal_eval as make_tuple
import sys
sys.path.insert(0, "/Users/mwoods/Folders/Programming/Python/HVAC/PATEC/LightGBM/python-package")
import lightgbm as lgb
from collections import deque
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.neural_network import MLPRegressor


from keras.models import Sequential
from keras.layers import Dense
from keras import backend as Kerbknd
#from sklearn.metrics import mean_squared_error
#import time
#from keras.callbacks import EarlyStopping
#from keras import optimizers



def table(S2):
    # from collections import Counter
    if isinstance(S2,pd.core.series.Series):
        S = S2
    elif isinstance(S2,list):
        S = pd.Series(S2)
    else:
        raise ValueError('table works on a list or a pandas series only.')
    return pd.DataFrame(pd.Series(Counter(S), name='Counts').sort_values(ascending=False))


def flatten(L):
    return [item for sublist in L for item in sublist]

def flat_list(l):
    return [item for sublist in l for item in sublist]

# compare two strings
def showStringDiff(a,b):
    import difflib
    print('{} => {}'.format(a,b))
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0]==' ': continue
        elif s[0]=='-':
            print(u'Delete "{}" from position {}'.format(s[-1],i))
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i))

def check_make_path(pn):
    if not os.path.exists(pn):
        os.makedirs(pn)

def getTrainTest(data,targets,testinds):
    traininds = [x for x in range(data.shape[0]) if x not in testinds]
    train_x = data.iloc[traininds].reset_index(drop=True)
    test_x = data.iloc[testinds].reset_index(drop=True)
    train_y = targets.iloc[traininds].reset_index(drop=True)
    test_y = targets.iloc[testinds].reset_index(drop=True)
    return train_x, train_y, test_x, test_y

# implement stratified cross validation splits
def sss(yy,K):
    outinds = [[] for x in range(K)]
    if isinstance(yy,pd.core.series.Series):
        y = yy.tolist()
    else:
        if not isinstance(yy,list):
            raise ValueError('Target values must be either a list or a pandas series')
        y=yy[:]
    ysi = np.argsort(y).tolist()
    bi = range(0,len(y),K)
    if len(y) not in bi:
        bi = bi +[len(y)]
    bir =[range(bi[x],bi[x+1]) for x in range(len(bi)-1)]
    for ii in range(len(bir)):
        random.shuffle(bir[ii])
    for bs in range(len(bir)):
        for bss in range(len(bir[bs])):
            outinds[bss]=outinds[bss] + [ysi[bir[bs].pop()]]
    return outinds

# here's a simple test that this works:
# Tar = np.random.choice(10,1000)
# Tdf = pd.DataFrame({'col':Tar})
# tmp = sss(Tar.tolist(),5)
# tmp2 = sss(Tdf['col'],5)
# IS = np.zeros((5,5))
# IS2 = np.zeros((5,5))
# for i in range(len(tmp)):
#     for j in range(len(tmp)):
#         IS[i][j] = len(set(tmp[i]).intersection(set(tmp[j])))
#         IS2[i][j] = len(set(tmp2[i]).intersection(set(tmp2[j])))


def linear_regression_All(x_train, y_train, x_test, y_test,use_poly_l,use_ridge_l):
    if use_ridge_l:
        lm = linear_model.Ridge(alpha=.5)
    else:
        lm = linear_model.LinearRegression()
    if use_poly_l:
        poly = PolynomialFeatures(degree=2)
        x_train_use =poly.fit_transform(x_train)
        x_test_use = poly.fit_transform(x_test)
    else:
        x_train_use = x_train
        x_test_use = x_test
    print("Starting the training")
    try:
        print('Fitting the model')
        model = lm.fit(x_train_use, y_train)
        print('Done fitting')
    except:
        print('There is an exception')
        print(model)
    predictions = model.predict(x_test_use)
    error = mean_squared_error(y_test, predictions)
    print("Finished Training")
    return error, predictions, model

def svm_regression_All(x_train, y_train, x_test, y_test,Cloc=1e3,gammaloc=0.1,epsilonloc=0.1):
    svr_rbf = SVR(kernel='rbf',C=Cloc,gamma=gammaloc,epsilon=epsilonloc)
    model = svr_rbf.fit(x_train, y_train)
    predictions = model.predict(x_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model

def decisionTree_regression_All(x_train, y_train, x_test, y_test,max_depth_loc=5):
    regr = DecisionTreeRegressor(max_depth=max_depth_loc)
    model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model

def xgBoost_regression_All(x_train, y_train, x_test, y_test,max_depth_loc=3,min_child_weight_loc=1, n_estimators_loc=100,colsample_bytree_loc=1):
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    regr = xgb.XGBRegressor(max_depth=max_depth_loc,
                            n_estimators=n_estimators_loc,
                            nthread=8,
                            min_child_weight=min_child_weight_loc,
                            colsample_bytree=colsample_bytree_loc)
    model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model

def mlp_regression_All(x_train, y_train, x_test, y_test,
                       hidden_layer_sizes_loc="5",
                       activation_loc='logistic',
                       solver_loc='adam',
                       learning_rate_loc='adaptive',
                       max_iter_loc=1000,
                       learning_rate_init_loc=0.01,
                       alpha_loc=0.01,
                       batch_size_loc='auto',
                       power_t_loc=0.5):
    hidden_layer_sizes_loc = tuple([int(xx) for xx in hidden_layer_sizes_loc.split('.')])
    regr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes_loc,
                                activation=activation_loc,
                                solver=solver_loc,
                                learning_rate=learning_rate_loc,
                                max_iter=max_iter_loc,
                                learning_rate_init=learning_rate_init_loc,
                                alpha=alpha_loc,
                                batch_size=batch_size_loc,
                                power_t=power_t_loc)
    # save xtrain,ytrain,xtest,ytest to here: '/Users/mwoods/Folders/Programming/Python/HVAC/PATEC/FromS3/FanDataMarch20/TimeLaggedAutoOnly/ModelSelection'
    # temppath = '/Users/mwoods/Folders/Programming/Python/HVAC/PATEC/FromS3/FanDataMarch20/TimeLaggedAutoOnly/ModelSelection/'
    # import pickle
    # pickle.dump(x_train, open(temppath + "xtrain", "wb"))
    # pickle.dump(y_train, open(temppath + "ytrain", "wb"))
    # pickle.dump(x_test, open(temppath + "xtest", "wb"))
    # pickle.dump(y_test, open(temppath + "ytest", "wb"))
    # print ("Should be exiting")
    # exit()
    print("------------------------------")
    print("in MLP Regressor.  Hidden Layers: ")
    print(hidden_layer_sizes_loc)
    print("hidden_layer_sizes_loc: " + str(hidden_layer_sizes_loc))
    print("activation_loc: " + str(activation_loc))
    print("solver_loc: " + str(solver_loc))
    print("learning_rate_loc: " + str(learning_rate_loc))
    print("max_iter_loc: " + str(max_iter_loc))
    print("learning_rate_init_loc: " + str(learning_rate_init_loc))
    print("alpha_loc: " + str(alpha_loc))
    print("batch_size_loc: "+ str(batch_size_loc))
    print("power_t_loc: " + str(power_t_loc))
    print("-----")
    print("training the model")
    print("Any NaN in X: " + str(np.nan in x_train))
    print("Any NaN in Y: " + str(np.nan in y_train))
    print("Any Inf in X: " + str(np.inf in x_train))
    print("Any Inf in Y: " + str(np.inf in y_train))
    model = regr.fit(x_train, y_train)
    print("printing the model")
    predictions = model.predict(x_test)
    print ("the largest prediction: " + str(max(predictions)))
    print ("the smallest prediction: " + str(min(predictions)))
    if np.nan in list(predictions):
        print("Found NaN values in the predictions")
        print(predictions)
    else:
        print("No Nans Found")
    if np.inf in list(predictions):
        print("Found Inf values in the predictions")
        print(predictions)
    else:
        print("No Infs found")
    error = mean_squared_error(y_test, predictions)
    print("------------------------------")
    return error, predictions, model

def TF_mlp_regression_All(x_train, y_train, x_test, y_test,
                       hidden_layer_sizes_loc="5",
                       activation_loc='relu',
                       solver_loc='adam',
                       batch_size_loc=16):
    # to set the number of cores to use
    Kerbknd.set_session(Kerbknd.tf.Session(config=Kerbknd.tf.ConfigProto(intra_op_parallelism_threads=8,
                                                                         inter_op_parallelism_threads=2,
                                                                         allow_soft_placement=True,
                                                                         device_count={'CPU': 8})))
    hidden_layer_sizes_loc = tuple([int(xx) for xx in hidden_layer_sizes_loc.split('.')])
    model = Sequential()
    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], activation=activation_loc))
    model.add(Dense(hidden_layer_sizes_loc[0], activation=activation_loc))
    #model.add(Dense(hidden_layer_sizes_loc[0], input_dim=x_train.shape[1], activation=activation_loc))
    if len(hidden_layer_sizes_loc)>1:
        for nnn in hidden_layer_sizes_loc[1:]:
            model.add(Dense(nnn, activation=activation_loc))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=solver_loc, metrics=['mse'])
    model.fit(x_train, y_train, epochs=20, batch_size=int(batch_size_loc)) # batch size should be a power of 2
    #scores = model.evaluate(x_test, y_test)
    Predictions = model.predict(x_test)
    Predictions = np.squeeze(Predictions)
    error = mean_squared_error(y_test, Predictions)
    return error, Predictions, model

def LGB_regression_All(x_train, y_train, x_test, y_test,num_leaves_loc = 150, max_depth_loc = 7,learning_rate_loc = 0.05,max_bin_loc = 200,num_round_loc=50):
    # https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
    train_data = lgb.Dataset(x_train, label=y_train)
    print("Running LGB with max_depth: %s"%max_depth_loc)
    param = {'num_leaves': num_leaves_loc,'max_depth': max_depth_loc,'learning_rate': learning_rate_loc,'max_bin': max_bin_loc,'objective': ['mse'], 'metric': ['mse']}
    num_round = num_round_loc
    model = lgb.train(param, train_data, num_round)
    predictions = model.predict(x_test)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model

# Plot the predicted vs actual for a single model
def res_plot(Pred,Tar,Err,Tar_Desc,Qual_Desc,imlocation,colmp=[]):
    varmin = min(min(Pred),min(Tar))
    varmax = max(max(Pred),max(Tar))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(mlines.Line2D([varmin, varmax], [varmin, varmax], color='red'))
    if colmp ==[]:
        ax.scatter(Pred, Tar)
    else:
        ax.scatter(Pred, Tar,color=colmp)
    major_ticks = np.arange(varmin, varmax, 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.ylabel('True Values')
    plt.xlabel('Predictions')
    plt.xticks(rotation='vertical')
    fig.suptitle(Tar_Desc)
    plt.title('Mean Squared Error: %9.8f' % Err)
    check_make_path(imlocation)
    plt.savefig(imlocation + "/Lag5_lr_"+ Tar_Desc + Qual_Desc + ".jpg")
    plt.close()

def coef_plot(M,cns,imloc,Tar_Desc,Qual_Desc):
    # from matplotlib import style
    coefs = list(M.coef_)
    y_pos = range(len(coefs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(y_pos,coefs,align='center',color='green',ecolor='black')
    ax.set_yticks(y_pos)
    ax.tick_params(axis='both',which='major',labelsize=5)
    ax.tick_params(axis='both',which='minor',labelsize=1)
    ax.set_yticklabels(cns)
    ax.set_xlabel('Coefficient')
    style.use('ggplot')
    fig.tight_layout()
    check_make_path(imloc)
    plt.savefig(imloc + "/Lag5_lr_"+ Tar_Desc + Qual_Desc + "_Model_coefficients.jpg", figsize=(14, 14),dpi=1000)
    plt.close()


def makeFV(numlag,rid,Data,LFloc,NLFloc):
    if rid < numlag:
        raise ValueError('Row number cannot be less than the lag')
    # this is assuming the rows are in temporal order  - ensure this earlier
    FV = Data[NLFloc].iloc[rid]
    LFm = Data[LFloc].iloc[(rid-numlag+1):(rid+1)][::-1]
    LFv = flatten(LFm.as_matrix().tolist())
    featvec = FV.tolist() + LFv
    featnames = list(FV.index) + [x+"_lag_" + str(y) for y in range(numlag) for x in LFloc]
    return featvec, featnames

def makeLagPredTar(numlag,LFloc,NLFloc,data,Tarname):
    unused, fns = makeFV(numlag,numlag,data,LFloc,NLFloc)
    DFpredictors = pd.DataFrame(columns = fns)
    for ii in range(numlag,data.shape[0]):
        DFpredictors.loc[ii], fns = makeFV(numlag,ii,data,LFloc,NLFloc)
    DFtargets = data[Tarname][numlag:]
    return DFpredictors, pd.DataFrame(DFtargets)

def makeLagTarOnly(numlag,data,Tarname):
    return pd.DataFrame(data[Tarname][numlag:])

def plot_LearingCurve(sbfh, mse_mean_h, mse_sd_h, vtitle,baselineErrloc = 0):
    xtms = mse_mean_h.index.tolist()
    tl = mse_mean_h.name
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Event History Trained')
    ax.set_ylabel('Mean Squared Error')
    ax.errorbar(xtms, mse_mean_h, yerr=mse_sd_h, fmt='o', color='blue', linestyle='dotted')
    if baselineErrloc != 0:
        ax.axhline(y=baselineErrloc,color='red')
    ax.set_title(vtitle + " " + tl + ' MSE')
    ax.legend(loc='upper right')
    plt.savefig(sbfh)
    plt.close()

def rank_features(X,Y,method='MI',continuous=True):
    if method == 'MI':
        if continuous:
            MIh = MIfsmethr(X,Y)
        else:
            MIh = MIfsmeth(X,Y)
        FI_orderh = list(np.argsort(MIh)[::-1])
    if method == 'MI2':
        MIh =[]
        for xcol in range(X.shape[1]):
            MIh.append(lmi.mutual_information((X.iloc[:, xcol].values.reshape(-1, 1), Y.values.reshape(-1, 1)),k=4))
        FI_orderh = list(np.argsort(MIh)[::-1])
    return FI_orderh

def normalizeColumn(datacp,colname,rnfact):
    if isinstance(rnfact,float) | isinstance(rnfact,int):
        datacp[colname] = datacp[colname] / float(rnfact)
    if isinstance(rnfact,list):
        if len(rnfact)==1:
            datacp[colname] = datacp[colname] / float(rnfact[0])
        if len(rnfact) == 2:
            datacp[colname] = (datacp[colname] - float(rnfact[0])) / (float(rnfact[1]) - float(rnfact[0]))
    return datacp

def FitAndTestRegression(train_x_h,train_y_h,test_x_h,test_y_h,modtype_h,MLparams_h):
    if modtype_h == 'linearRegression':
        error_2, predictions_2, model_2 = linear_regression_All(train_x_h,
                                                                train_y_h,
                                                                test_x_h,
                                                                test_y_h,
                                                                use_poly_l=MLparams_h['use_polynomial'],
                                                                use_ridge_l=MLparams_h['use_ridgeregression'])
        #print (predictions_2)
    if modtype_h == 'SVR':
        error_2, predictions_2, model_2 = svm_regression_All(train_x_h,
                                                             train_y_h,
                                                             test_x_h,
                                                             test_y_h,
                                                             Cloc=MLparams_h['C'],
                                                             gammaloc=MLparams_h['gamma'],
                                                             epsilonloc=MLparams_h['epsilon'])
    if modtype_h == 'DT':
        error_2, predictions_2, model_2 = decisionTree_regression_All(train_x_h,
                                                                      train_y_h,
                                                                      test_x_h,
                                                                      test_y_h,
                                                                      max_depth_loc=MLparams_h['max_depth'])
    if modtype_h == 'GBT':
        if 'n_estimators' not in MLparams_h:
            MLparams_h['n_estimators'] = 100
        if 'colsample_bytree' not in MLparams_h:
            MLparams_h['colsample_bytree'] = 1
        error_2, predictions_2, model_2 = xgBoost_regression_All(train_x_h,
                                                                 train_y_h,
                                                                 test_x_h,
                                                                 test_y_h,
                                                                 max_depth_loc=MLparams_h['max_depth'],
                                                                 min_child_weight_loc=MLparams_h['min_child_weight'],
                                                                 n_estimators_loc = MLparams_h['n_estimators'],
                                                                 colsample_bytree_loc = MLparams_h['colsample_bytree'])
    if modtype_h == 'MLP':
        if 'hidden_layer_sizes' not in MLparams_h:
            MLparams_h['hidden_layer_sizes'] = ["5"]
        if 'activation' not in MLparams_h:
            MLparams_h['activation'] = 'logistic'
        if 'solver' not in MLparams_h:
            MLparams_h['solver'] = 'adam'
        if 'learning_rate' not in MLparams_h:
            MLparams_h['learning_rate'] = 'adaptive'
        if 'max_iter' not in MLparams_h:
            MLparams_h['max_iter'] = 1000
        if 'learning_rate_init' not in MLparams_h:
            MLparams_h['learning_rate_init'] = 0.01
        if 'alpha' not in MLparams_h:
            MLparams_h['alpha'] = 0.01
        if 'batch_size' not in MLparams_h:
            MLparams_h['batch_size'] = 'auto'
        if 'power_t' not in MLparams_h:
            MLparams_h['power_t'] = 0.5
        error_2, predictions_2, model_2 = mlp_regression_All(train_x_h,
                                                             train_y_h,
                                                             test_x_h,
                                                             test_y_h,
                                                             hidden_layer_sizes_loc = MLparams_h['hidden_layer_sizes'],
                                                             activation_loc = MLparams_h['activation'],
                                                             solver_loc = MLparams_h['solver'],
                                                             learning_rate_loc = MLparams_h['learning_rate'],
                                                             max_iter_loc = MLparams_h['max_iter'],
                                                             learning_rate_init_loc = MLparams_h['learning_rate_init'],
                                                             alpha_loc = MLparams_h['alpha'],
                                                             batch_size_loc=MLparams_h['batch_size'],
                                                             power_t_loc=MLparams_h['power_t'])
    if modtype_h == 'TF_MLP':
        if 'hidden_layer_sizes' not in MLparams_h:
            MLparams_h['hidden_layer_sizes'] = ["5"]
        if 'activation' not in MLparams_h:
            MLparams_h['activation'] = 'relu'
        if 'solver' not in MLparams_h:
            MLparams_h['solver'] = 'adam'
        if 'batch_size' not in MLparams_h:
            MLparams_h['batch_size'] = [16]
        error_2, predictions_2, model_2 = TF_mlp_regression_All(train_x_h,
                                                                train_y_h,
                                                                test_x_h,
                                                                test_y_h,
                                                                hidden_layer_sizes_loc= MLparams_h['hidden_layer_sizes'],
                                                                activation_loc=MLparams_h['activation'],
                                                                solver_loc=MLparams_h['solver'],
                                                                batch_size_loc=MLparams_h['batch_size'])
    if modtype_h == 'LGB':
        if 'num_leaves' not in MLparams_h:
            MLparams_h['num_leaves'] = 150
        if 'max_depth' not in MLparams_h:
            MLparams_h['max_depth'] = 7
        if 'learning_rate' not in MLparams_h:
            MLparams_h['learning_rate'] = 0.05
        if 'max_bin' not in MLparams_h:
            MLparams_h['max_bin'] = 200
        if 'num_round' not in MLparams_h:
            MLparams_h['num_round'] = 50
        error_2, predictions_2, model_2 = LGB_regression_All(train_x_h,
                                                             train_y_h,
                                                             test_x_h,
                                                             test_y_h,
                                                             num_leaves_loc=MLparams_h['num_leaves'],
                                                             max_depth_loc=MLparams_h['max_depth'],
                                                             learning_rate_loc=MLparams_h['learning_rate'],
                                                             max_bin_loc=MLparams_h['max_bin'],
                                                             num_round_loc=MLparams_h['num_round'])

    return error_2, predictions_2, model_2


def balanceData(train_x_h, train_y_h, cntar_h):
    L1 = train_x_h[cntar_h + "_lag_1"].tolist()
    L2 = train_y_h.tolist()
    nochangebool = [L1[xx] == L2[xx] for xx in range(len(L2))]
    changeinds = [xx for xx in range(len(nochangebool)) if not nochangebool[xx]]
    nochangeinds = [xx for xx in range(len(nochangebool)) if nochangebool[xx]]
    nochangess = np.random.choice(nochangeinds, len(changeinds), replace=False).tolist()
    nki = nochangess + changeinds
    train_x_f = train_x_h.iloc[nki]
    train_y_f = train_y_h.iloc[nki]
    return train_x_f, train_y_f, nki

def FilterFeatures(train_x_h,train_y_h,test_x_h,FSparams_h):
    if FSparams_h['do_FS']:
        FI_order = rank_features(train_x_h,
                                 train_y_h,
                                 method=FSparams_h['FSmethod'],
                                 continuous=True)
        SelectedFeatures_loc = FI_order[:FSparams_h['NSF']]
        train_x_f = train_x_h.iloc[:,SelectedFeatures_loc]
        test_x_f = test_x_h.iloc[:,SelectedFeatures_loc]
    else:
        train_x_f = train_x_h
        test_x_f = test_x_h
        SelectedFeatures_loc = -1
    return train_x_f, test_x_f, SelectedFeatures_loc


def fsloopsfromdict(FSparamdict_h):
    FSparamsLooplist_h = []
    if False in FSparamdict_h['do_FS']:
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': False,'FSmethod': 'NoFS','NSF': 0}]
    if True in FSparamdict_h['do_FS']:
        IL =list(itertools.product([True],FSparamdict_h['FSmethod'],FSparamdict_h['NSF']))
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': xx[0],'FSmethod': xx[1],'NSF': xx[2]} for xx in IL]
    return FSparamsLooplist_h

def AssemblePreds(FullPredsDict_l):
    FullPredsDict_l['FullPredictionsAssembled']={}
    foldind = FullPredsDict_l['FullPredictionsKeyMeaning'].index('Fold')
    fpk = [make_tuple(xx) for xx in FullPredsDict_l['FullPredictions'].keys()]
    fpkl = [list(xx) for xx in fpk]
    fpklm = [xx[:foldind] + xx[foldind + 1:] for xx in fpkl]
    fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
    nwof = FullPredsDict_l['FullPredictionsKeyMeaning'][:foldind] + \
           FullPredsDict_l['FullPredictionsKeyMeaning'][foldind +1:]
    FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] = nwof
    newrepind = nwof.index('Rep')
    for ml in fpklmu:
        FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)]=np.full(len(FullPredsDict_l['Target']),np.nan)
        for fn in range(FullPredsDict_l['K']):
            predinds = FullPredsDict_l['RepFoldIndices'][ml[newrepind]][fn]
            fullkey = ml[:]
            fullkey.insert(foldind,fn)
            FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)][predinds] = FullPredsDict_l['FullPredictions'][str(tuple(fullkey))]
    return FullPredsDict_l

def CalcMSEs(FullPredsDict_l):
    FullPredsDict_l['MSE']={}
    for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
        preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
        FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
    return FullPredsDict_l


def CalcMSE_mean_std(FullPredsDict_l):
    msekeys = FullPredsDict_l['MSE'].keys()
    repind = FullPredsDict_l['FullPredictionsAssembledKeyMeaning'].index("Rep")
    fpkl = [list(xx) for xx in msekeys]
    fpklm = [xx[:repind] + xx[repind + 1:] for xx in fpkl]
    fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
    FullPredsDict_l['MSE_mean']={}
    FullPredsDict_l['MSE_std']={}
    for thiskey in fpklmu:
        tkc = thiskey[:]
        thesemses = []
        for rep in range(FullPredsDict_l["Reps"]):
            trk = tkc[:]
            trk.insert(repind,rep)
            thesemses.append(FullPredsDict_l['MSE'][tuple(trk)])
        FullPredsDict_l['MSE_mean'][tuple(thiskey)] = np.mean(thesemses)
        FullPredsDict_l['MSE_std'][tuple(thiskey)] = np.std(thesemses)
    FullPredsDict_l['MSE_mean_std_KeyMeaning'] =[xx for xx in FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] if xx != "Rep"]
    return FullPredsDict_l


def ParseMeanStdMSE(FullPredsDict_l):
    # for each combination of ML method and FS method, create 3D mean mse array and 3D std mse array
    FullPredsDict_l['ResultsByMethod_MLxFS_mean'] = {}
    FullPredsDict_l['ResultsByMethod_MLxFS_std'] = {}
    modparamnames = FullPredsDict_l['modelParameters'].keys()
    modparamlens = [len(FullPredsDict_l['modelParameters'][xx]) for xx in modparamnames]
    FSnums = FullPredsDict_l['FSParameters']['NSF']
    keymeaning = FullPredsDict_l['MSE_mean_std_KeyMeaning']
    modparaminds = [keymeaning.index(xx) for xx in modparamnames]
    FSmethodindex = keymeaning.index('FSmethod')
    FSnumindex = keymeaning.index('NumFS')
    for dfs in FullPredsDict_l['FSParameters']['do_FS']:
        if not dfs:
            mmse = np.full(modparamlens, np.nan)
            smse = np.full(modparamlens, np.nan)
            thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if xx[FSmethodindex] == 'NoFS']
            for inds, vals in np.ndenumerate(mmse):
                thiskey = list(thesekeys[0])
                for i in range(len(inds)):
                    thiskey[modparaminds[i]]=FullPredsDict_l['modelParameters'][modparamnames[i]][inds[i]]
                mmse[inds] = FullPredsDict_l['MSE_mean'][tuple(thiskey)]
                smse[inds] = FullPredsDict_l['MSE_std'][tuple(thiskey)]
            FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype'] + "_NoFS"] = mmse
            FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_NoFS"] = smse
        else:
            MPFd = modparamlens + [len(FSnums)]
            for fsm in FullPredsDict_l['FSParameters']['FSmethod']:
                mmse = np.full(MPFd, np.nan)
                smse = np.full(MPFd, np.nan)
                thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if xx[FSmethodindex] == fsm]
                for inds, vals in np.ndenumerate(mmse):
                    thiskey = list(thesekeys[0])
                    for i in range(len(inds)-1):
                        thiskey[modparaminds[i]] = FullPredsDict_l['modelParameters'][modparamnames[i]][inds[i]]
                    thiskey[FSnumindex] = FullPredsDict_l['FSParameters']['NSF'][inds[len(inds)-1]]
                    mmse[inds] = FullPredsDict_l['MSE_mean'][tuple(thiskey)]
                    smse[inds] = FullPredsDict_l['MSE_std'][tuple(thiskey)]
                FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype'] + "_" + fsm] = mmse
                FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_" + fsm] = smse
    return FullPredsDict_l

def ParseMLpipelineResults(FullPredsDict_l):
    FullPredsDict_l['MSE']={}
    for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
        preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
        FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
    # get names for all of the independent runs for which we want mean and std of MSE
    FullPredsDict_l['MSE_mean']={}
    FullPredsDict_l['MSE_std']={}
    URN = list(set(["Train"+ x.split("_Train")[1] for x in FullPredsDict_l['FullPredictionsAssembled'].keys()]))
    for urn in URN:
        thesekeys = [x for x in FullPredsDict_l['FullPredictionsAssembled'].keys() if urn in x]
        thesemses = [FullPredsDict_l['MSE'][xx] for xx in thesekeys]
        FullPredsDict_l['MSE_mean'][urn] = np.mean(thesemses)
        FullPredsDict_l['MSE_std'][urn] = np.mean(thesemses)
    # for each combination of ML method and FS method, create 3D mean mse array and 3D std mse array
    FullPredsDict_l['ResultsByMethod_MLxFS_mean']={}
    FullPredsDict_l['ResultsByMethod_MLxFS_std']={}
    modparamnames = FullPredsDict_l['modelParameters'].keys()
    modparamlens = [len(FullPredsDict_l['modelParameters'][xx]) for xx in modparamnames]
    FSnums = FullPredsDict_l['FSParameters']['NSF']
    for dfs in FullPredsDict_l['FSParameters']['do_FS']:
        if not dfs:
            mmse = np.full(modparamlens,np.nan)
            smse = np.full(modparamlens, np.nan)
            thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if 'FSmethod_NoFS' in xx]
            for inds, vals in np.ndenumerate(mmse):
                matchterms = [modparamnames[xx]+"_%s"%FullPredsDict_l['modelParameters'][modparamnames[xx]][inds[xx]] for xx in range(len(modparamnames))]
                thiskey =[xx for xx in thesekeys if all([matchterms[zz] in xx for zz in range(len(matchterms))])]
                if len(thiskey) > 1:
                    print("Problem with too many matches")
                mmse[inds] = FullPredsDict_l['MSE_mean'][thiskey[0]]
                smse[inds] = FullPredsDict_l['MSE_std'][thiskey[0]]
            FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype']+"_NoFS"] = mmse
            FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_NoFS"] = mmse
        else:
            MPFd = modparamlens + [len(FSnums)]
            for fsm in FullPredsDict_l['FSParameters']['FSmethod']:
                mmse = np.full(MPFd, np.nan)
                smse = np.full(MPFd, np.nan)
                thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if 'FSmethod_'+fsm in xx]
                for inds, vals in np.ndenumerate(mmse):
                    matchterms = [modparamnames[xx] + "_%s" % FullPredsDict_l['modelParameters'][modparamnames[xx]][inds[xx]]
                              for xx in range(len(modparamnames))] +['NumFS_%s'%FSnums[inds[len(modparamnames)]]]
                    thiskey = [xx for xx in thesekeys if all([matchterms[zz] in xx for zz in range(len(matchterms))])]
                    if len(thiskey) > 1:
                        print("Problem with too many matches")
                    mmse[inds] = FullPredsDict_l['MSE_mean'][thiskey[0]]
                    smse[inds] = FullPredsDict_l['MSE_std'][thiskey[0]]
                FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype'] + "_" + fsm] = mmse
                FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_" + fsm] = mmse
    return FullPredsDict_l

def mergeFullPreds(fpd1_in,fpd2):
    fpd1 = copy.deepcopy(fpd1_in)
    fpd1['SelectedFeatures'].update(fpd2['SelectedFeatures'])
    fpd1['FullPredictions'].update(fpd2['FullPredictions'])
    return fpd1


def PredictionFeedback(mod,dat,Target_loc,numlags):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    PredictionsList = []
    ds0 = dat.shape[0]
    bufferd = deque(dat.iloc[[0]][adjustcols].values[0])
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
        x1.loc[x1.index[0], adjustcols] = bufferd
        Pred1 = mod.predict(x1)
        bufferd.appendleft(Pred1[0])
        bufferd.pop()
        PredictionsList.append(Pred1[0])
    return PredictionsList

def nearest_round(myList, myNumber):
    return min(myList, key=lambda x: abs(x - myNumber))

def ApplyThreshold(myList, myNumber):
    return(min(myList[1],max(myList[0],myNumber)))

def PredictionFeedback2(mod,dat,Target_loc,numlags,Roundlist=[],Threshlist=[]):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    ds0 = dat.shape[0]
    PredictionsList = [None] * ds0
    bufferd = deque(dat.iloc[[0]][adjustcols].values[0])
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
        x1.loc[x1.index[0], adjustcols] = bufferd
        Pred1 = mod.predict(x1)[0]
        if len(Roundlist)>0:
            Pred1 = nearest_round(Roundlist, Pred1)
        if len(Threshlist)>0:
            Pred1 = ApplyThreshold(Threshlist, Pred1)
        bufferd.appendleft(Pred1)
        bufferd.pop()
        PredictionsList[sampnum] = Pred1
    return PredictionsList


# PredictionsPMFeedback = hf.PredictionFeedbackStacked(mod_PM , TripData_X_withGMpreds, 'GM_Preds', numlags=5)


def PredictionFeedbackStacked(mod,dat,Target_loc,numlags,seed_loc):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    PredictionsList = []
    ds0 = dat.shape[0]
    bufferd = deque([seed_loc] * len(adjustcols))
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
        x1.loc[x1.index[0], adjustcols] = bufferd
        Pred1 = mod.predict(x1)
        bufferd.appendleft(Pred1[0])
        bufferd.pop()
        PredictionsList.append(Pred1[0])
    return PredictionsList


def PredictionFeedbackStackedSimultaneous(mod_gm,mod_pm,colsforGm,colsforPm,dat,Target_loc,numlags,seed_loc=np.nan,GM_gets_truth=True):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    PredictionsListStacked = []
    PredictionsListGM = []
    PredictionsListPM = []
    ds0 = dat.shape[0]
    bufferdGM = deque(dat.iloc[[0]][adjustcols].values[0])
    if np.isnan(seed_loc):
        bufferdPM = deque(dat.iloc[[0]][adjustcols].values[0])
    else:
        bufferdPM = deque([seed_loc] * len(adjustcols))
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
        if not GM_gets_truth:
            x1.loc[x1.index[0], adjustcols] = bufferdGM
        PredGM = mod_gm.predict(x1[colsforGm])
        if PredGM[0] >1:
            PredGM[0] =1
        if PredGM[0] <0:
            PredGM[0] =0
        x1.loc[x1.index[0], adjustcols] = bufferdPM
        PredPM = mod_pm.predict(x1[colsforPm])
        stackedPred = PredGM[0]+PredPM[0]
        if stackedPred >1:
            stackedPred =1
        if stackedPred <0:
            stackedPred =0
        bufferdGM.appendleft(PredGM[0])
        bufferdGM.pop()
        bufferdPM.appendleft(stackedPred)
        bufferdPM.pop()
        PredictionsListStacked.append(stackedPred)
        PredictionsListGM.append(PredGM[0])
        PredictionsListPM.append(PredPM[0])
    return PredictionsListStacked, PredictionsListGM, PredictionsListPM

def PredictionFeedbackStackedSimultaneous_preExistingGMPreds(PredictionsListGM,mod_pm,colsforPm,dat,Target_loc,numlags,seed_loc=np.nan):
    adjustcols = [Target_loc.split("_")[0] + "_lag_" + str(xx) for xx in range(1, numlags)]
    PredictionsListStacked = []
    PredictionsListPM = []
    ds0 = dat.shape[0]
    if np.isnan(seed_loc):
        bufferdPM = deque(dat.iloc[[0]][adjustcols].values[0])
    else:
        bufferdPM = deque([seed_loc] * len(adjustcols))
    for sampnum in range(ds0):
        x1 = dat.iloc[[sampnum]].copy()
        PredGM = [PredictionsListGM[sampnum]]
        if PredGM[0] >1:
            PredGM[0] =1
        if PredGM[0] <0:
            PredGM[0] =0
        x1.loc[x1.index[0], adjustcols] = bufferdPM
        PredPM = mod_pm.predict(x1[colsforPm])
        stackedPred = PredGM[0]+PredPM[0]
        if stackedPred >1:
            stackedPred =1
        if stackedPred <0:
            stackedPred =0
        bufferdPM.appendleft(stackedPred)
        bufferdPM.pop()
        PredictionsListStacked.append(stackedPred)
        PredictionsListPM.append(PredPM[0])
    return PredictionsListStacked, PredictionsListGM, PredictionsListPM

def CheckTimestampOrder(dat):
    tsdt = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return tsdt.tolist() == sorted(tsdt.tolist())

# take a look at the two sets of predictions vs true
def FullPlot(winpreds_h,Tar_h,ptss_h):
    fig,axarr = plt.subplots(1,sharex=True,figsize=(10, 1))
    axarr.plot(winpreds_h[0:ptss_h], color='blue')
    axarr.plot(Tar_h[0:ptss_h], color='red')
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
    return plt


def FullPlot_ax(winpreds_h,Tar_h,ptss_h,axarr, ttl="",Leg=['Predicted','True Value'],ylims=[-0.05,1.05],colset=['green','black']):
    axarr.set_ylim(ylims)
    if Leg == ['Predicted', 'True Value']:
        colset = ['blue','red']
    axarr.plot(winpreds_h[0:ptss_h], color=colset[0])
    axarr.plot(Tar_h[0:ptss_h], color=colset[1])
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(Leg, loc='center left', bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt


def FeaturePlot_ax(TripData_h,ptss_h,axarr,ttl="Environment"):
    colset = cm.rainbow(np.linspace(0, 1, 8))
    axarr.plot((TripData_h['OtsAirTmpCrVal_lag_1'][0:ptss_h]/max(TripData_h['OtsAirTmpCrVal_lag_1'])).tolist(), color = colset[0])
    axarr.plot((TripData_h['IPSnsrSolrInt_lag_1'][0:ptss_h]/max(TripData_h['IPSnsrSolrInt_lag_1'])).tolist(), color = colset[1])
    axarr.plot((TripData_h['EngSpd_lag_1'][0:ptss_h]/max(TripData_h['EngSpd_lag_1'])).tolist(), color = colset[2])
    axarr.plot((TripData_h['DriverSetTemp_lag_1'][0:ptss_h]/max(TripData_h['DriverSetTemp_lag_1'])).tolist(), color = colset[3])
    axarr.plot((TripData_h['WindPattern_lag_1'][0:ptss_h]).tolist(), color = colset[4])
    axarr.plot((TripData_h['WindLevel_lag_1'][0:ptss_h]).tolist(), color = colset[5])
    axarr.plot((TripData_h['LftLoDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftLoDctTemp_lag_1'])).tolist(), color = colset[6])
    axarr.plot((TripData_h['LftUpDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftUpDctTemp_lag_1'])).tolist(), color = colset[7])
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 +0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['OtsAirTmpCrVal_lag_1', 'IPSnsrSolrInt_lag_1', 'EngSpd_lag_1', 'DriverSetTemp_lag_1', 'WindPattern_lag_1', 'WindLevel_lag_1','LftLoDctTemp_lag_1','LftUpDctTemp_lag_1'],
                 loc='center left',
                 fontsize=4,
                 bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt


def ChenHuiScoring(pred_h,tar_h,tarname_h):
    if tarname_h == "Wind Pattern":
        def sp(ot, p, pp):
            if ot >= 25:
                return pp == 'face' or pp == p
            else:
                return pp == p
    if tarname_h == "Left Temp_1":
        # need round your predictions
        def st(t, tp):
            return tp in [t - 1, t, t + 1]
    if tarname_h == "Left Temp_2":
        #need round your predictions
        def stt(t, tp):
            return tp in [t - 2, t - 1, t, t + 1, t + 2]
    if tarname_h == "Wind Level":
        # Wind Level, need round your predictions
        def sl(l, lp):
            return lp in [l - 1, l, l + 1]




