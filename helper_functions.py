# encoding: utf-8
"""
    This is an importable collection of functions that are used by other scripts in the repository.
    We are now following the convention that all function names are lowercase, underscore separated.

    Refactored on Aug 7, 2018.  Matt Woods
"""
import os
import copy
import math
import time
import random
import joblib
import datetime
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
# import lightgbm as lgb
import mutual_info as lmi
from sklearn.svm import SVR
from matplotlib import style
from collections import deque
from sunposition import sunpos
from keras.layers import Dense
from matplotlib import cm as cm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.lines as mlines
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras import backend as Kerbknd
from sklearn.utils import class_weight
from ast import literal_eval as make_tuple
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif as MIfsmeth
from sklearn.feature_selection import mutual_info_regression as MIfsmethr
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import sys
sys.path.insert(0, "/Users/mwoods/Folders/Programming/Python/HVAC/PATEC/LightGBM/python-package")


#########################
# 1
#########################
def table(s2):
    """
    Return the number of occurrences of each unique value in the input.

    :param s2: A series or a list.
    :return: A pandas DataFrame.
    """
    if isinstance(s2, pd.core.series.Series):
        s = s2
    elif isinstance(s2, list):
        s = pd.Series(s2)
    else:
        raise ValueError('table works on a list or a pandas series only.')
    return pd.DataFrame(pd.Series(Counter(s), name='Counts').sort_values(ascending=False))


#########################
# 2
#########################
def flatten(l):
    """
    Return a single list containing the elements from a list of lists.

    :param l: A list of lists.
    :return: A list.
    """
    return [item for sublist in l for item in sublist]


#########################
# 3
#########################
def show_string_diff(a, b):
    """
    Print the difference(s) between two strings.

    :param a: A string.
    :param b: A string.
    :return: None.
    """
    # compare two strings
    import difflib
    print('{} => {}'.format(a, b))
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0] == ' ':
            continue
        elif s[0] == '-':
            print(u'Delete "{}" from position {}'.format(s[-1], i))
        elif s[0] == '+':
            print(u'Add "{}" to position {}'.format(s[-1], i))


#########################
# 4
#########################
def check_make_path(pn):
    """
    Check if a path exists, create it if it does not.

    :param pn: A string.  The path Name.
    :return: None.
    """
    if not os.path.exists(pn):
        os.makedirs(pn)


#########################
# 5
#########################
def get_train_test(data, targets, testinds):
    """
    Return training and testing sets and class labels (predictands).

    :param data: A pandas DataFrame.  The full dataset with columns as features, rows as samples.
    :param targets: A pandas DataFrame. A single column DataFrame with class labels (predictands).
    :param testinds: A list.  The indices of samples to be used as the training set.
    :return:
        training set data, training set class labels (predictands), test set data, test set class labels(predictands)
    """
    traininds = [x for x in range(data.shape[0]) if x not in testinds]
    train_x = data.iloc[traininds].reset_index(drop=True)
    test_x = data.iloc[testinds].reset_index(drop=True)
    train_y = targets.iloc[traininds].reset_index(drop=True)
    test_y = targets.iloc[testinds].reset_index(drop=True)
    return train_x, train_y, test_x, test_y


#########################
# 6
#########################
def stratified_cross_validation_splits(yy, K):
    """
    Return a list of K lists (of approximately equal length) of indices of the input (yy)
    where each list has approximately the same distribution of yy values as yy itself.

    :param yy:  A pandas series or a list.  The target values (floats or integers) to be stratified and subdivided.
    :param K: An integer.  The number of partitions of the targeet.
    :return: A list of lists of indices.
    """
    # implement stratified cross validation splits
    #
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
    outinds = [[] for x in range(K)]
    if isinstance(yy, pd.core.series.Series):
        y = yy.tolist()
    else:
        if not isinstance(yy, list):
            raise ValueError('Target values must be either a list or a pandas series')
        y = yy[:]
    ysi = np.argsort(y).tolist()
    bi = range(0, len(y), K)
    if len(y) not in bi:
        bi = bi + [len(y)]
    bir = [range(bi[x], bi[x+1]) for x in range(len(bi)-1)]
    for ii in range(len(bir)):
        random.shuffle(bir[ii])
    for bs in range(len(bir)):
        for bss in range(len(bir[bs])):
            outinds[bss] = outinds[bss] + [ysi[bir[bs].pop()]]
    return outinds


#########################
# 7
#########################
def linear_regression_all(x_train, y_train, x_test, y_test, use_poly_l, use_ridge_l):
    """
    Fit a linear regression model and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train: A DataFrame. The training targets.
    :param x_test: A DataFrame. The test set.
    :param y_test: A DataFrame. The test targets.
    :param use_poly_l:  Boolean.  TRUE: use polynomial regression, FALSE: use linear regression.
    :param use_ridge_l: Boolean.  TRUE: use ridge regression with alpha =0.5.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
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
    print ("Starting the training")
    try:
        print ('Fitting the model')
        model = lm.fit(x_train_use, y_train)
        print('Done fitting')
    except:
        print ('There is an exception')
        print model
    predictions = model.predict(x_test_use)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    print("Finished Training")
    return error, predictions, model


#########################
# 8
#########################
def svm_regression_all(x_train, y_train, x_test, y_test, Cloc=1e3, gammaloc=0.1, epsilonloc=0.1):
    """
    Train a Support Vector Machine for regression and apply it to a test set.

    :param x_train:  A DataFrame. The training set.
    :param y_train:  A DataFrame. THe training targets.
    :param x_test:  A DataFrame. The test sets.
    :param y_test:  A DataFrame. The test targets.
    :param Cloc:  Int.  The SVM hyperparameter C.
    :param gammaloc:  Float.  The SVM hyperparameter Gamma.
    :param epsilonloc:  Float.  The SVM hyperparameter Epsilon.
    :return:  Mean squared error of predictions, predictions, and the fitted model.
    """
    svr_rbf = SVR(kernel='rbf', C=Cloc, gamma=gammaloc, epsilon=epsilonloc)
    model = svr_rbf.fit(x_train, y_train)
    predictions = model.predict(x_test)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model


#########################
# 9
#########################
def decision_tree_regression_all(x_train, y_train, x_test, y_test, max_depth_loc=5):
    """
    Train a decision tree regressor and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param max_depth_loc:  Int.  The decision tree hyperparameter Max Depth.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
    regr = DecisionTreeRegressor(max_depth=max_depth_loc)
    model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model


#########################
# 10
#########################
def xgBoost_regression_all(x_train, y_train, x_test, y_test, max_depth_loc=10, learning_rate_loc=0.1,  n_estimators_loc=100, min_child_weight_loc=3, colsample_bytree_loc=0.8):
    """
    Train a Gradient Boosted Tree model and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param max_depth_loc:  Int.  The hyperparameter Max Depth.
    :param min_child_weight_loc:  Int.  The hyperparameter Min Child Weight.
    :param n_estimators_loc:  Int.  The hyperparameter number of estimators.
    :param colsample_bytree_loc:  Float.  The hyperparameter column sampling.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    regr = xgb.XGBRegressor(max_depth=max_depth_loc,
                            learning_rate=learning_rate_loc,
                            n_estimators=n_estimators_loc,
                            nthread=8,
                            min_child_weight=min_child_weight_loc,
                            colsample_bytree=colsample_bytree_loc)
    model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model


#########################
# 11
#########################
def multi_layer_perceptron_regression_all(x_train, y_train, x_test, y_test,
                                          hidden_layer_sizes_loc="5",
                                          activation_loc='logistic',
                                          solver_loc='adam',
                                          learning_rate_loc='adaptive',
                                          max_iter_loc=1000,
                                          learning_rate_init_loc=0.01,
                                          alpha_loc=0.01,
                                          batch_size_loc='auto',
                                          power_t_loc=0.5):
    """
    Fit a Multi-Layer Perceptron Neural Network for regreesion and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param hidden_layer_sizes_loc:  Str.  Specification of the architecture following the format
                                        "a.b.c..."  where a,b,c,... are Ints indicating the
                                        number of nodes in the first,second,third, etc., hidden layers, respectively.
    :param activation_loc:  Str.  The activation function.
    :param solver_loc:  Str.  The solver.
    :param learning_rate_loc:  Float.  The Learning rate.
    :param max_iter_loc:  Int.  The maximum number of iterations.
    :param learning_rate_init_loc:  Float.  The initial learning rate.
    :param alpha_loc:  Float.  The regularization parameter alpha.
    :param batch_size_loc:  Int.  The number of samples in each training batch.
    :param power_t_loc:  Float.  The hyperparameter specifying the power_t exponent.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
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
    model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model


#########################
# 12
#########################
def tensor_flow_regression_all(x_train, y_train, x_test, y_test,
                               hidden_layer_sizes_loc="5",
                               activation_loc='relu',
                               solver_loc='adam',
                               batch_size_loc=16):
    """
    Use TensorFlow to fit a Multi-Layer Perceptron Neural Network for regression and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param hidden_layer_sizes_loc:  Str.  Specification of the architecture following the format
                                        "a.b.c..."  where a,b,c,... are Ints indicating the
                                        number of nodes in the first,second,third, etc., hidden layers, respectively.
    :param activation_loc:  Str.  The activation function.
    :param solver_loc:  Str.  The solver.
    :param batch_size_loc:  Int.  The number of samples in each training batch.
    :return:  Mean squared error of predictions, predictions, and the fitted model.
    """
    # to set the number of cores to use
    Kerbknd.set_session(Kerbknd.tf.Session(config=Kerbknd.tf.ConfigProto(intra_op_parallelism_threads=8,
                                                                         inter_op_parallelism_threads=2,
                                                                         allow_soft_placement=True,
                                                                         device_count={'CPU': 8})))
    hidden_layer_sizes_loc = tuple([int(xx) for xx in hidden_layer_sizes_loc.split('.')])
    model = Sequential()
    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], activation=activation_loc))
    model.add(Dense(hidden_layer_sizes_loc[0], activation=activation_loc))
    if len(hidden_layer_sizes_loc)>1:
        for nnn in hidden_layer_sizes_loc[1:]:
            model.add(Dense(nnn, activation=activation_loc))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=solver_loc, metrics=['mse'])
    model.fit(x_train, y_train, epochs=20, batch_size=int(batch_size_loc)) # batch size should be a power of 2
    Predictions = model.predict(x_test)
    Predictions = np.squeeze(Predictions)
    check_preds(Predictions)
    error = mean_squared_error(y_test, Predictions)
    return error, Predictions, model


#########################
# 13
#########################
# def light_gbm_regression_all(x_train, y_train, x_test, y_test, num_leaves_loc=150, max_depth_loc=7, learning_rate_loc=0.05, max_bin_loc=200, num_round_loc=50):
#     """
#     Train a Light Gradient Boosted Tree model and apply it to a test set.
#
#     :param x_train:  A DataFrame.  The training set.
#     :param y_train:  A DataFrame.  The training targets.
#     :param x_test:  A DataFrame.  The test set.
#     :param y_test:  A DatFrame.  The test targets.
#     :param num_leaves_loc: Int.  The hyperparameter Number of Leaves.
#     :param max_depth_loc:  Int.  The hyperparameter Max Depth.
#     :param learning_rate_loc:  Float.  The learning rate.
#     :param max_bin_loc:  Int.  The hyperparameter Max Bins.
#     :param num_round_loc:  Int.  The hyperparameter Num Round.
#     :return:  Mean squared error of predictions, predictions, and the fitted model.
#     """
#     # https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
#     train_data = lgb.Dataset(x_train, label=y_train)
#     print "Running LGB with max_depth: %s"%max_depth_loc
#     param = {'num_leaves': num_leaves_loc,'max_depth': max_depth_loc,'learning_rate': learning_rate_loc,'max_bin': max_bin_loc,'objective': ['mse'], 'metric': ['mse']}
#     num_round = num_round_loc
#     model = lgb.train(param, train_data, num_round)
#     predictions = model.predict(x_test)
#     error = mean_squared_error(y_test, predictions)
#     return error, predictions, model


#########################
# 14
#########################
def res_plot(Pred, Tar, Err, Tar_Desc, Qual_Desc, imlocation, colmp=[]):
    """
    Plot the predictec vs actual targets for a single model.

    :param Pred:  A list or a DataFrame.  The predictions.
    :param Tar:  A list or a DataFrame.  The ground truth target values.
    :param Err:  Float.  The mean squared error.
    :param Tar_Desc:  Str.  The description or name of the target.
    :param Qual_Desc:  Str.  Additional description of the model used in the name of the saved image.
    :param imlocation: Str.  The path to save the image.
    :param colmp:  List.  The color map.
    :return: None.
    """
    # Plot the predicted vs actual for a single model
    varmin = min(min(Pred),min(Tar))
    varmax = max(max(Pred),max(Tar))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_line(mlines.Line2D([varmin, varmax], [varmin, varmax], color='red'))
    if colmp == []:
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
    plt.savefig(imlocation + "/Lag5_lr_" + Tar_Desc + Qual_Desc + ".jpg")
    plt.close()


#########################
# 15
#########################
def coef_plot(M, cns, imloc, Tar_Desc, Qual_Desc):
    """
    Save a horizontal bar plot of the coefficients from a linear regression model.

    :param M:  Obj.  The fitted linear regression model.
    :param cns:  List.  The names of the predictors.
    :param imloc:  Str.  The path to save the image.
    :param Tar_Desc:  Str.  The description or name of the target.
    :param Qual_Desc:  Str.  Additional description of the model used in the name of the saved image.
    :return: None.
    """
    # from matplotlib import style
    coefs = list(M.coef_)
    y_pos = range(len(coefs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(y_pos, coefs, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=1)
    ax.set_yticklabels(cns)
    ax.set_xlabel('Coefficient')
    style.use('ggplot')
    fig.tight_layout()
    check_make_path(imloc)
    plt.savefig(imloc + "/Lag5_lr_" + Tar_Desc + Qual_Desc + "_Model_coefficients.jpg", figsize=(14, 14), dpi=1000)
    plt.close()


#########################
# 16
#########################
def make_lagged_feature_vector(numlag, rid, Data, LFloc, NLFloc):
    """
    Create a single feature vector with lagged variables taken from prior rows in the dataset.

    :param numlag:  Int.  The number of lags to produce.
    :param rid:  Int.  The id of the row to be used to produce the feature vector.
    :param Data:  DataFrame.  The full dataset.
    :param LFloc: List.  The names of the features to be lagged.
    :param NLFloc: List.  The names of the features that are not lagged.
    :return:  The lagged feature vector, and the names of the feature vector values.
    """
    if rid < numlag:
        raise ValueError('Row number cannot be less than the lag')
    # this is assuming the rows are in temporal order  - ensure this earlier
    FV = Data[NLFloc].iloc[rid]
    LFm = Data[LFloc].iloc[(rid - numlag + 1):(rid + 1)][::-1]
    LFv = flatten(LFm.as_matrix().tolist())
    featvec = FV.tolist() + LFv
    featnames = list(FV.index) + [x + "_lag_" + str(y) for y in range(numlag) for x in LFloc]
    return featvec, featnames


#########################
# 17
#########################
def make_lagged_predictors_and_targets(numlag, LFloc, NLFloc, data, Tarname):
    """
    Make a version of a dataset that contains lagged predictors and lagged targets.

    :param numlag:  Int.  The number of lags.
    :param LFloc:  List.  The training features to be lagged.
    :param NLFloc: List.  The training features that should not be lagged.
    :param data:  DataFrame.  The full dataset.
    :param Tarname:  Str.  THe name of the target variable.
    :return:  The lagged Predictors dataset as a DataFrame, and the lagged targets as a DataFrame.
    """
    unused, fns = make_lagged_feature_vector(numlag,numlag,data,LFloc,NLFloc)
    DFpredictors = pd.DataFrame(columns = fns)
    for ii in range(numlag,data.shape[0]):
        DFpredictors.loc[ii], fns = make_lagged_feature_vector(numlag,ii,data,LFloc,NLFloc)
    DFtargets = data[Tarname][numlag:]
    return DFpredictors, pd.DataFrame(DFtargets)


#########################
# 18
#########################
def make_lagged_targets_only(numlag, data, Tarname):
    """
    Make a lagged version of the targets.

    :param numlag:  Int.  The number of lags.
    :param data:  A DataFrame.  The full dataset.
    :param Tarname:  Str.  The name of the target variable.
    :return:  The lagged targets as a DataFrame.
    """
    return pd.DataFrame(data[Tarname][numlag:])


#########################
# 19
#########################
def plot_learing_curve(sbfh, mse_mean_h, mse_sd_h, vtitle, baselineErrloc=0):
    """
    Plot model performance as a function of training set size and save image.

    :param sbfh:  Str.  The path and name of the image to save.
    :param mse_mean_h:  DataFrame.  The mean mse for each value of the training set size.
    :param mse_sd_h:  DataFrame.  The std of the mse for each value of the training set size.
    :param vtitle:  Str.  Title of the figure.
    :param baselineErrloc:  Float.  The baseline mse, displayed as a horizontal red line.
    :return:  None
    """
    xtms = mse_mean_h.index.tolist()
    tl = mse_mean_h.name
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Event History Trained')
    ax.set_ylabel('Mean Squared Error')
    ax.errorbar(xtms, mse_mean_h, yerr=mse_sd_h, fmt='o', color='blue', linestyle='dotted')
    if baselineErrloc != 0:
        ax.axhline(y=baselineErrloc, color='red')
    ax.set_title(vtitle + " " + tl + ' MSE')
    ax.legend(loc='upper right')
    plt.savefig(sbfh)
    plt.close()


#########################
# 20
#########################
def rank_features(X, Y, method='MI', continuous=True):
    """
    Rank predictive features on the basis of their Mutual Information with the target.

    :param X:  A DataFrame.  The predictors.
    :param Y:  A Pandas Series.  The targets.
    :param method:  Str.  there are two implementations of MI available, indicated with 'MI' or 'MI2'
    :param continuous:  Bool.  False means the targets are treated as catecorical.
    :return:  A list of indices of predictors ranked by mutual information with the target.
    """
    if method == 'MI':
        if continuous:
            MIh = MIfsmethr(X, Y)
        else:
            MIh = MIfsmeth(X, Y)
        FI_orderh = list(np.argsort(MIh)[::-1])
    if method == 'MI2':
        MIh =[]
        for xcol in range(X.shape[1]):
            MIh.append(lmi.mutual_information((X.iloc[:, xcol].values.reshape(-1, 1), Y.values.reshape(-1, 1)), k=4))
        FI_orderh = list(np.argsort(MIh)[::-1])
    return FI_orderh


#########################
# 21
#########################
def normalize_column(datacp, colname, rnfact):
    """
    Normalize a column of data

    :param datacp:  DataFrame.  A copy of the full dataset.
    :param colname:  Str.  The name of the column to be normalized.
    :param rnfact:  A float or a list.  If this is a float or a list of length 1, normalize the column by deviding by this value.
                    If this is a list of length 2, normalize into the range specified by the list elements.
    :return:  DataFrame.  The dataset with the indicated column replaced by a normalized version of the column.
    """
    if isinstance(rnfact, float) | isinstance(rnfact, int):
        datacp[colname] = datacp[colname] / float(rnfact)
    if isinstance(rnfact, list):
        if len(rnfact) == 1:
            datacp[colname] = datacp[colname] / float(rnfact[0])
        if len(rnfact) == 2:
            datacp[colname] = (datacp[colname] - float(rnfact[0])) / (float(rnfact[1]) - float(rnfact[0]))
    return datacp


#########################
# 22
#########################
def fit_and_test_regression(train_x_h, train_y_h, test_x_h, test_y_h, modtype_h, MLparams_h):
    """
    Fit and evaluate a multivariate regression ML model to the data

    :param train_x_h:  DataFrame.  The training dataset.
    :param train_y_h:  DataFrame.  The training set targets.
    :param test_x_h:  DataFrame.  The test dataset.
    :param test_y_h:  DataFrame.  The test set targets.
    :param modtype_h:  Str.  The type of ML method to use, among: linearRegression,
                            SVR, DT, GBT, MLP, TF_MLP, LGB.
    :param MLparams_h:  Dict.  The hyperparameters for the ML model.
    :return:  Mean squared error of predictions, predictions, and the fitted model.
    """
    if modtype_h == 'linearRegression':
        error_2, predictions_2, model_2 = linear_regression_all(train_x_h,
                                                                train_y_h,
                                                                test_x_h,
                                                                test_y_h,
                                                                use_poly_l=MLparams_h['use_polynomial'],
                                                                use_ridge_l=MLparams_h['use_ridgeregression'])
    if modtype_h == 'SVR':
        error_2, predictions_2, model_2 = svm_regression_all(train_x_h,
                                                             train_y_h,
                                                             test_x_h,
                                                             test_y_h,
                                                             Cloc=MLparams_h['C'],
                                                             gammaloc=MLparams_h['gamma'],
                                                             epsilonloc=MLparams_h['epsilon'])
    if modtype_h == 'DT':
        error_2, predictions_2, model_2 = decision_tree_regression_all(train_x_h,
                                                                      train_y_h,
                                                                      test_x_h,
                                                                      test_y_h,
                                                                      max_depth_loc=MLparams_h['max_depth'])
    if modtype_h == 'GBT':
        if 'n_estimators' not in MLparams_h:
            MLparams_h['n_estimators'] = 100
        if 'colsample_bytree' not in MLparams_h:
            MLparams_h['colsample_bytree'] = 1
        error_2, predictions_2, model_2 = xgBoost_regression_all(train_x_h,
                                                                 train_y_h,
                                                                 test_x_h,
                                                                 test_y_h,
                                                                 max_depth_loc=MLparams_h['max_depth'],
                                                                 min_child_weight_loc=MLparams_h['min_child_weight'],
                                                                 n_estimators_loc = MLparams_h['n_estimators'],
                                                                 colsample_bytree_loc = MLparams_h['colsample_bytree'])
    if modtype_h == 'MLP':
        if 'hidden_layer_sizes' not in MLparams_h:
            MLparams_h['hidden_layer_sizes'] = "5"
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
        error_2, predictions_2, model_2 = multi_layer_perceptron_regression_all(train_x_h,
                                                             train_y_h,
                                                             test_x_h,
                                                             test_y_h,
                                                             hidden_layer_sizes_loc=MLparams_h['hidden_layer_sizes'],
                                                             activation_loc=MLparams_h['activation'],
                                                             solver_loc=MLparams_h['solver'],
                                                             learning_rate_loc=MLparams_h['learning_rate'],
                                                             max_iter_loc=MLparams_h['max_iter'],
                                                             learning_rate_init_loc=MLparams_h['learning_rate_init'],
                                                             alpha_loc=MLparams_h['alpha'],
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
        error_2, predictions_2, model_2 = tensor_flow_regression_all(train_x_h,
                                                                train_y_h,
                                                                test_x_h,
                                                                test_y_h,
                                                                hidden_layer_sizes_loc= MLparams_h['hidden_layer_sizes'],
                                                                activation_loc=MLparams_h['activation'],
                                                                solver_loc=MLparams_h['solver'],
                                                                batch_size_loc=MLparams_h['batch_size'])
    if modtype_h == 'RF_R':
        if 'max_depth' not in MLparams_h:
            MLparams_h['max_depth'] = 18
        if 'min_samples_split' not in MLparams_h:
            MLparams_h['min_samples_split'] = 3
        if 'n_estimators' not in MLparams_h:
            MLparams_h['n_estimators'] = 50
        if 'max_features' not in MLparams_h:
            MLparams_h['max_features'] = 0.5
        error_2, predictions_2, model_2 = randomforest_regression_all(train_x_h,
                                                                train_y_h,
                                                                test_x_h,
                                                                test_y_h,
                                                                max_depth_loc=MLparams_h['max_depth'],
                                                                min_samples_split_loc=MLparams_h['min_samples_split'],
                                                                n_estimators_loc=MLparams_h['n_estimators'],
                                                                max_features_loc=MLparams_h['max_features'])
        # if modtype_h == 'LGB':
    #     if 'num_leaves' not in MLparams_h:
    #         MLparams_h['num_leaves'] = 150
    #     if 'max_depth' not in MLparams_h:
    #         MLparams_h['max_depth'] = 7
    #     if 'learning_rate' not in MLparams_h:
    #         MLparams_h['learning_rate'] = 0.05
    #     if 'max_bin' not in MLparams_h:
    #         MLparams_h['max_bin'] = 200
    #     if 'num_round' not in MLparams_h:
    #         MLparams_h['num_round'] = 50
    #     error_2, predictions_2, model_2 = LGB_regression_All(train_x_h,
    #                                                          train_y_h,
    #                                                          test_x_h,
    #                                                          test_y_h,
    #                                                          num_leaves_loc=MLparams_h['num_leaves'],
    #                                                          max_depth_loc=MLparams_h['max_depth'],
    #                                                          learning_rate_loc=MLparams_h['learning_rate'],
    #                                                          max_bin_loc=MLparams_h['max_bin'],
    #                                                          num_round_loc=MLparams_h['num_round'])
    return error_2, predictions_2, model_2


#########################
# 23
#########################
def balance_data(train_x_h, train_y_h, cntar_h):
    """
    Create subset of the data in which the target value is different from the lag 1 target value for 50% of the samples.

    :param train_x_h:  DataFrame.  The training set.
    :param train_y_h:  DataFrame.  The training set targets.
    :param cntar_h:  Str.  The name of the target column.
    :return:  The data subset, the targets subset, the indices of the kept data in the original data.
    """
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


#########################
# 24
#########################
def filter_features(train_x_h, train_y_h, test_x_h, FSparams_h):
    """
    Create a subset of the data in which only the highest ranked column found through feature selection are kept.

    :param train_x_h:  DataFrame.  The training data.
    :param train_y_h:  Pandas Series.  The training set targets.
    :param test_x_h:  DataFrame.  The test set.
    :param FSparams_h:  Dict.  The feature selection method and number of features to select.
    :return:  the filtered training data, the filtered test data, the list of selected features.
    """
    if FSparams_h['do_FS']:
        if FSparams_h['FSmethod'] == 'f_regression':
            selector = SelectKBest(f_regression, k=FSparams_h['NSF'])
            selector.fit(train_x_h, train_y_h)
            SelectedFeatures_loc = selector.get_support(indices=True).tolist()
            train_x_f = train_x_h.iloc[:,SelectedFeatures_loc]
            test_x_f = test_x_h.iloc[:,SelectedFeatures_loc]
        else:
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


#########################
# 25
#########################
def feature_selection_loops_from_dictionary(FSparamdict_h):
    """
    Create a list of dictionaries of feature selection settings from a dictionary

    :param FSparamdict_h:  Dict.  The dictionary of feature selection settings.
    :return:  List.  The list of dictionaries of feature selection settings.
    """
    FSparamsLooplist_h = []
    if False in FSparamdict_h['do_FS']:
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': False,'FSmethod': 'NoFS','NSF': 0}]
    if True in FSparamdict_h['do_FS']:
        IL =list(itertools.product([True],FSparamdict_h['FSmethod'],FSparamdict_h['NSF']))
        FSparamsLooplist_h = FSparamsLooplist_h + [{'do_FS': xx[0], 'FSmethod': xx[1], 'NSF': xx[2]} for xx in IL]
    return FSparamsLooplist_h


#########################
# 26
#########################
def assemble_predictions(FullPredsDict_l):
    """
    Add a dictionary of assembled (over cross validation folds) predictions to a ML results dictionary.

    :param FullPredsDict_l:  Dict.  An ML results dictionary.
    :return:  The amended ML results dictionary.
    """
    FullPredsDict_l['FullPredictionsAssembled']={}
    foldind = FullPredsDict_l['FullPredictionsKeyMeaning'].index('Fold')
    fpk = [make_tuple(xx) for xx in FullPredsDict_l['FullPredictions'].keys()]
    fpkl = [list(xx) for xx in fpk]
    fpklm = [xx[:foldind] + xx[foldind + 1:] for xx in fpkl]
    fpklmu = [list(i) for i in set(tuple(i) for i in fpklm)]
    nwof = FullPredsDict_l['FullPredictionsKeyMeaning'][:foldind] + \
           FullPredsDict_l['FullPredictionsKeyMeaning'][foldind + 1:]
    FullPredsDict_l['FullPredictionsAssembledKeyMeaning'] = nwof
    newrepind = nwof.index('Rep')
    for ml in fpklmu:
        FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)] = np.full(len(FullPredsDict_l['Target']), np.nan)
        for fn in range(FullPredsDict_l['K']):
            predinds = FullPredsDict_l['RepFoldIndices'][ml[newrepind]][fn]
            fullkey = ml[:]
            fullkey.insert(foldind,fn)
            FullPredsDict_l['FullPredictionsAssembled'][tuple(ml)][predinds] = FullPredsDict_l['FullPredictions'][str(tuple(fullkey))]
    for tk in FullPredsDict_l['FullPredictionsAssembled'].keys():
        FullPredsDict_l['FullPredictionsAssembled'][tk] = FullPredsDict_l['FullPredictionsAssembled'][tk].tolist()
    return FullPredsDict_l


#########################
# 27
#########################
def calculate_mean_squared_errors(FullPredsDict_l):
    """
    Add mse's of assembled predictions to a ML results dictionary

    :param FullPredsDict_l:  Dict.  The ML results dictionary that must contain assembled (over CV folds) predictions.
    :return:  The amended ML results dictionary
    """
    FullPredsDict_l['MSE']={}
    for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
        preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
        FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
    return FullPredsDict_l


#########################
# 28
#########################
def calculate_mean_and_std_of_mse(FullPredsDict_l):
    """
    Add the mean and std of the mse's of the assembled predictions to a ML results dictionary

    :param FullPredsDict_l:  Dict.  A ML results dictionary that must contain assembled (over CV folds) predictions.
    :return: The amended ML results dictionary
    """
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


#########################
# 29
#########################
def parse_mean_and_std_of_mse(FullPredsDict_l):
    """
    For each combination of ML method and FS method, create 3D mean mse array and 3D std mse array, add them to a ML results dictionary

    :param FullPredsDict_l: Dict.  A ML results dictionary that must contain assembled (over CV folds) predictions
                                    and their means and stds.
    :return:  The amended ML results dictionary
    """
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


#########################
# 30
#########################
# def parse_machine_learning_pipeline_results(FullPredsDict_l):
#     """
#     Add mse, mean mse, std mse, to a ML results dictionary for each combination of model type, model hyperparameters,
#     feature selection method, and number of selected features.
#
#     :param FullPredsDict_l:  Dict.  A ML results dictionary.
#     :return: The amended ML results dictionary
#     """
#     FullPredsDict_l['MSE']={}
#     for thiskey in FullPredsDict_l['FullPredictionsAssembled'].keys():
#         preds = FullPredsDict_l['FullPredictionsAssembled'][thiskey]
#         FullPredsDict_l['MSE'][thiskey] = mean_squared_error(preds, FullPredsDict_l['Target']) # this was Tar..
#     # get names for all of the independent runs for which we want mean and std of MSE
#     FullPredsDict_l['MSE_mean']={}
#     FullPredsDict_l['MSE_std']={}
#     URN = list(set(["Train"+ x.split("_Train")[1] for x in FullPredsDict_l['FullPredictionsAssembled'].keys()]))
#     for urn in URN:
#         thesekeys = [x for x in FullPredsDict_l['FullPredictionsAssembled'].keys() if urn in x]
#         thesemses = [FullPredsDict_l['MSE'][xx] for xx in thesekeys]
#         FullPredsDict_l['MSE_mean'][urn] = np.mean(thesemses)
#         FullPredsDict_l['MSE_std'][urn] = np.mean(thesemses)
#     # for each combination of ML method and FS method, create 3D mean mse array and 3D std mse array
#     FullPredsDict_l['ResultsByMethod_MLxFS_mean']={}
#     FullPredsDict_l['ResultsByMethod_MLxFS_std']={}
#     modparamnames = FullPredsDict_l['modelParameters'].keys()
#     modparamlens = [len(FullPredsDict_l['modelParameters'][xx]) for xx in modparamnames]
#     FSnums = FullPredsDict_l['FSParameters']['NSF']
#     for dfs in FullPredsDict_l['FSParameters']['do_FS']:
#         if not dfs:
#             mmse = np.full(modparamlens,np.nan)
#             smse = np.full(modparamlens, np.nan)
#             thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if 'FSmethod_NoFS' in xx]
#             for inds, vals in np.ndenumerate(mmse):
#                 matchterms = [modparamnames[xx]+"_%s"%FullPredsDict_l['modelParameters'][modparamnames[xx]][inds[xx]] for xx in range(len(modparamnames))]
#                 thiskey =[xx for xx in thesekeys if all([matchterms[zz] in xx for zz in range(len(matchterms))])]
#                 if len(thiskey) > 1:
#                     print "Problem with too many matches"
#                 mmse[inds] = FullPredsDict_l['MSE_mean'][thiskey[0]]
#                 smse[inds] = FullPredsDict_l['MSE_std'][thiskey[0]]
#             FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype']+"_NoFS"] = mmse
#             FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_NoFS"] = mmse
#         else:
#             MPFd = modparamlens + [len(FSnums)]
#             for fsm in FullPredsDict_l['FSParameters']['FSmethod']:
#                 mmse = np.full(MPFd, np.nan)
#                 smse = np.full(MPFd, np.nan)
#                 thesekeys = [xx for xx in FullPredsDict_l['MSE_mean'].keys() if 'FSmethod_'+fsm in xx]
#                 for inds, vals in np.ndenumerate(mmse):
#                     matchterms = [modparamnames[xx] + "_%s" % FullPredsDict_l['modelParameters'][modparamnames[xx]][inds[xx]]
#                               for xx in range(len(modparamnames))] +['NumFS_%s'%FSnums[inds[len(modparamnames)]]]
#                     thiskey = [xx for xx in thesekeys if all([matchterms[zz] in xx for zz in range(len(matchterms))])]
#                     if len(thiskey) > 1:
#                         print "Problem with too many matches"
#                     mmse[inds] = FullPredsDict_l['MSE_mean'][thiskey[0]]
#                     smse[inds] = FullPredsDict_l['MSE_std'][thiskey[0]]
#                 FullPredsDict_l['ResultsByMethod_MLxFS_mean'][FullPredsDict_l['modtype'] + "_" + fsm] = mmse
#                 FullPredsDict_l['ResultsByMethod_MLxFS_std'][FullPredsDict_l['modtype'] + "_" + fsm] = mmse
#     return FullPredsDict_l


#########################
# 31
#########################
def merge_full_predictions(fpd1_in, fpd2):
    """
    Merge two dictionaries of predictions and selected features.

    :param fpd1_in:  Dict.  A dictionary of predictions and selected features.
    :param fpd2:  Dict.  A dictionary of predictions and selected features.
    :return:  The merged dictionary of predictions and selected features.
    """
    fpd1 = copy.deepcopy(fpd1_in)
    fpd1['SelectedFeatures'].update(fpd2['SelectedFeatures'])
    fpd1['FullPredictions'].update(fpd2['FullPredictions'])
    return fpd1


#########################
# 32
#########################
def apply_threshold(my_list, my_number):
    """
    Return a number which is my_number if my_number falls within (my_list[0], my_list[1]), otherwise the closest element
    in my_list to my_number.

    :param my_list: List. The list of minimum threshold and maximum threshold.
    :param my_number: Float or Int. The model prediction value.
    :return: Float or Int. The modified prediction value.
    """
    return min(my_list[1], max(my_list[0], my_number))


#########################
# 33
#########################
def prediction_feedback(mod, dat, Target_loc, numlags, Roundlist=[], Threshlist=[]):
    """
    Apply a trained mixed autoregressive model to predict targets using feedback.  This means that after every time point,
    the predictions are used as predictors for the next time point.

    :param mod:  Obj.  The trained model.
    :param dat:  DataFrame.  The test dataset.
    :param Target_loc:  Str.  The name of the target variable.
    :param numlags:  Int.  The number of lags
    :param Roundlist:  List.  The list of values that the predictions should be rounded to.
    :param Threshlist:  List.  The list of thresholds which the predictions should be truncated into.
    :return:  The predictions.
    """
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
            Pred1 = apply_threshold(Threshlist, Pred1)
        bufferd.appendleft(Pred1)
        bufferd.pop()
        PredictionsList[sampnum] = Pred1
    return PredictionsList


#########################
# 34
#########################
def prediction_feedback_stacked(mod, dat, Target_loc, numlags, seed_loc):
    """
    Apply a trained mixed autoregressive model to predict targets using feedback given a target value for the first time point.

    :param mod:  Obj.  The trained model.
    :param dat:  DataFrame.  The test dataset.
    :param Target_loc:  Str.  The name of the target variable.
    :param numlags:  Int.  The number of lags
    :param seed_loc:  Float.  The target value for the first time point.
    :return:  The predictions.
    """
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


#########################
# 35
#########################
def prediction_feedback_stacked_simultaneous(mod_gm, mod_pm, colsforGm, colsforPm, dat, Target_loc, numlags, seed_loc=np.nan, GM_gets_truth=True):
    """
    Apply two trained mixed autoregressive models to make stacked predictions using feedback.

    :param mod_gm:  Obj.  The trained General model.
    :param mod_pm:  Obj.  The trained Personal model.
    :param colsforGm:  List.  The list of predictors used by the General Model.
    :param colsforPm:  List.  The list of predictors used by the Personal Model.
    :param dat:  DataFrame.  The test data.
    :param Target_loc:  Str.  The name of the target variable.
    :param numlags:  Int.  The number of lags.
    :param seed_loc:  Float.  The target value for the first time point.
    :param GM_gets_truth:  Boolean.  True mean the General Model predicts on the basis of the lagged ground truth.  False means it uses feecback prediction.
    :return:  Stacked Predictions, GM predictions, PM predictions
    """
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


#########################
# 36
#########################
def prediction_feedback_stacked_simultaneous_preexisting_gm_predictions(PredictionsListGM, mod_pm, colsforPm, dat, Target_loc, numlags, seed_loc=np.nan):
    """
    Apply two trained mixed autoregressive models to make stacked predictions using feedback, with pre-existing GM predictions calculated beforehand.

    :param PredictionsListGM:  DataFrame.  The pre-caluclated General Model predictions.
    :param mod_pm:  Obj.  The personal model.
    :param colsforPm:  List.  The names of the predictors for the Personal Model.
    :param dat:  DataFrame.  The test data.
    :param Target_loc:  Str.  The name of the target variable.
    :param numlags:  Int.  The number of lags.
    :param seed_loc:  Float.  The intitial value of the delta between GM prediction and ground truth target value.
    :return:  Stacked predictions, General Model predictions, Personal Model predictions.
    """
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


#########################
# 37
#########################
def check_timestamp_order(dat):
    """
    Assert that the 'timestamp' column in a dataset is in order from earliest to latest.

    :param dat:  DataFrame.  A dataset containing a column called 'timestamp'
    :return: True if the timestamp column is in the correct order, otherwise False
    """
    tsdt = pd.to_datetime(dat['timestamp'], format='%Y-%m-%d %H:%M:%S')
    return tsdt.tolist() == sorted(tsdt.tolist())


#########################
# 38
#########################
def full_plot(winpreds_h, Tar_h, ptss_h):
    """
    Produce a plot of target values on the y-axis, and time on the x-axis.  Blue predictions, Red for truth.

    :param winpreds_h:  list.  Predicted target values.
    :param Tar_h:  list.  True target values.
    :param ptss_h:  Int.  The number of time points to display in the plot (starting from the first)
    :return:  a handle to the plot
    """
    # take a look at the two sets of predictions vs true
    fig,axarr = plt.subplots(1, sharex=True, figsize=(10, 1))
    axarr.plot(winpreds_h[0:ptss_h], color='blue')
    axarr.plot(Tar_h[0:ptss_h], color='red')
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 + 0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['Predicted', 'True Value'], loc='center left', bbox_to_anchor=(1, 0.5))
    return plt


#########################
# 39
#########################
def full_plot_axis(winpreds_h, Tar_h, ptss_h, axarr, ttl="", Leg=['Predicted', 'True Value'], ylims=[-0.05,1.05], colset=['green', 'black']):
    """
    Produce a plot of target values on the y-axis, and time on the x-axis, on the given matplotlib axis.

    :param winpreds_h:  list.  Predicted target values.
    :param Tar_h:  list.  True target values.
    :param ptss_h:  Int.  The number of time points to display in the plot (starting from the first)
    :param axarr:  Matplotlib axis Obj.  The axis on which to make the plot.
    :param ttl:  Str.  The title string for the plot.
    :param Leg:  list.  A list of strings indicating the names assigned to the two lists of predictions.
    :param ylims:  list.  The min and max y-limits.
    :param colset:  list.  The colors to be used for the two curves.
    :return: a handle to the plot
    """
    axarr.set_ylim(ylims)
    if Leg == ['Predicted', 'True Value']:
        colset = ['blue','red']
    axarr.plot(winpreds_h[0:ptss_h], color=colset[0])
    axarr.plot(Tar_h[0:ptss_h], color=colset[1])
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 + 0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(Leg, loc='center left', bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt


#########################
# 40
#########################
def feature_plot_axis(TripData_h, ptss_h, axarr, ttl="Environment"):
    """
    Plot the evironmental conditions for a trip against time on a given matplotlib axis.

    :param TripData_h:  DataFrame.  The trip data containing the needed environmental variables.
    :param ptss_h:  Int.  The number of time points (from the trip start) to include in the plot.
    :param axarr:  Matplotlib axis Ojb.  The axis on which to make the plot.
    :param ttl:  Str.  The title for the figure.
    :return: a handle to the plot
    """
    colset = cm.rainbow(np.linspace(0, 1, 8))
    axarr.plot((TripData_h['OtsAirTmpCrVal_lag_1'][0:ptss_h]/max(TripData_h['OtsAirTmpCrVal_lag_1'])).tolist(), color=colset[0])
    axarr.plot((TripData_h['IPSnsrSolrInt_lag_1'][0:ptss_h]/max(TripData_h['IPSnsrSolrInt_lag_1'])).tolist(), color=colset[1])
    axarr.plot((TripData_h['EngSpd_lag_1'][0:ptss_h]/max(TripData_h['EngSpd_lag_1'])).tolist(), color=colset[2])
    axarr.plot((TripData_h['DriverSetTemp_lag_1'][0:ptss_h]/max(TripData_h['DriverSetTemp_lag_1'])).tolist(), color=colset[3])
    axarr.plot((TripData_h['WindPattern_lag_1'][0:ptss_h]).tolist(), color=colset[4])
    axarr.plot((TripData_h['WindLevel_lag_1'][0:ptss_h]).tolist(), color=colset[5])
    axarr.plot((TripData_h['LftLoDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftLoDctTemp_lag_1'])).tolist(), color=colset[6])
    axarr.plot((TripData_h['LftUpDctTemp_lag_1'][0:ptss_h]/max(TripData_h['LftUpDctTemp_lag_1'])).tolist(), color=colset[7])
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0 + 0.1, box.width * 0.8, box.height * 0.7])
    axarr.legend(['OtsAirTmpCrVal_lag_1', 'IPSnsrSolrInt_lag_1', 'EngSpd_lag_1', 'DriverSetTemp_lag_1', 'WindPattern_lag_1', 'WindLevel_lag_1', 'LftLoDctTemp_lag_1', 'LftUpDctTemp_lag_1'],
                 loc='center left',
                 fontsize=4,
                 bbox_to_anchor=(1, 0.5))
    axarr.set_title(ttl)
    return plt


#########################
# 41
#########################
def score_buffer(true_value, prediction, buffer=1):
    """
    Return True or False variable to indicate if the prediction meet the requirement given a buffer

    :param true_value: Float or Int. true target value.
    :param prediction: Float or Int. model prediction.
    :param buffer: Float or Int. given allowance of difference between true and predicted values
    :return: new variable indicating if the prediction meet the requirement given a buffer
    """
    return true_value - buffer <= prediction <= true_value + buffer


#########################
# 42
#########################
def calculate_scores(true_value, prediction, range):
    """
    Return accuracy_score, buffer_score, mean squared error and relative_error.

    :param true_value: pandas Series. true target value.
    :param prediction: pandas Series. model prediction.
    :param range: Float or Int. the range of this target
    :return: accuracy_score, buffer_score, mean_squared_error and relative_error
    """
    a_score = accuracy_score(true_value, prediction)
    dff = pd.DataFrame({'tv': true_value, 'pred': prediction})
    dff['s'] = dff.apply(lambda row: score_buffer(row['tv'], row['pred']), axis=1)
    dff['re'] = abs(dff['tv'] - dff['pred']) / range
    b_score = dff['s'].mean()
    mse_score = mean_squared_error(true_value, prediction)
    r_score = dff['re'].mean()
    return a_score, b_score, mse_score, r_score


#########################
# 43
#########################
def list_all_files(address):
    """
    List all files with absolute path under certain address

    :param address: string. an absolute address
    :return: a list named absolute_file_list, containing all the files under address
    """
    absolute_file_list = []
    for root, dirs, files in os.walk(address):
        if len(files):
            for file in files:
                if not file.startswith("."):
                    f = root + '/' + file
                    absolute_file_list.append(f)
    return absolute_file_list


#########################
# 44
#########################
def encoding_wind_pattern_prediction_map(face, facefoot, foot, footdefrost):
    """
    Transfer encoded 4 prediction values for wind pattern back to one prediction.

    :param face: Float. a number prediction
    :param facefoot: Float. a number prediction
    :param foot: Float. a number prediction
    :param footdefrost: Float. a number prediction
    :return: a string prediction for wind pattern
    """
    lst = [face, facefoot, foot, footdefrost]
    a = lst.index(max(lst))
    if a == 0:
        return 'face'
    elif a == 1:
        return 'facefoot'
    elif a == 2:
        return 'foot'
    else:
        return 'footdefrost'


#########################
# 45
#########################
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Print and plot the confusion matrix. Normalization can be applied by setting `normalize=True`.

    :param cm: numpy.ndarray. confusion matrix
    :param classes: List. should be a list of all possible values of the target variable
    :param normalize: Boolean. Normalization can be applied by setting `normalize=True`.
    :param title: String. title of the plot
    :param cmap: matplotlib.colors.LinearSegmentedColormap. color map
    :return: a plot of confusion matrix
    example:
    from sklearn.metrics import confusion_matrix
    pattern_cnf_matrix = confusion_matrix(pattern_y_test, pattern_y_pred)
    plot_confusion_matrix(pattern_cnf_matrix, classes=['face', 'facefoot', 'foot', 'footdefrost'], title='windPattern')
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True value')


#########################
# 46
#########################
def lat(pvc):
    """
    Map from Province name to latitude

    :param pvc: String. province name
    :return: latitude
    """
    return pvc.map({'江苏省': 32.950455135, '浙江省': 29.445178145, '上海市': 31.21645245, '安徽省': 32.22516349,
                    '山东省': 36.160414245, '江西省': 27.82497988, '河南省': 34.10517794, '北京市': 39.88240928,
                    '四川省': 29.489994525, '广东省': 22.99997723, '辽宁省': 40.61160689, '湖南省': 27.71626559,
                    '湖北省': 31.2542796, '福建省': 25.87518933, '云南省': 25.05053682, '河北省': 38.77017743,
                    '广西壮族自治区': 23.3801995, '贵州省': 26.39522219, '山西省': 37.7901984, '陕西省': 35.48164431,
                    '内蒙古自治区': 44.81116529, '重庆市': 30.30770947, '天津市': 39.18099215})


#########################
# 47
#########################
def lng(pvc):
    """
    Map from Province name to longitude

    :param pvc: String. province name
    :return: longitude
    """
    return pvc.map({'江苏省': 119.0025189, '浙江省': 120.21001625, '上海市': 121.4365047, '安徽省': 117.0526403,
                    '山东省': 118.7750134, '江西省': 115.9100203, '河南省': 113.86502885, '北京市': 116.08913085,
                    '四川省': 103.930002199999, '广东省': 113.5250238, '辽宁省': 122.3735465, '湖南省': 111.5390468,
                    '湖北省': 112.9399949, '福建省': 117.9683443, '云南省': 101.3581437, '河北省': 117.0500024,
                    '广西壮族自治区': 108.9666715, '贵州省': 106.9816764, '山西省': 112.3900026, '陕西省': 108.7299935,
                    '内蒙古自治区': 114.6926528, '重庆市': 108.0558227, '天津市': 117.4885028})


#########################
# 48
#########################
def correlation_matrix(absolute_data_file, label_list):
    """
    Create a correlation matrix for a data file and its certain columns defined in the label_list and save a .png plot
    in the same directory with the same name as the file.

    :param absolute_data_file: string. data file with absolute address
    :param label_list: List. a list of columns you want to be included in the correlation matrix
    :return: save a correlation matrix .png plot in the same directory with the same name as the file.
    """
    f = pd.read_csv(absolute_data_file)
    df = f[label_list]
    df['Season'] = df['Season'].map({'Winter': 1.0, 'Fall': 2.0, 'Spring': 3.0, 'Summer': 4.0})
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    major_ticks = np.arange(0, len(label_list), 1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    cmap = cm.get_cmap('jet')
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True, which='major', linestyle='--', color='grey')
    plt.title('Features Correlation')
    labels = label_list
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    plt.xticks(rotation=45)
    absolute_address = absolute_data_file.rsplit('/', 1)[0]
    file_name = absolute_data_file.rsplit('/', 1)[1].rsplit('.', 1)[0]
    plt.savefig(absolute_address + '/' + file_name + 'png')
    plt.close()


#########################
# 49
#########################
def nearest_round(my_list, my_number):
    """
    Round a float number to the nearest level provided in the list

    :param my_list: List. a list of levels, for example: [1, 2, 3, 4, 5, 6, 7, 8]
    :param my_number: Float or Int. a number, for example: 6.86
    :return: a level in the list which is the closest to the number
    """
    return min(my_list, key=lambda x: abs(x - my_number))


#########################
# 50
#########################
def normalize_df_column(datacp, colname, min, max):
    """
    Take a DataFrame as input, normalize a column, then return the new DataFrame with the normalized column

    :param datacp: pandas DataFrame
    :param colname: String. the column name you want to normalize
    :param min: Float or Int. The minimum possible value of the column
    :param max: Float or Int. The maximum possible value of the column
    :return: the new DataFrame with the normalized column
    example:
    Normalization = {'DayOfYear': (0.0, 365.0)}
    for normkey in Normalization.keys():
        dataset = hf.normalizeColumn(dataset, normkey, Normalization[normkey][0], Normalization[normkey][1])
    """
    datacp[colname + '_normalized'] = (datacp[colname] - min) / float(max - min)
    return datacp


#########################
# 52
#########################
def denormalize_df_column(datacp, colname, min, max):
    """
    Take a DataFrame as input, de-normalize a column, then return the new DataFrame with the normalized column

    :param datacp: pandas DataFrame
    :param colname: String. The column name you want to denormalize
    :param min: Float or Int. the minimum possible value of the column
    :param max: Float or Int. the maximum possible value of the column
    :return: the new DataFrame with the de-normalized column
    """
    datacp[colname + '_denormalized'] = datacp[colname] * float(max - min) + min
    return datacp


#########################
# 53
#########################
def many_to_one_row_load_data(data, section_id, predictor_set, target_set):
    """
    Given a DataFrame, group by section_id, generate a list of X which has all rows of the predictor_set in each section
    and a list of y which has the last row of the target_set in each section.

    :param data: pandas DataFrame
    :param section_id: String. the column name representing the section_id
    :param predictor_set: List. a list of Strings of the predictors' names
    :param target_set: List. a list of String of the targets' names
    :return: a list of X and a list of y
    """
    docX, docY = [], []
    vtTab = data[section_id].unique()
    vts = list(vtTab)
    numvts = len(vts)
    for tn in range(numvts):
        vt = vts[tn]
        vtdat = data[data[section_id] == vt]
        X = vtdat[predictor_set]
        y = vtdat[target_set].iloc[-1]
        docX.append(X)
        docY.append(y)
    # alsX = np.array(docX)
    # alsY = np.array(docY)
    return docX, docY


#########################
# 54
#########################
def number_sec(df, col_name):
    """
    Return the number of sections for certain column. In one section the value of this column continuously keep in the same level

    :param df: pandas DataFrame
    :param col_name: String. the column name
    :return: total number of sections in df
    """
    test = df
    test['change1'] = test[col_name] - test[col_name].shift(-1)
    test2 = test[test['change1'] == 0.0]
    if len(test2) > 0:
        test2['change2'] = test2[col_name] - test2[col_name].shift()
        test2['label'] = test2['change2'].apply(lambda x: x if x == 0.0 else 1)
        test2['secID'] = test2['label'].astype(int).cumsum()
        return test2['secID'].max()
    else:
        return len(test)


#########################
# 55
#########################
def valid_section_add_id(test, dlimit):
    """
    Add Section ID to a DataFrame. Valid section means the duration is greater than the dlimit

    :param test: pandas DataFrame
    :param dlimit: Float or Int. duration minimum limit in seconds
    :return: a new DataFrame with valid sections and secID
    """
    colset1 = list(test.columns)
    test['change1'] = test['DriverSetTemp'] - test['DriverSetTemp'].shift(-1)
    test2 = test[test['change1'] == 0.0]
    if len(test2) > 0:
        test2['change2'] = test2['DriverSetTemp'] - test2['DriverSetTemp'].shift()
        test2['label'] = test2['change2'].apply(lambda x: x if x == 0.0 else 1)
        test2['secID'] = test2['label'].astype(int).cumsum()
        m = test2['secID'].max()
        # print('Number of Changes: %s' % m)
        sdflst = [pd.DataFrame()] * m
        for i in range(m):
            sdflst[i] = test2[test2['secID'] == i+1]
            sdflst[i]['ts'] = sdflst[i]['timestamp'].\
                apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
            # maxt = datetime.datetime.strptime(sdflst[i]['timestamp'].max(), "%Y-%m-%d %H:%M:%S.%f")
            # mint = datetime.datetime.strptime(sdflst[i]['timestamp'].min(), "%Y-%m-%d %H:%M:%S.%f")
            maxt = sdflst[i]['ts'].max()
            mint = sdflst[i]['ts'].min()
            tduration = (maxt - mint).seconds
            sdflst[i]['flag'] = sdflst[i]['secID'].apply(lambda x: x if tduration > dlimit else 0)
        test2df = pd.concat(sdflst)
        return test2df[test2df['flag'] > 0][colset1 + ['secID']]
    else:
        return pd.DataFrame(columns=colset1+['secID'])


#########################
# 56
#########################
def percent_sample(df3, beta, alpha, uncomfortable_time=120):
    """
    Modify the Target DriverSetTemp. the length is alpha percent of last section plus beta percent of current section.
    The value should be the same value as current section.

    :param df3: pandas DataFrame
    :param beta: Float. percent of current section
    :param alpha: Float. percent of last section
    :param uncomfortable_time: Float or Int. the last part (seconds) of time in a section.
    :return: a new DataFrame with DriverSetTemp_newTarget
    """
    colset2 = list(df3.columns)
    if len(df3) > 0:
        uID = df3['secID'].unique()
        luID = len(uID)
        seclst = [pd.DataFrame()] * luID
        for aa in range(luID):
            if aa == 0:
                s_now = df3[df3['secID'] == uID[aa]]
                l_now = len(s_now)
                tt = s_now[:int(l_now*beta)]
                tt['DriverSetTemp_newTarget'] = tt['DriverSetTemp_Target']
                seclst[aa] = tt
            else:
                s_last = df3[df3['secID'] == uID[aa - 1]]
                l_last = len(s_last)
                s_last['ts'] = s_last['timestamp'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))
                maxt = s_last['ts'].max()
                threT = maxt - datetime.timedelta(0, uncomfortable_time, 0)
                alpha_ = 1 - len(s_last[s_last['ts'] < threT]) / float(len(s_last))
                s_last = s_last.drop(['ts'], axis=1)
                s_now = df3[df3['secID'] == uID[aa]]
                l_now = len(s_now)
                tt = pd.concat([s_last[int(l_last*(1-min(alpha, alpha_, 1-beta))):], s_now[:int(l_now*beta)]])
                tt['DriverSetTemp_newTarget'] = len(tt) * [tt['DriverSetTemp_Target'].iloc[-1]]
                seclst[aa] = tt
        return pd.concat(seclst)
    else:
        return pd.DataFrame(columns=colset2+['DriverSetTemp_newTarget'])


#########################
# 57
#########################
def plot_compare(df2, tid, beta, alpha, ut=120):
    """
    Visualize the original and new DriverSetTemp

    :param df2: pandas DataFrame
    :param tid: Int. trip ID
    :param beta: Float. percent of current section
    :param alpha: Float. percent of last section
    :param ut: Int. the last part (in seconds) of time in a section.
    :return: a graph with all lines of DriverSetTemp
    """
    colset3 = list(df2.columns)
    t = df2[df2['tripID'] == tid]
    t_ = valid_section_add_id(t, 120)
    p = t.join(t_, how='left', rsuffix='_Target')
    p = p[colset3 + ['DriverSetTemp_Target', 'secID']]
    p = p.fillna(method='bfill')
    pp = percent_sample(p.dropna(axis=0), beta, alpha, uncomfortable_time=ut)
    p.join(pp, how='left', rsuffix='_1').plot(x='timestamp', y=['DriverSetTemp', 'DriverSetTemp_Target',
                                                                'DriverSetTemp_newTarget'], marker='o')
    plt.show()


#########################
# 58
#########################
def check_preds(preds_h):
    """
    check predictions for infinite or missing values

    :param preds_h:  list.  The predictions made by a Machine Learning method.
    :return: Raise a ValueError if the condition is not met
    """
    if not np.isfinite(preds_h).all():
        raise ValueError('Predictions include infinite or missing values.')


#########################
# 59
#########################
def randomforest_classifier_all(x_train, y_train, x_test, y_test, max_depth_loc=18, min_samples_split_loc=3,
                                n_estimators_loc=50, max_features_loc=0.5, sample_weight=False):
    """
    Train a Random Forest Classifier Tree model and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param max_depth_loc:  Int.  The hyperparameter Max Depth.
    :param min_samples_split_loc:  Int.  The hyperparameter Min Sample Split.
    :param n_estimators_loc:  Int.  The hyperparameter number of estimators.
    :param max_features_loc:  Float.  The number of features to consider when looking for the best split.
    :param sample_weight: Boolean. Option to do sample weight or not during training.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators_loc,
                                 max_depth=max_depth_loc,
                                 min_samples_split=min_samples_split_loc,
                                 random_state=0,
                                 max_features=max_features_loc)
    if sample_weight:
        model = clf.fit(x_train, y_train, sample_weight=class_weight.compute_sample_weight('balanced', y_train))
    else:
        model = clf.fit(x_train, y_train)
    predictions = model.predict(x_test)
    # check_preds(predictions)
    acscore = accuracy_score(y_test, predictions)
    return acscore, predictions, model


#########################
# 60
#########################
def randomforest_regression_all(x_train, y_train, x_test, y_test, max_depth_loc=18, min_samples_split_loc=3, n_estimators_loc=50, max_features_loc=0.5):
    """
    Train a Random Forest Regression Tree model and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param max_depth_loc:  Int.  The hyperparameter Max Depth.
    :param min_samples_split_loc:  Int.  The hyperparameter Min Sample Split.
    :param n_estimators_loc:  Int.  The hyperparameter number of estimators.
    :param max_features_loc:  Float.  The number of features to consider when looking for the best split.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
    clf = RandomForestRegressor(n_estimators=n_estimators_loc,
                                max_depth=max_depth_loc,
                                min_samples_split=min_samples_split_loc,
                                random_state=0,
                                max_features=max_features_loc)
    model = clf.fit(x_train, y_train)
    predictions = model.predict(x_test)
    check_preds(predictions)
    error = mean_squared_error(y_test, predictions)
    return error, predictions, model


#########################
# 61
#########################
def list_files(address):
    """
    List all files under certain conditions in a directory.

    :param address: Str. An absolute path
    :return: A list of Strings. Each string is a complete path and file name.
    """
    file_Paths = []
    for root, dirs, files in os.walk(address):
        if len(files):
            for file in files:
                if not file.startswith("."):
                    f = root + '/' + file
                    file_Paths.append(f)
    return file_Paths


#########################
# 62
#########################
def fetch_vehicle(f, vid):
    """
    Return a dataFrame with all of one vehicle's data from the list of files.

    :param f: List of Str. A list of Strings. Each string is a complete path and file name.
    :param vid: Str. Vehicle ID. eg. "vehicleId=68002"
    :return: pandas DataFrame
    """
    df = pd.DataFrame()
    for file_path in f:
        path, month, file_name = file_path.rsplit('/', 2)
        vehicle, track, name = file_name.split('_', 2)
        if vehicle in [vid]:
            df_new = pd.read_csv(file_path)
            if len(df) == 0:
                df = df_new
            else:
                df = df.append(df_new)
    return df


#########################
# 63
#########################
def pre_process_df(dff):
    """
    Add drop na, filter bad data, transformation, normalized columns, longitude and latitude, Az and Zen and
    their sin and cos values to a dataFrame.

    :param dff: pandas DataFrame. the input dataFrame
    :return: a dataFrame with more columns
    """
    # this function won't change the input dataFrame
    if 'Unnamed: 0' in dff.columns:
        df = dff.drop(['Unnamed: 0'], axis=1)
    else:
        df = dff
    # replace all NaN value of acDual with 'off'
    df['acDual'] = df['acDual'].fillna('off')
    # drop any row with missing values
    df = df.dropna(axis=0)
    if len(df) == 0:
        print("No data available.")
        return
    # pre-processing
    if 'str' in str(list(set(map(type, df.windPattern)))):
        df = df[~(df.windPattern == 'auto')]
    tl = list(set(map(type, df.windLevel)))
    if 'str' in str(tl):
        df = df[~((df.windLevel == 'auto') | (df.windLevel == '0') | (df.windLevel == 0))]
    else:
        df = df[df.windLevel != 0]
    df = df[df.vehicleOutsideTemp_last != -40.0]
    df['LacTemperature'] = df['LacTemperature'].apply(lambda x: 15.0 if x == 14.0 else (29.0 if x == 30.0 else x))
    df['RacTemperature'] = df['RacTemperature'].apply(lambda x: 15.0 if x == 14.0 else (29.0 if x == 30.0 else x))
    df["Season"] = df["season"]. \
        apply(lambda x: 1.0 if x == 'Winter' else (2.0 if x == 'Fall' else (3.0 if x == 'Spring' else 4.0)))
    df["WindLevel"] = df["windLevel"].apply(lambda x: int(x) if isinstance(x, str) else x)
    df["WindPattern"] = df["windPattern"]. \
        apply(lambda x: 1.0 if x == 'face' else (2.0 if x == 'facefoot' else (3.0 if x == 'foot' else 4.0)))
    df["AcRecirculation"] = df["acRecirculation"].apply(lambda x: 1.0 if x == 'in' else 0.0)
    df["AcAC"] = df["acAC"].apply(lambda x: 1.0 if x == 'on' else 0.0)
    df["AcAutoMode"] = df["acAutoMode"].apply(lambda x: 1.0 if x == 'on' else 0.0)
    df["AcDual"] = df["acDual"].apply(lambda x: 1.0 if x == 'on' else 0.0)
    # map the province to longitude and latitude
    df['plat'], df['plng'] = lat(df['province']), lng(df['province'])
    # calculate local timestamp and greenwich timestamp
    if 'timestamp' not in df.columns:
        df['timestamp'] = (pd.to_datetime(df['avnTimestamp']) + datetime.timedelta(hours=8)).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['GWavnTimestamp'] = pd.to_datetime(df['avnTimestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        df['GWavnTimestamp'] = (pd.to_datetime(df['avnTimestamp'], utc=True) - datetime.timedelta(hours=8)).dt.strftime(
            '%Y-%m-%d %H:%M:%S')
    # calculate week of year
    df['weekOfYear'] = [int(time.strftime("%W", time.strptime(d, '%Y-%m-%d %H:%M:%S'))) for d in df['timestamp']]
    df['cosWOY'] = df['weekOfYear'].apply(lambda cc: round(math.cos(2 * math.pi * cc / 53), 10))
    df['sinWOY'] = df['weekOfYear'].apply(lambda ss: round(math.sin(2 * math.pi * ss / 53), 10))
    # calculate month of year
    df['monthOfYear'] = [int(d.split('-')[1]) for d in df['yearmonth']]
    df['cosMOY'] = df['monthOfYear'].apply(lambda cc: round(math.cos(2 * math.pi * cc / 12), 10))
    df['sinMOY'] = df['monthOfYear'].apply(lambda ss: round(math.sin(2 * math.pi * ss / 12), 10))
    # normalize some columns
    Normalization = {'dayOfYear': (1.0, 365.0),
                     'dayOfWeek': (1.0, 7.0),
                     'hourOfDay': (0.0, 23.0),
                     'weekOfYear': (0.0, 52.0),
                     'monthOfYear': (1.0, 12.0)}
    for normkey in Normalization.keys():
        df = normalize_df_column(df, normkey, Normalization[normkey][0], Normalization[normkey][1])
    # calculate AZ and ZEN and their sin and cos values
    cst = 2 * math.pi / 360
    df['GWaz'] = df.apply(lambda row:
                          sunpos(datetime.datetime.strptime(row['GWavnTimestamp'], '%Y-%m-%d %H:%M:%S'), row['plat'],
                                 row['plng'], 0)[0], axis=1)
    df['GWzen'] = df.apply(lambda row:
                           sunpos(datetime.datetime.strptime(row['GWavnTimestamp'], '%Y-%m-%d %H:%M:%S'), row['plat'],
                                  row['plng'], 0)[1], axis=1)
    df['sinGWAz'] = df['GWaz'].apply(lambda x: round(math.sin(x * cst), 10))
    df['cosGWAz'] = df['GWaz'].apply(lambda x: round(math.cos(x * cst), 10))
    df['sinGWZen'] = df['GWzen'].apply(lambda x: round(math.sin(x * cst), 10))
    df['cosGWZen'] = df['GWzen'].apply(lambda x: round(math.cos(x * cst), 10))
    return df


#########################
# 64
#########################
def xgBoost_classifier_all(x_train, y_train, x_test, y_test, max_depth_loc=14, learning_rate_loc=0.3,
                           n_estimators_loc=5, min_child_weight_loc=3, colsample_bytree_loc=0.8, sample_weight=False):
    """
    Train a Gradient Boosted Classifier Tree model and apply it to a test set.

    :param x_train:  A DataFrame.  The training set.
    :param y_train:  A DataFrame.  The training targets.
    :param x_test:  A DataFrame.  The test set.
    :param y_test:  A DatFrame.  The test targets.
    :param max_depth_loc:  Int.  The hyperparameter Max Depth.
    :param learning_rate_loc: Float. The hyperparameter learning rate.
    :param n_estimators_loc: Int.  The hyperparameter number of estimators.
    :param min_child_weight_loc: Int.  The hyperparameter Min Child Weight.
    :param colsample_bytree_loc: Float.  The hyperparameter column sampling.
    :param sample_weight: Boolean. Option to do sample weight or not during training.
    :return: Mean squared error of predictions, predictions, and the fitted model.
    """
    regr = xgb.XGBClassifier(max_depth=max_depth_loc,
                             learning_rate=learning_rate_loc,
                             n_estimators=n_estimators_loc,
                             nthread=8,
                             min_child_weight=min_child_weight_loc,
                             colsample_bytree=colsample_bytree_loc)
    if sample_weight:
        model = regr.fit(x_train, y_train, sample_weight=class_weight.compute_sample_weight('balanced', y_train))
    else:
        model = regr.fit(x_train, y_train)
    predictions = model.predict(x_test)
    # check_preds(predictions)
    acscore = accuracy_score(y_test, predictions)
    return acscore, predictions, model


#########################
# 65
#########################
def list_unit(data, unit="week"):
    """
        Return a list of unit in a dataframe and Add unit column into the dataframe.

        :param data:  A DataFrame.
        :param unit:  Str.  The way to split units: "day", "week", "month".
        :return: a list of units in the dataframe.
        """
    if unit == "day":
        unitag = "%j"
    elif unit == "week":
        unitag = "%W"
    elif unit == "month":
        unitag = "%m"
    dateunit = [time.strftime('%Y', time.strptime(d, '%Y-%m-%d %H:%M:%S')) + '-' +
                time.strftime(unitag, time.strptime(d, '%Y-%m-%d %H:%M:%S')) for d in data.timestamp]
    data[unit] = dateunit
    return sorted(set(dateunit))


#########################
# 66
#########################
def learning_curve_split(data, y, unit_loc="week"):
    """
    Return a list of tuples. each tuple contains train unit_loc index and test unit_loc index. The test has y unit_loc

    :param data: pandas DataFrame. input data
    :param y: Int. number of units data for testing. if >= total number of units is provided, (total-1) will be used
    :param unit_loc: Str. The way to split units: "day", "week", "month".
    :return: a list of tuples. each tuple contains the train unit_loc index and test unit_loc index
    """
    lu = list_unit(data, unit=unit_loc)
    llu = len(lu)
    yy = min(llu-1, y)
    if yy <= 0:
        print("\nThere are %d %s in input data. No Learning Curve can be done for %d %s" % (llu, unit_loc, y, unit_loc))
        return
    elif yy >= 1:
        idx_list = []
        for i in range(1, llu-yy+1):
            idx_list.append((lu[:i], lu[i: i+yy]))
        r = len(idx_list)
        print("\nThere are %d rounds Learning for testing on the following %d %s" % (r, yy, unit_loc))
        return idx_list


#########################
# 67
#########################
def model_compress_size(model, path="/tmp/model.pkl"):
    """
    Compress and save the model to path and return the size of it. Default path is in /tmp/ the model will be deleted
    automatically upon reboot. If want to save the model permanently, provide a path elsewhere.

    :param model: model. A machine learning model
    :param path: Str. A directory with model name
    :return: the size of model in KB.
    """
    joblib.dump(model, path, compress=9)
    size = os.path.getsize(path)
    # size_str = 'File size: {:.2f}'.format(size / 1024.0), 'KB'
    # if mode == "delete":
    #     os.remove(path)
    return size/1000.0


#########################
# 68
#########################
def cross_validation_xgb(data, predictors, targets, k, reps, xgb_depth=18, xgb_n=80):
    """
    Fit xgBoost regression model for k fold and reps repeats, return the average mse and accuracy scores.

    :param data: pandas DataFrame. The input data you want to use for cross validation
    :param predictors: List of str. predictors name set
    :param targets: Str. target name
    :param k: Int. number of fold
    :param reps: Int. number of repeats
    :param xgb_depth: Int. max_depth in xgb
    :param xgb_n: Int. n_estimators in xgb
    :return: 2 float. mean_mse, mean_accuracy
    """
    X_full = data[predictors]
    Y_full = data[targets]
    # time the model fitting
    start_time = time.time()
    # the stratification now requires a choice of one of the targets
    sindsLTset = [stratified_cross_validation_splits(Y_full, k) for rp in range(reps)]
    msevecrep = []
    acsvecrep = []
    for repnum in range(reps):
        msevecfold = []
        acsvecfold = []
        for ii in range(k):
            sindsLT = sindsLTset[repnum]
            testinds = sindsLT[ii]
            test_x = X_full.iloc[testinds]
            test_y = Y_full.iloc[testinds]
            # prepare the training set
            traininds = flatten(sindsLT[:ii] + sindsLT[ii + 1:])
            train_x = X_full.iloc[traininds]
            train_y = Y_full.iloc[traininds]
            # create regression model
            error, predictions, model = xgBoost_regression_all(train_x, train_y, test_x, test_y,
                                                               max_depth_loc=xgb_depth, n_estimators_loc=xgb_n)
            acs = accuracy_score(test_y, predictions.round())
            msevec = []
            msevec.append(error)
            acsvec = []
            acsvec.append(acs)
            msevecfold.append(msevec)
            acsvecfold.append(acsvec)
        msevecrep.append(msevecfold)
        acsvecrep.append(acsvecfold)
    # time the model fitting
    end_time = time.time()
    print ("\nTraining xgBoost_regression_all depth=%d and n=%d took %s seconds" % (xgb_depth, xgb_n, end_time-start_time))
    # evaluate the results
    msevecrepf = flatten(msevecrep)
    acsvecrepf = flatten(acsvecrep)
    msedf = pd.DataFrame({'MSE': flatten(msevecrepf), 'Accuracy': flatten(acsvecrepf)})
    mean_mse = msedf['MSE'].mean()
    mean_accuracy = msedf['Accuracy'].mean()
    print("\nMean:")
    print(msedf.mean())
    print("\nStd:")
    print(msedf.std())
    print("\n%s Full Scores:" % targets)
    print(msedf)
    return mean_mse, mean_accuracy


#########################
# 69
#########################
def cross_validation_rf(data, predictors, targets, k, reps, rf_depth=18, rf_n=80):
    """
    Fit randomForest regression model for k fold and reps repeats, return the average mse and accuracy scores.

    :param data: pandas DataFrame. The input data you want to use for cross validation
    :param predictors: List of str. predictors name set
    :param targets: Str. target name
    :param k: Int. number of fold
    :param reps: Int. number of repeats
    :param xgb_depth: Int. max_depth in xgb
    :param xgb_n: Int. n_estimators in xgb
    :return: 2 float. mean_mse, mean_accuracy
    """
    X_full = data[predictors]
    Y_full = data[targets]
    # time the model fitting
    start_time = time.time()
    # the stratification now requires a choice of one of the targets
    sindsLTset = [stratified_cross_validation_splits(Y_full, k) for rp in range(reps)]
    msevecrep = []
    acsvecrep = []
    for repnum in range(reps):
        msevecfold = []
        acsvecfold = []
        for ii in range(k):
            sindsLT = sindsLTset[repnum]
            testinds = sindsLT[ii]
            test_x = X_full.iloc[testinds]
            test_y = Y_full.iloc[testinds]
            # prepare the training set
            traininds = flatten(sindsLT[:ii] + sindsLT[ii + 1:])
            train_x = X_full.iloc[traininds]
            train_y = Y_full.iloc[traininds]
            # create regression model
            error, predictions, model = randomforest_regression_all(train_x, train_y, test_x, test_y,
                                                                    max_depth_loc=rf_depth, n_estimators_loc=rf_n)
            acs = accuracy_score(test_y, predictions.round())
            msevec = []
            msevec.append(error)
            acsvec = []
            acsvec.append(acs)
            msevecfold.append(msevec)
            acsvecfold.append(acsvec)
        msevecrep.append(msevecfold)
        acsvecrep.append(acsvecfold)
    # time the model fitting
    end_time = time.time()
    print ("\nTraining randomforest_regression_all depth=%d and n=%d took %s seconds" % (rf_depth, rf_n, end_time-start_time))
    # evaluate the results
    msevecrepf = flatten(msevecrep)
    acsvecrepf = flatten(acsvecrep)
    msedf = pd.DataFrame({'MSE': flatten(msevecrepf), 'Accuracy': flatten(acsvecrepf)})
    mean_mse = msedf['MSE'].mean()
    mean_accuracy = msedf['Accuracy'].mean()
    print("\nMean:")
    print(msedf.mean())
    print("\nStd:")
    print(msedf.std())
    print("\n%s Full Scores:" % targets)
    print(msedf)
    return mean_mse, mean_accuracy


#########################
# 70
#########################
def list_files_endwith(address, endwithstr="logs.csv"):
    """
    List all files end with "logs.csv" in a directory.

    :param address: Str. An absolute path
    :param endwithstr: Str. The condition: file must end with this Str.
    :return: A list of Strings. Each string is a complete path and file name.
    """
    file_Paths = []
    for root, dirs, files in os.walk(address):
        if len(files):
            for file in files:
                if file.endswith(endwithstr):
                    f = root + '/' + file
                    file_Paths.append(f)
    return file_Paths


#########################
# 71
#########################
def score_prediction(true_value, prediction, delta):
    """
    Return a Boolean value of whether or not the prediction meet the delta difference criteria.

    :param true_value: Int or Float. The true value of the target variable
    :param prediction: Int or Float. The prediction of the target variable
    :param delta: Int or Float. The difference buffer allowed for this criteria
    :return: A Boolean value of whether or not the prediction meet the delta difference criteria.
    """
    return true_value - delta <= prediction <= true_value + delta


#########################
# 72
#########################
def backward_learning_curve_split(data, y, unit_loc="week"):
    """
    Return a list of tuples. each tuple contains train unit_loc index and test unit_loc index. The test has y unit_loc

    :param data: pandas DataFrame. input data
    :param y: Int. number of units data for testing. if >= total number of units is provided, (total-1) will be used
    :param unit_loc: Str. The way to split units: "day", "week", "month".
    :return: a list of tuples. each tuple contains the train unit_loc index and test unit_loc index
    """
    lu = list_unit(data, unit=unit_loc)
    llu = len(lu)
    yy = min(llu-1, y)
    if yy <= 0:
        print("\nThere are %d %s in input data. No Learning Curve can be done for %d %s" % (llu, unit_loc, y, unit_loc))
        return
    elif yy >= 1:
        idx_list = []
        for i in range(1, llu-yy+1):
            idx_list.append((lu[(llu-yy-i):(llu-yy)], lu[(llu-yy):]))
        r = len(idx_list)
        print("\nThere are %d rounds Learning for testing on the last %d %s" % (r, yy, unit_loc))
        return idx_list


#########################################################
# These are the aliases that will be needed to make sure
# that code relying on the old function names continues to work
#########################################################
showStringDiff = show_string_diff
flat_list = flatten
getTrainTest = get_train_test
sss = stratified_cross_validation_splits
linear_regression_All = linear_regression_all
svm_regression_All = svm_regression_all
decisionTree_regression_All = decision_tree_regression_all
xgBoost_regression_All = xgBoost_regression_all
mlp_regression_All = multi_layer_perceptron_regression_all
TF_mlp_regression_All = tensor_flow_regression_all
# LGB_regression_All = light_gbm_regression_all
makeFV = make_lagged_feature_vector
makeLagPredTar = make_lagged_predictors_and_targets
makeLagTarOnly = make_lagged_targets_only
plot_LearingCurve = plot_learing_curve
normalizeColumn = normalize_column
FitAndTestRegression = fit_and_test_regression
balanceData = balance_data
FilterFeatures = filter_features
fsloopsfromdict = feature_selection_loops_from_dictionary
AssemblePreds = assemble_predictions
CalcMSEs = calculate_mean_squared_errors
CalcMSE_mean_std = calculate_mean_and_std_of_mse
ParseMeanStdMSE = parse_mean_and_std_of_mse
#ParseMLpipelineResults = parse_machine_learning_pipeline_results
mergeFullPreds = merge_full_predictions
ApplyThreshold = apply_threshold
PredictionFeedback = prediction_feedback
PredictionFeedbackStacked = prediction_feedback_stacked
PredictionFeedbackStackedSimultaneous = prediction_feedback_stacked_simultaneous
PredictionFeedbackStackedSimultaneous_preExistingGMPreds = prediction_feedback_stacked_simultaneous_preexisting_gm_predictions
CheckTimestampOrder = check_timestamp_order
FullPlot = full_plot
FullPlot_ax = full_plot_axis
FeaturePlot_ax = feature_plot_axis
# ChenHuiScoring = chen_hui_scoring