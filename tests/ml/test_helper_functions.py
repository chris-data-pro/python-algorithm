import os
import sys
import math
import random
import sklearn
import xgboost
import unittest
import StringIO
from helper_functions import *
from nose.tools import assert_dict_equal
from sklearn.tree import DecisionTreeRegressor
from pandas.util.testing import assert_frame_equal
import warnings
warnings.filterwarnings(action="ignore") # , module="scipy", message="^internal gelsd")


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        self.expected = [[1], [2], [3]]
        self.result = [1, 2, 3]
        self.a = 'add'
        self.b = 'and'
        self.list = ['add => and', '\n', u'Add "n" to position 1', '\n', u'Delete "d" from position 3', '\n']
        self.directory_path = './test_check_path'
        # test_getTrainTest
        self.data = pd.DataFrame([[1, 2], [4, 5]])
        self.targets = pd.DataFrame([[10, 20], [40, 50]])
        self.testinds = [1]
        self.result_train_x = pd.DataFrame([[1, 2]])
        self.result_train_y = pd.DataFrame([[10, 20]])
        self.result_test_x = pd.DataFrame([[4, 5]])
        self.result_test_y = pd.DataFrame([[40, 50]])
        # test_sss
        self.k = 1
        self.yy_raise = 1
        self.yy = [[1, 2, 3], [4, 5, 6]]
        self.outinds = [[[0, 1, 2], [0, 1, 2]]]
        # test_makeFV & test_makeLagPredTar & test_makeLagTarOnly & test_normalizeColumn
        self.numlag = 1
        self.rid = 2
        self.df = pd.DataFrame([[0, 2], [0, 4], [10, 20]], index=[0, 1, 2], columns=['A', 'B'])
        self.LFloc = ['A']
        self.NLFloc = ['B']
        self.Tarname = 'A'
        self.rnfact = 4
        self.featvec_result = [20, 10]
        self.featnames_result = ['B', 'A_lag_0']
        self.DFpredictors_result = pd.DataFrame([[4, 0], [20, 10]], index=[1, 2],
                                                columns=['B', 'A_lag_0'], dtype=np.object)
        self.DFtargets_result = pd.DataFrame([[0], [10]], index=[1, 2], columns=['A'])
        self.normalizeColumn = pd.DataFrame([[0.0, 2], [0.0, 4], [2.5, 20]], index=[0, 1, 2], columns=['A', 'B'])
        self.FSparamdict_h = {'do_FS': [False, True], 'FSmethod': 'a', 'NSF': 'c'}
        self.FSparamsLooplist_h = [{'do_FS': False, 'NSF': 0, 'FSmethod': 'NoFS'},
                                   {'do_FS': True, 'NSF': 'c', 'FSmethod': 'a'}]
        # test_mergeFullPreds
        self.fpd1_in = {'SelectedFeatures': {'a': 1}, 'FullPredictions': {'b': 2}}
        self.fpd2 = {'SelectedFeatures': {'c': 3}, 'FullPredictions': {'d': 4}}
        self.fpd1 = {'FullPredictions': {'b': 2, 'd': 4}, 'SelectedFeatures': {'a': 1, 'c': 3}}
        # test_PredictionFeedback & test_PredictionFeedback2
        self.dat_a = pd.DataFrame([[4, 0], [20, 10]], columns=['A_lag_1', 'A_lag_2'])
        self.regressor_a = DecisionTreeRegressor(max_depth=2)
        self.mod_a = self.regressor_a.fit(self.dat_a, [4, 5])
        self.Target_loc_a = 'A_abc'
        self.numlags_a = 3
        self.PredictionsList_1 = [4.0, 4.0]
        self.Roundlist_a = [5, 6]
        self.Threshlist_a = [2.5, 3.5, 4.5]
        self.PredictionsList_2 = [3.5, 3.5]
        self.seed_loc_a = 5.0
        self.PredictionsList_3 = [4.0, 4.0]
        # test_nearest_round & test_ApplyThreshold
        self.myList = [2, 3, 4]
        self.myNumber = 3
        self.nearest_round = 3
        self.dat_time = pd.DataFrame([[1381419670, 4, 0], [1381419650, 20, 10]],
                                     columns=['timestamp', 'A_lag_1', 'A_lag_2'])
        self.df1 = pd.DataFrame({'x1': [1, 2, 3, 4], 'y': [random.random() for x in range(4)]})
        self.df1['x2'] = self.df1['y'].apply(lambda x: math.cos(x))
        self.expected_ranking = [1, 0]

        self.x_train = pd.DataFrame([[1, 10], [2, 10]])
        self.y_train = pd.DataFrame([[11], [12]])
        self.x_test = pd.DataFrame([[4, 10], [5, 10]])
        self.y_test = pd.DataFrame([[14], [15]])
        # test_linear_regression_all
        self.use_poly_l = True
        self.use_ridge_l = True
        self.error_1 = 0.052613424235046106
        self.predictions_1 = np.array([[14.13963964], [15.29279279]])
        self.model_1 = sklearn.linear_model.ridge.Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
                                                      normalize=False, random_state=None, solver='auto', tol=0.001)
        # test_svm_regression_all
        self.Cloc = 1e2
        self.gammaloc = 0.2
        self.epsilonloc = 0.2
        self.error_2 = 7.48517303185265
        self.predictions_2 = np.array([11.970069, 11.706108])
        self.model_2 = sklearn.svm.classes.SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma=0.2,
                                               kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        # test_decision_tree_regression_all
        self.max_depth_loc = 6
        self.error_3 = 6.5
        self.predictions_3 = np.array([12., 12.])
        self.model_3 = sklearn.tree.tree.DecisionTreeRegressor(criterion='mse', max_depth=6, max_features=None,
                                                               max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                               min_impurity_split=None, min_samples_leaf=1,
                                                               min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                               presort=False, random_state=None, splitter='best')
        # test_xgBoost_regression_all
        self.min_child_weight_loc = 2
        self.error_4 = 9.316676152434411
        self.predictions_4 = np.array([11.488908, 11.488908], dtype=np.float32)
        self.model_4 = xgboost.sklearn.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                                    colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                                                    max_depth=6, min_child_weight=2, missing=None, n_estimators=100,
                                                    n_jobs=1, nthread=8, objective='reg:linear', random_state=0,
                                                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                                                    silent=True, subsample=1)


        # test_multi_layer_perceptron_regression_all
        # self.error_5 = 9.493073429757136
        # self.predictions_5 = np.array([11.446426, 11.447509])
        self.model_5 = sklearn.neural_network.multilayer_perceptron.MLPRegressor(activation='logistic', alpha=0.01,
                                                                                 batch_size='auto', beta_1=0.9,
                                                                                 beta_2=0.999, early_stopping=False,
                                                                                 epsilon=1e-08,
                                                                                 hidden_layer_sizes=(5,),
                                                                                 learning_rate='adaptive',
                                                                                 learning_rate_init=0.01, max_iter=1000,
                                                                                 momentum=0.9,
                                                                                 nesterovs_momentum=True, power_t=0.5,
                                                                                 random_state=None,
                                                                                 shuffle=True, solver='adam',
                                                                                 tol=0.0001, validation_fraction=0.1,
                                                                                 verbose=False, warm_start=False)
        # test_tensor_flow_regression_all
        # undefined functions
        # test_fit_and_test_regression
        self.modtype_h_1 = 'linearRegression'
        self.modtype_h_2 = 'SVR'
        self.modtype_h_3 = 'DT'
        self.modtype_h_4 = 'GBT'
        self.modtype_h_5 = 'MLP'
        self.modtype_h_6 = 'TF_MLP'
        self.MLparams_h = {'use_polynomial': True, 'use_ridgeregression': True, 'C': 1e2, 'gamma': 0.2, 'epsilon': 0.2,
                           'max_depth': 6, 'min_child_weight': 2}

        # test_balance_data
        self.train_x_h = pd.DataFrame([[4, 0], [20, 10]], columns=['A', 'A_lag_1'])
        self.train_y_h = pd.Series([9, 10])
        self.cntar_h = 'A'
        self.train_x_f = pd.DataFrame([[20, 10], [4, 0]], index=[1, 0], columns=['A', 'A_lag_1'])
        self.train_y_f = pd.Series([10, 9], index=[1, 0])
        self.nki = [1, 0]

        # test_rank_features
        self.X = pd.DataFrame([[1, 10], [2, 10], [4, 10], [5, 10]], dtype=np.float)
        # self.Y = pd.DataFrame([[11], [12], [14], [15]], dtype=np.float)
        self.Y = pd.Series([11, 12, 14, 15], dtype=np.float)
        self.method = 'MI2'
        self.FI_orderh =[0, 1]
        # test_filter_features
        self.test_x_h = pd.DataFrame([[6, 7], [8, 9]], dtype=np.float)
        self.FSparams_h = {'do_FS': True, 'FSmethod': 'MI2', 'NSF': 1}
        self.train_x_f_1 = pd.DataFrame([[1.0], [2.0], [4.0], [5.0]])
        self.test_x_f = pd.DataFrame([[6.0], [8.0]])
        self.SelectedFeatures_loc = [0]
        # test_prediction_feedback_stacked_simultaneous
        self.dat = pd.DataFrame([[14, 9], [20, 10]], columns=['A_lag_1', 'A_lag_2'])
        regressor = DecisionTreeRegressor(max_depth=6)
        self.mod_gm = regressor.fit(self.dat, [4, 5])
        clf = SVR(kernel='linear')
        self.mod_pm= clf.fit(self.dat, [4, 5])
        self.colsforGm=['A_lag_1', 'A_lag_2']
        self.colsforPm=['A_lag_1', 'A_lag_2']
        self.Target_loc = 'A_lag'
        self.numlags = 2
        self.PredictionsListStacked = [1, 1]
        self.PredictionsListGM = [1.0, 1.0]
        self.PredictionsListPM = [4.1, 2.435135135135134]
        # self.PredictionsListGM = pd.Series([1.0, 1.0])

        # test_assemble_predictions
        self.FullPredsDict_l_1 = {'FullPredictionsKeyMeaning': ["Fold", "Rep"],
                                  'FullPredictions': {"(0, 0)": [-1, 2, 1], "(0, 1)": [2, -3, 1], "(1, 0)": [6, -2, 4],
                                                      "(1, 1)": [2, -2, 3], "(2, 0)": [1, -1, 1], "(2, 1)": [2, -2, 2]},
                                  'Target': [0.0, 0.0, 0.0],
                                  'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                                                     [[0, 1, 2], [0, 1, 2], [0, 1, 2]]],
                                  'K': 3}
        self.result_FullPredsDict_1 = "{'FullPredictionsKeyMeaning': ['Fold', 'Rep'], 'Target': [0.0, 0.0, 0.0], 'FullPredictions': {'(0, 1)': [2, -3, 1], '(0, 0)': [-1, 2, 1], '(1, 0)': [6, -2, 4], '(1, 1)': [2, -2, 3], '(2, 1)': [2, -2, 2], '(2, 0)': [1, -1, 1]}, 'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]], 'K': 3, 'FullPredictionsAssembledKeyMeaning': ['Rep'], 'FullPredictionsAssembled': {(0,): [1.0, -1.0, 1.0], (1,): [2.0, -2.0, 2.0]}}"
        # test_calculate_mean_squared_errors
        self.FullPredsDict_l_2 = {'FullPredictionsKeyMeaning': ['Fold', 'Rep'],
                                  'Target': [0.0, 0.0, 0.0],
                                  'FullPredictions': {'(0, 1)': [2, -3, 1], '(0, 0)': [-1, 2, 1],
                                                      '(1, 0)': [6, -2, 4], '(1, 1)': [2, -2, 3],
                                                      '(2, 1)': [2, -2, 2], '(2, 0)': [1, -1, 1]},
                                  'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                                                     [[0, 1, 2], [0, 1, 2], [0, 1, 2]]],
                                  'K': 3,
                                  'FullPredictionsAssembledKeyMeaning': ['Rep'],
                                  'FullPredictionsAssembled': {(0,): np.array([1., -1., 1.]),
                                                               (1,): np.array([2., -2., 2.])}}
        self.result_FullPredsDict_2 = "{'FullPredictionsKeyMeaning': ['Fold', 'Rep'], 'Target': [0.0, 0.0, 0.0], 'FullPredictions': {'(0, 1)': [2, -3, 1], '(0, 0)': [-1, 2, 1], '(1, 0)': [6, -2, 4], '(1, 1)': [2, -2, 3], '(2, 1)': [2, -2, 2], '(2, 0)': [1, -1, 1]}, 'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]], 'K': 3, 'FullPredictionsAssembledKeyMeaning': ['Rep'], 'MSE': {(0,): 1.0, (1,): 4.0}, 'FullPredictionsAssembled': {(0,): array([ 1., -1.,  1.]), (1,): array([ 2., -2.,  2.])}}"
        # test_calculate_mean_and_std_of_mse
        self.FullPredsDict_l_3 = {'FullPredictionsKeyMeaning': ['Fold', 'Rep'],
                                  'Target': [0.0, 0.0, 0.0],
                                  'FullPredictions': {'(0, 1)': [2, -3, 1], '(0, 0)': [-1, 2, 1],
                                                      '(1, 0)': [6, -2, 4], '(1, 1)': [2, -2, 3],
                                                      '(2, 1)': [2, -2, 2], '(2, 0)': [1, -1, 1]},
                                  'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
                                                     [[0, 1, 2], [0, 1, 2], [0, 1, 2]]],
                                  'K': 3,
                                  'Reps': 2,
                                  'FullPredictionsAssembledKeyMeaning': ['Rep'],
                                  'MSE': {(0,): 1.0, (1,): 4.0},
                                  'FullPredictionsAssembled': {(0,): np.array([1., -1., 1.]),
                                                               (1,): np.array([2., -2., 2.])}}
        self.result_FullPredsDict_3 = "{'FullPredictionsKeyMeaning': ['Fold', 'Rep'], 'Target': [0.0, 0.0, 0.0], 'FullPredictions': {'(0, 1)': [2, -3, 1], '(0, 0)': [-1, 2, 1], '(1, 0)': [6, -2, 4], '(1, 1)': [2, -2, 3], '(2, 1)': [2, -2, 2], '(2, 0)': [1, -1, 1]}, 'K': 3, 'MSE_mean': {(): 2.5}, 'MSE': {(0,): 1.0, (1,): 4.0}, 'MSE_std': {(): 1.5}, 'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]], 'Reps': 2, 'FullPredictionsAssembledKeyMeaning': ['Rep'], 'MSE_mean_std_KeyMeaning': [], 'FullPredictionsAssembled': {(0,): array([ 1., -1.,  1.]), (1,): array([ 2., -2.,  2.])}}"
        # test_parse_mean_and_std_of_mse
        self.FullPredsDict_l_4 = {
            'FullPredictionsKeyMeaning': ['Fold', 'Rep', 'do_FS', 'NSF', 'FSmethod', 'NumFS', 'n_estimators',
                                          'colsmaple_bytree'],
            'Target': [0.0, 0.0, 0.0],
            'FullPredictions': {'(0, 1, False, 15, "NoFS", 0, 120, 0.8)': [2, -3, 1],
                                '(0, 0, False, 15, "NoFS", 0, 120, 0.8)': [-1, 2, 1],
                                '(1, 0, False, 15, "NoFS", 0, 120, 0.8)': [6, -2, 4],
                                '(1, 1, False, 15, "NoFS", 0, 120, 0.8)': [2, -2, 3],
                                '(2, 1, False, 15, "NoFS", 0, 120, 0.8)': [2, -2, 2],
                                '(2, 0, False, 15, "NoFS", 0, 120, 0.8)': [1, -1, 1]},
            'K': 3,
            'MSE_mean': {(False, 15, 'NoFS', 0, 120, 0.8): 2.5},
            'MSE': {(0, False, 15, 'NoFS', 0, 120, 0.8): 1.0,
                    (1, False, 15, 'NoFS', 0, 120, 0.8): 4.0},
            'MSE_std': {(False, 15, 'NoFS', 0, 120, 0.8): 1.5},
            'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]],
            'modtype': 'GBT',
            'Reps': 2,
            'modelParameters': {'n_estimators': [120], 'colsmaple_bytree': [0.8]},
            'FSParameters': {'do_FS': [False], 'NSF': [15], 'FSmethod': ['MI'], 'NumFS': [0]},
            'FullPredictionsAssembledKeyMeaning': ['Rep'],
            'MSE_mean_std_KeyMeaning': ['do_FS', 'NSF', 'FSmethod', 'NumFS', 'n_estimators', 'colsmaple_bytree'],
            'FullPredictionsAssembled': {(0,): np.array([1., -1., 1.]), (1,): np.array([2., -2., 2.])}}
        self.result_FullPredsDict_4 = "{'FullPredictionsKeyMeaning': ['Fold', 'Rep', 'do_FS', 'NSF', 'FSmethod', 'NumFS', 'n_estimators', 'colsmaple_bytree'], 'ResultsByMethod_MLxFS_std': {'GBT_NoFS': array([[ 1.5]])}, 'Target': [0.0, 0.0, 0.0], 'FullPredictions': {'(0, 0, False, 15, \"NoFS\", 0, 120, 0.8)': [-1, 2, 1], '(2, 1, False, 15, \"NoFS\", 0, 120, 0.8)': [2, -2, 2], '(1, 0, False, 15, \"NoFS\", 0, 120, 0.8)': [6, -2, 4], '(0, 1, False, 15, \"NoFS\", 0, 120, 0.8)': [2, -3, 1], '(1, 1, False, 15, \"NoFS\", 0, 120, 0.8)': [2, -2, 3], '(2, 0, False, 15, \"NoFS\", 0, 120, 0.8)': [1, -1, 1]}, 'K': 3, 'MSE_mean': {(False, 15, 'NoFS', 0, 120, 0.8): 2.5}, 'modtype': 'GBT', 'modelParameters': {'n_estimators': [120], 'colsmaple_bytree': [0.8]}, 'MSE': {(0, False, 15, 'NoFS', 0, 120, 0.8): 1.0, (1, False, 15, 'NoFS', 0, 120, 0.8): 4.0}, 'ResultsByMethod_MLxFS_mean': {'GBT_NoFS': array([[ 2.5]])}, 'MSE_std': {(False, 15, 'NoFS', 0, 120, 0.8): 1.5}, 'RepFoldIndices': [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]], 'Reps': 2, 'FullPredictionsAssembledKeyMeaning': ['Rep'], 'FSParameters': {'do_FS': [False], 'NumFS': [0], 'FSmethod': ['MI'], 'NSF': [15]}, 'FullPredictionsAssembled': {(0,): array([ 1., -1.,  1.]), (1,): array([ 2., -2.,  2.])}, 'MSE_mean_std_KeyMeaning': ['do_FS', 'NSF', 'FSmethod', 'NumFS', 'n_estimators', 'colsmaple_bytree']}"

    #########################
    # 1
    #########################
    def test_table(self):
        series_example = pd.Series(1)
        test_series = pd.DataFrame(pd.Series(Counter(pd.Series(1)), name='Counts').sort_values(ascending=False))
        # assert_dict_equal(test_series.to_dict(), table(series_example).to_dict(), "BUG - if isinstance(S2, pd.core.series.Series):")
        assert_frame_equal(test_series, table(series_example))
        list_example = [1]
        test_list = pd.DataFrame(pd.Series(Counter([1]), name='Counts').sort_values(ascending=False))
        # assert_dict_equal(test_list.to_dict(), table(list_example).to_dict(), "BUG - elif isinstance(S2, list):")
        assert_frame_equal(test_list, table(list_example))
        with self.assertRaises(ValueError):
            table(1)

    #########################
    # 2
    #########################
    def test_flatten(self):
        self.assertItemsEqual(self.result, flatten(self.expected))

    #########################
    # 3
    #########################
    def test_show_string_diff(self):
        capturedOutput = StringIO.StringIO()
        sys.stdout = capturedOutput
        show_string_diff(self.a, self.b)
        # sys.stdout = sys.__stdout__  # Reset redirect.
        target = capturedOutput.buflist
        self.assertListEqual(self.list, target)

    #########################
    # 4
    #########################
    def test_check_make_path(self):
        check_make_path(self.directory_path)
        self.assertTrue(os.path.exists(self.directory_path))

    #########################
    # 5
    #########################
    def test_get_train_test(self):
        train_x, train_y, test_x, test_y = get_train_test(self.data, self.targets, self.testinds)
        assert_frame_equal(train_x, self.result_train_x)
        assert_frame_equal(train_y, self.result_train_y)
        assert_frame_equal(test_x, self.result_test_x)
        assert_frame_equal(test_y, self.result_test_y)

    #########################
    # 6
    #########################
    def test_stratified_cross_validation_splits(self):
        with self.assertRaises(ValueError):
            stratified_cross_validation_splits(self.yy_raise, self.k)
        self.assertListEqual(stratified_cross_validation_splits(self.yy, self.k), self.outinds)

    #########################
    # 7
    #########################
    def test_linear_regression_all(self):
        result_error, result_predictions, result_model = linear_regression_all(self.x_train, self.y_train, self.x_test,
                                                          self.y_test, self.use_poly_l, self.use_ridge_l)
        self.assertEqual(self.error_1, result_error)
        np.testing.assert_allclose(self.predictions_1, result_predictions)
        self.assertEqual(str(self.model_1), str(result_model))

    #########################
    # 8
    #########################
    def test_svm_regression_all(self):
        result_error, result_predictions, result_model = svm_regression_all(self.x_train, self.y_train, self.x_test,
                                                                            self.y_test, self.Cloc, self.gammaloc,
                                                                            self.epsilonloc)
        self.assertEqual(self.error_2, result_error)
        np.testing.assert_allclose(self.predictions_2, result_predictions)
        self.assertEqual(str(self.model_2), str(result_model))

    #########################
    # 9
    #########################
    def test_decision_tree_regression_all(self):
        result_error, result_predictions, result_model = decision_tree_regression_all(self.x_train, self.y_train,
                                                                                      self.x_test, self.y_test,
                                                                                      self.max_depth_loc)
        self.assertEqual(self.error_3, result_error)
        np.testing.assert_allclose(self.predictions_3, result_predictions)
        self.assertEqual(str(self.model_3), str(result_model))

    #########################
    # 10
    #########################
    def test_xgBoost_regression_all(self):
        result_error, result_predictions, result_model = xgBoost_regression_all(self.x_train, self.y_train,
                                                                                self.x_test, self.y_test,
                                                                                self.max_depth_loc,
                                                                                self.min_child_weight_loc)
        self.assertEqual(self.error_4, result_error)
        np.testing.assert_allclose(self.predictions_4, result_predictions)
        self.assertEqual(str(self.model_4), str(result_model))

    #########################
    # 11
    #########################
    def test_multi_layer_perceptron_regression_all(self):
        result_error, result_predictions, result_model = multi_layer_perceptron_regression_all(self.x_train,
                                                                                               self.y_train,
                                                                                               self.x_test,
                                                                                               self.y_test)
        # self.assertEqual(self.error_5, result_error)
        self.assertIsInstance(result_error, np.float64)
        # np.testing.assert_allclose(self.predictions_5, result_predictions)
        self.assertIsInstance(result_predictions, np.ndarray)
        self.assertEqual(str(self.model_5), str(result_model))

    #########################
    # 12
    #########################
    def test_tensor_flow_regression_all(self):
        pass

    #########################
    # 16
    #########################
    def test_make_lagged_feature_vector(self):
        featvec, featnames = make_lagged_feature_vector(self.numlag, self.rid, self.df, self.LFloc, self.NLFloc)
        self.assertListEqual(featvec, self.featvec_result)
        self.assertListEqual(featnames, self.featnames_result)

    #########################
    # 17
    #########################
    def test_make_lagged_predictors_and_targets(self):
        DFpredictors, DFtargets = make_lagged_predictors_and_targets(self.numlag, self.LFloc, self.NLFloc, self.df, self.Tarname)
        assert_frame_equal(DFpredictors, self.DFpredictors_result)
        assert_frame_equal(DFtargets, self.DFtargets_result)

    #########################
    # 18
    #########################
    def test_make_lagged_targets_only(self):
        assert_frame_equal(make_lagged_targets_only(self.numlag, self.df, self.Tarname), self.DFtargets_result)

    #########################
    # 20
    #########################
    def test_rank_features(self):
        self.assertListEqual(rank_features(self.df1[['x1', 'x2']], self.df1['y'], method='MI', continuous=True),
                             self.expected_ranking)
        result_FI_orderh = rank_features(self.X, self.Y, self.method)
        self.assertListEqual(self.FI_orderh, result_FI_orderh)

    #########################
    # 21
    #########################
    def test_normalize_column(self):
        assert_frame_equal(normalize_column(self.df, self.Tarname, self.rnfact), self.normalizeColumn)

    #########################
    # 22
    #########################
    def test_fit_and_test_regression(self):
        result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
                                                                                 self.x_test, self.y_test,
                                                                                 self.modtype_h_1, self.MLparams_h)
        self.assertEqual(self.error_1, result_error)
        np.testing.assert_allclose(self.predictions_1, result_predictions)
        self.assertEqual(str(self.model_1), str(result_model))

        result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
                                                                                 self.x_test, self.y_test,
                                                                                 self.modtype_h_2, self.MLparams_h)
        self.assertEqual(self.error_2, result_error)
        np.testing.assert_allclose(self.predictions_2, result_predictions)
        self.assertEqual(str(self.model_2), str(result_model))

        result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
                                                                                 self.x_test, self.y_test,
                                                                                 self.modtype_h_3, self.MLparams_h)
        self.assertEqual(self.error_3, result_error)
        np.testing.assert_allclose(self.predictions_3, result_predictions)
        self.assertEqual(str(self.model_3), str(result_model))

        result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
                                                                                 self.x_test, self.y_test,
                                                                                 self.modtype_h_4, self.MLparams_h)
        self.assertEqual(self.error_4, result_error)
        np.testing.assert_allclose(self.predictions_4, result_predictions)
        self.assertEqual(str(self.model_4), str(result_model))

        result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
                                                                                 self.x_test, self.y_test,
                                                                                 self.modtype_h_5, self.MLparams_h)
        # self.assertEqual(self.error_5, result_error)
        self.assertIsInstance(result_error, np.float64)
        # np.testing.assert_allclose(self.predictions_5, result_predictions)
        self.assertIsInstance(result_predictions, np.ndarray)
        self.assertEqual(str(self.model_5), str(result_model))

        # result_error, result_predictions, result_model = fit_and_test_regression(self.x_train, self.y_train,
        #                                                                          self.x_test, self.y_test,
        #                                                                          self.modtype_h_6, self.MLparams_h)
        # self.assertEqual(self.error_6, result_error)
        # np.testing.assert_allclose(self.predictions_6, result_predictions)
        # self.assertEqual(str(self.model_6), str(result_model))

    #########################
    # 23
    #########################
    def test_balance_data(self):
        result_train_x_f, result_train_y_f, result_nki = balance_data(self.train_x_h, self.train_y_h, self.cntar_h)
        pd.util.testing.assert_frame_equal(self.train_x_f, result_train_x_f)
        pd.util.testing.assert_series_equal(self.train_y_f, result_train_y_f)
        self.assertListEqual(self.nki, result_nki)

    #########################
    # 24
    #########################
    def test_filter_features(self):
        result_train_x_f, result_test_x_f, result_SelectedFeatures_loc = filter_features(self.X, self.Y,
                                                                                         self.test_x_h, self.FSparams_h)
        pd.util.testing.assert_frame_equal(self.train_x_f_1, result_train_x_f)
        pd.util.testing.assert_frame_equal(self.test_x_f, result_test_x_f)
        self.assertListEqual(self.SelectedFeatures_loc, result_SelectedFeatures_loc)

    #########################
    # 25
    #########################
    def test_feature_selection_loops_from_dictionary(self):
        result = feature_selection_loops_from_dictionary(self.FSparamdict_h)
        self.assertListEqual(result, self.FSparamsLooplist_h)

    #########################
    # 26
    #########################
    def test_assemble_predictions(self):
        result_FullPredsDict_l = assemble_predictions(self.FullPredsDict_l_1)
        self.assertEqual(self.result_FullPredsDict_1, str(result_FullPredsDict_l))

    #########################
    # 27
    #########################
    def test_calculate_mean_squared_errors(self):
        result_FullPredsDict_l = calculate_mean_squared_errors(self.FullPredsDict_l_2)
        self.assertEqual(self.result_FullPredsDict_2, str(result_FullPredsDict_l))

    #########################
    # 28
    #########################
    def test_calculate_mean_and_std_of_mse(self):
        result_FullPredsDict_l = calculate_mean_and_std_of_mse(self.FullPredsDict_l_3)
        self.assertEqual(self.result_FullPredsDict_3, str(result_FullPredsDict_l))

    #########################
    # 29
    #########################
    def test_parse_mean_and_std_of_mse(self):
        result_FullPredsDict_l = parse_mean_and_std_of_mse(self.FullPredsDict_l_4)
        self.assertEqual(self.result_FullPredsDict_4, str(result_FullPredsDict_l))

    #########################
    # 30
    #########################
    def test_parse_machine_learning_pipeline_results(self):
        pass

    #########################
    # 31
    #########################
    def test_merge_full_predictions(self):
        result = merge_full_predictions(self.fpd1_in, self.fpd2)
        self.assertDictEqual(result, self.fpd1)

    #########################
    # 32
    #########################
    def test_apply_threshold(self):
        result = apply_threshold(self.myList, self.myNumber)
        self.assertEqual(result, self.nearest_round)

    #########################
    # 33
    #########################
    def test_prediction_feedback(self):
        result1 = prediction_feedback(self.mod_a, self.dat_a, self.Target_loc_a, self.numlags_a)
        self.assertListEqual(result1, self.PredictionsList_1)
        result2 = prediction_feedback(self.mod_a, self.dat_a, self.Target_loc_a, self.numlags_a, self.Roundlist_a, self.Threshlist_a)
        self.assertListEqual(result2, self.PredictionsList_2)

    #########################
    # 34
    #########################
    def test_prediction_feedback_stacked(self):
        result = prediction_feedback_stacked(self.mod_a, self.dat_a, self.Target_loc_a, self.numlags_a, self.seed_loc_a)
        self.assertListEqual(result, self.PredictionsList_3)

    #########################
    # 35
    #########################
    def test_prediction_feedback_stacked_simultaneous(self):
        result_PredictionsListStacked, result_PredictionsListGM, result_PredictionsListPM = \
            prediction_feedback_stacked_simultaneous(self.mod_gm,
                                                     self.mod_pm,
                                                     self.colsforGm,
                                                     self.colsforPm,
                                                     self.dat,
                                                     self.Target_loc,
                                                     self.numlags)
        self.assertListEqual(self.PredictionsListStacked, result_PredictionsListStacked)
        self.assertListEqual(self.PredictionsListGM, result_PredictionsListGM)
        self.assertListEqual(self.PredictionsListPM, result_PredictionsListPM)

    #########################
    # 36
    #########################
    def test_prediction_feedback_stacked_simultaneous_preexisting_gm_predictions(self):
        result_PredictionsListStacked, result_PredictionsListGM, result_PredictionsListPM = \
            prediction_feedback_stacked_simultaneous_preexisting_gm_predictions(self.PredictionsListGM,
                                                                                self.mod_pm,
                                                                                self.colsforPm,
                                                                                self.dat,
                                                                                self.Target_loc,
                                                                                self.numlags)
        self.assertListEqual(self.PredictionsListStacked, result_PredictionsListStacked)
        self.assertListEqual(self.PredictionsListGM, result_PredictionsListGM)
        self.assertListEqual(self.PredictionsListPM, result_PredictionsListPM)

    #########################
    # 37
    #########################
    def test_check_timestamp_order(self):
        result = check_timestamp_order(self.dat_time)
        self.assertEqual(result, False)

    #########################
    # 49
    #########################
    def test_nearest_round(self):
        result = nearest_round(self.myList, self.myNumber)
        self.assertEqual(result, self.nearest_round)

    #########################
    # 58
    #########################
    def test_check_preds(self):
        preds_h = [np.nan]
        with self.assertRaises(ValueError):
            check_preds(preds_h)


if __name__ == '__main__':
    unittest.main()