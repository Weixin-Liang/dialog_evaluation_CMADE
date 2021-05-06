# -*- coding: utf-8 -*-

#######################
# Apply Data Shapely Methods to assign a value for each training datum
# And verify that Removing training data with low Shapley
# value improves the performance of the KNN regressor
#
# Beyond User Self-Reported Likert Scale Ratings: A Comparison Model for Automatic Dialog Evaluation (ACL 2020)
# Weixin Liang, James Zou and Zhou Yu
# 
# HERALD: An Annotation Efficient Method to Train User Engagement Predictors in Dialogs (ACL 2021)
# Weixin Liang, Kai-Hui Liang and Zhou Yu
#######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import os
import sys
import time
import sklearn
import pickle
import csv
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve, classification_report, confusion_matrix
from scipy.stats import spearmanr, pearsonr


def detailed_do_knn_shapley(extract_prefix = './extract_engagement/'):

    with open(extract_prefix + "Gunrock_engagement_train.pkl",'rb') as pkl_file:
        train_result_dict = pickle.load(pkl_file)
        X, y = train_result_dict['t_features_all_cpu'], np.asarray(train_result_dict['t_targets_all_cpu'], dtype=np.int)

        conversation_id_list = np.asarray(train_result_dict['conversation_id_list'])
        conversation_turn_num_list = np.asarray(train_result_dict['conversation_turn_num_list'])
        unnormed_score_pred = np.squeeze( train_result_dict['t_outputs_socre_all'] ) # 't_outputs_all_cpu'
        print("X.shape", X.shape, type(X), "y.shape", y.shape, type(y), "unnormed_score_pred", unnormed_score_pred.shape, unnormed_score_pred.dtype)
        list_text_left_list = train_result_dict["list_text_left_list"]

    X = np.concatenate((X, X), axis=0)
    y = np.concatenate((y, 1-y), axis=0)
    list_text_left_list = np.concatenate((list_text_left_list, list_text_left_list), axis=0)
    conversation_id_list = np.concatenate((conversation_id_list, conversation_id_list), axis=0)

    with open(extract_prefix + "Gunrock_engagement_testdev.pkl",'rb') as pkl_file:
        testdev_result_dict = pickle.load(pkl_file)
        X_testdev, y_testdev = testdev_result_dict['t_features_all_cpu'], np.asarray(testdev_result_dict['t_targets_all_cpu'], dtype=np.int)
        print("X_testdev.shape", X_testdev.shape, type(X_testdev), "y_testdev.shape", y_testdev.shape, type(y_testdev))

    with open(extract_prefix + "Gunrock_engagement_test.pkl",'rb') as pkl_file:
        test_result_dict = pickle.load(pkl_file)
        X_test, y_test = test_result_dict['t_features_all_cpu'], np.asarray(test_result_dict['t_targets_all_cpu'], dtype=np.int)
        print("X_test.shape", X_test.shape, type(X_test), "y_test.shape", y_test.shape, type(y_test))

    N = X.shape[0]
    K = 10

    def single_point_shapley(xt_query, y_tdev_label):
        distance1 = np.sum(np.square(X-xt_query), axis=1)
        alpha = np.argsort(distance1)
        shapley_arr = np.zeros(N)
        for i in range(N-1, -1, -1): 
            if i == N-1:
                shapley_arr[alpha[i]] = int(y[alpha[i]] == y_tdev_label) /N
            else:
                shapley_arr[alpha[i]] = shapley_arr[alpha[i+1]] + ( int(y[alpha[i]]==y_tdev_label) - int(y[alpha[i+1]]==y_tdev_label) )/K * min(K,i+1)/(i+1)
        return shapley_arr


    global_shapley_arr = np.zeros(N)
    for x_tdev, y_tdev_label in zip(X_testdev, y_testdev):
        s1 = single_point_shapley(x_tdev, y_tdev_label)
        global_shapley_arr += s1
    global_shapley_arr /= y_testdev.shape[0]
    print("negative count:", np.sum(global_shapley_arr < 0), "all:", X.shape[0]  )
    _ = plt.hist(global_shapley_arr)  
    plt.title("Calculated Shapley Value for Step 3")
    plt.savefig("./shapleyOut/HistShapleyVal.png", bbox_inches = 'tight')
    plt.clf()

    shapley_pkl_path = './shapleyOut/shapley_value.pkl'
    with open(shapley_pkl_path, 'wb') as pkl_file:
        data_dict = {
            "global_shapley_arr": global_shapley_arr,
            "conversation_id_list": conversation_id_list,
            "X": X,
            "y": y,
            "list_text_left_list" : list_text_left_list,
        }
        pickle.dump(data_dict, pkl_file)

    X_clened, y_cleaned = np.zeros((0, X.shape[1])), np.zeros(0, dtype=int)
    for i in range(y.shape[0]):
        if global_shapley_arr[i] > 0.:
            X_clened = np.concatenate([X_clened, X[[i]]])
            y_cleaned = np.concatenate([y_cleaned, y[[i]]])


    print("X_clened",X_clened.shape)
    for n_neighbors in [K]:
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(X_clened, y_cleaned)
        y_pred_testdev = neigh.predict(X_testdev)
        y_pred_pair = (y_pred_testdev>0.5).astype(int)
        print("n_neighbors",n_neighbors, "DEV accuracy_score", accuracy_score(y_testdev, y_pred_pair))
        print("classification_report", classification_report(y_testdev,y_pred_pair))

        y_pred_test = neigh.predict(X_test)
        y_pred_pair = (y_pred_test>0.5).astype(int)
        print("n_neighbors",n_neighbors, "TEST accuracy_score", accuracy_score(y_test, y_pred_pair))
        print("classification_report", classification_report(y_test,y_pred_pair))

    shaley_rank_array = np.argsort(global_shapley_arr)

    for idx in shaley_rank_array[:20]:
        print("========= 20 Most Negative Rank {}:=========\n".format(idx),
        "shapley value", global_shapley_arr[idx],
        "LABEL:", y[idx]
        )
        print(list_text_left_list[idx])

    for idx in shaley_rank_array[-20:]:
        print("========= 20 Most Positve Rank {}:=========\n".format(idx),
        "shapley value", global_shapley_arr[idx],
        "LABEL:", y[idx]
        )
        print(list_text_left_list[idx])


    def __subsets_performance_(datalens, idxarray, printDetails=False, n_neighbors = K):
        perfs = []
        for datalen in datalens:
            batch_data_ids = idxarray[:datalen]
            X_cleaned_tmp = X[batch_data_ids]
            y_cleaned_tmp = y[batch_data_ids]

            neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
            neigh.fit(X_cleaned_tmp, y_cleaned_tmp)
            y_pred_test = neigh.predict(X_test)
            y_pred_pair = (y_pred_test > 0.5).astype(int)
            perfs.append( 100 * accuracy_score(y_test, y_pred_pair) ) # (%)
            if printDetails:
                print("datalen", datalen, "n_neighbors",n_neighbors, "accuracy_score", accuracy_score(y_test, y_pred_pair))
        return perfs

    plt.rcParams['figure.figsize'] = 8,8
    plt.rcParams['font.size'] = 25
    params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
    plt.rcParams.update(params)

    plt.xlabel('Fraction of train data removed (%)')
    plt.ylabel('Test bACC (%)', fontsize=20)
    plot_points = list( range( int(N ), max(int(N * 0.05), 50)  , -25) )
    print("N", N, "plot_points", plot_points)

    retain_useful_perfs1 = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array[::-1], printDetails=True, n_neighbors=1)
    retain_useful_perfs5 = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array[::-1], printDetails=True, n_neighbors=5)
    retain_useful_perfs10 = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array[::-1], printDetails=True, n_neighbors=10)
    retain_useful_perfs25 = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array[::-1], printDetails=True, n_neighbors=25)
    retain_useful_perfs50 = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array[::-1], printDetails=True, n_neighbors=50)

    retain_hurtful_perfs = __subsets_performance_(datalens = plot_points, idxarray=shaley_rank_array, n_neighbors=1, printDetails=True)

    random_perfs = np.mean([
        __subsets_performance_(datalens = plot_points,
        idxarray=np.random.permutation(N), n_neighbors = K, printDetails=True )
        for _ in range(3)], 0)


    plot_points = (1 - np.array(plot_points)/N) * 100
    plt.plot(plot_points, retain_useful_perfs1, '-', lw=5, ms=10, color='k')
    plt.plot(plot_points, retain_useful_perfs5, '-', lw=5, ms=10, color='orange')
    plt.plot(plot_points, retain_useful_perfs10,  '-', lw=5, ms=10, color='y')
    plt.plot(plot_points, retain_useful_perfs25, '-', lw=5, ms=10, color='g')
    plt.plot(plot_points, retain_useful_perfs50, '-', lw=5, ms=10, color='c')
    plt.plot(plot_points, retain_hurtful_perfs, '-', lw=5, ms=10, color='b')
    plt.plot(plot_points, random_perfs, ':', lw=5, ms=10, color='r')
    legends = [
        'Shapley ($K_{test}=1$)',
        'Shapley ($K_{test}=5$)',
        'Shapley ($K_{test}=10$)',
        'Shapley ($K_{test}=25$)',
        'Shapley ($K_{test}=50$)',
        'RetainHurtful ($K_{test}=' + str(K) +'$)',
        'Random ($K_{test}=' + str(K) + '$)',
        ]
    plt.legend(legends, loc='lower left', bbox_to_anchor=(1.04, -0) )
    plt.savefig('{}.png'.format("./shapleyOut/shapleyRemove"),
                bbox_inches = 'tight')
    plt.close()
    return global_shapley_arr


if __name__ == "__main__":

    detailed_do_knn_shapley(extract_prefix = './extract_engagement/')
