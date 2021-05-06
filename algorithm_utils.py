# -*- coding: utf-8 -*-
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


def knn_baseline():
    extract_prefix = './extract/' 
    with open(extract_prefix + "bert_alexa10_train.pkl",'rb') as pkl_file: 
        train_result_dict = pickle.load(pkl_file)
        X, y = train_result_dict['t_features_all_cpu'], np.asarray(train_result_dict['rating'], dtype=np.int)
        unnormed_score_pred = np.squeeze( train_result_dict['t_outputs_all_cpu'] )
        print("X.shape", X.shape, type(X), "y.shape", y.shape, type(y), unnormed_score_pred.shape, unnormed_score_pred.dtype)
        # also calculate the correlation for training data
        rho, pval = spearmanr(unnormed_score_pred, y)
        print("spearmanr rho, pval:", rho, pval)

        rho, pval = pearsonr(unnormed_score_pred, y)
        print("pearsonr rho, pval:", rho, pval)


    with open(extract_prefix + "bert_pair_alexa10.pkl",'rb') as pkl_file: 
        pair_result_dict = pickle.load(pkl_file)
        X_test_1 = pair_result_dict['dial1_t_features_all']
        X_test_2 = pair_result_dict['dial2_t_features_all']
        y_test_pair = pair_result_dict['t_targets_all_cpu']      
        print("X_test_1.shape", X_test_1.shape, type(X_test_1), "y_test_pair.shape", y_test_pair.shape, type(y_test_pair))


    for n_neighbors in range(1, X.shape[0]-100, 20):
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)  
        neigh.fit(X, y) 
        y_pred_1 = neigh.predict(X_test_1)
        y_pred_2 = neigh.predict(X_test_2)
        y_pred_pair = np.less(y_pred_1, y_pred_2).astype(int)
        
        print("n_neighbors",n_neighbors, "accuracy_score", accuracy_score(y_test_pair, y_pred_pair))

        correctly_answered = (y_test_pair == y_pred_pair)

        wrong_pair_idx = []

        for i in range(correctly_answered.shape[0]):
            if correctly_answered[i].item() is False:
                wrong_pair_idx.append(i)

        print("wrong_pair_idx", wrong_pair_idx)


    

def knn_smooth_scores():

    extract_prefix = './extract/' 
    with open(extract_prefix + "bert_alexa10_train.pkl",'rb') as pkl_file: 
        train_result_dict = pickle.load(pkl_file)
        X, y = train_result_dict['t_features_all_cpu'], np.asarray(train_result_dict['rating'], dtype=np.int)
        unnormed_score_pred = np.squeeze( train_result_dict['t_outputs_all_cpu'] )
        print("X.shape", X.shape, type(X), "y.shape", y.shape, type(y), unnormed_score_pred.shape, unnormed_score_pred.dtype)
        t_ids_all_cpu = train_result_dict['t_ids_all_cpu']
        # also calculate the correlation for training data
        rho, pval = spearmanr(unnormed_score_pred, y)
        print("spearmanr rho, pval:", rho, pval)

        rho, pval = pearsonr(unnormed_score_pred, y)
        print("pearsonr rho, pval:", rho, pval)

    for n_neighbors in [50]: 
        print("n_neighbors",n_neighbors) 
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)  
        neigh.fit(X, y) 
        y_smoothed = neigh.predict(X) # KNN-smoothing

        # also calculate the correlation for training data
        rho, pval = spearmanr(unnormed_score_pred, y_smoothed)
        print("spearmanr rho, pval:", rho, pval)

        rho, pval = pearsonr(unnormed_score_pred, y_smoothed)
        print("pearsonr rho, pval:", rho, pval)

    return y_smoothed

def do_knn_shapley():
    extract_prefix = './extract/' 
    with open(extract_prefix + "bert_alexa10_train.pkl",'rb') as pkl_file: 
        train_result_dict = pickle.load(pkl_file)
        X, y = train_result_dict['t_features_all_cpu'], np.asarray(train_result_dict['rating'], dtype=np.int)
        unnormed_score_pred = np.squeeze( train_result_dict['t_outputs_all_cpu'] )
        print("X.shape", X.shape, type(X), "y.shape", y.shape, type(y), unnormed_score_pred.shape, unnormed_score_pred.dtype)
        # also calculate the correlation for training data
        rho, pval = spearmanr(unnormed_score_pred, y)
        print("spearmanr rho, pval:", rho, pval)
        rho, pval = pearsonr(unnormed_score_pred, y)
        print("pearsonr rho, pval:", rho, pval)

    with open(extract_prefix + "bert_pair_alexa10_dev.pkl",'rb') as pkl_file: 
        pair_result_dict = pickle.load(pkl_file)
        X_test_1 = pair_result_dict['dial1_t_features_all']
        X_test_2 = pair_result_dict['dial2_t_features_all']
        y_test_pair = pair_result_dict['t_targets_all_cpu']
        print("X_test_1.shape", X_test_1.shape, type(X_test_1), "y_test_pair.shape", y_test_pair.shape, type(y_test_pair))

    N = X.shape[0]
    K = 50 

    def single_point_shapley(xt_query):
        distance1 = np.sum(np.square(X-xt_query), axis=1)
        alpha = np.argsort(distance1)
        shapley_arr = np.zeros(N)
        for i in range(N-1, -1, -1): 
            if i == N-1: 
                shapley_arr[alpha[i]] = y[alpha[i]]/N
                continue
            else:
                shapley_arr[alpha[i]] = shapley_arr[alpha[i+1]] + (y[alpha[i]] - y[alpha[i+1]])/K * min(K,i+1)/(i+1)
        
        return shapley_arr

    global_shapley_arr = np.zeros(N)
    for xt1, xt2, cmplabel in zip(X_test_1, X_test_2, y_test_pair):
        s1 = single_point_shapley(xt1)
        s2 = single_point_shapley(xt2)

        if cmplabel==0:
            global_shapley_arr += s1 - s2
        else:
            assert cmplabel==1
            global_shapley_arr += s2 - s1
        
    global_shapley_arr /= y_test_pair.shape[0] 
    print("global_shapley_arr", global_shapley_arr)
    print("negative count:", np.sum(global_shapley_arr < 0), "all:", X.shape[0]  )

    _ = plt.hist(global_shapley_arr, bins='auto')  
    plt.title("Generated Raw Scores of the Training Set")
    plt.savefig("shapleyVal.png", bbox_inches = 'tight')
    plt.clf()

    X_clened, y_cleaned = np.zeros((0, X.shape[1])), np.zeros(0, dtype=int)  
    for i in range(y.shape[0]):
        # pass
        if global_shapley_arr[i] > 0.:
            X_clened = np.concatenate([X_clened, X[[i]]])
            y_cleaned = np.concatenate([y_cleaned, y[[i]]])

    print("X_clened",X_clened.shape)
    for n_neighbors in range(1, X_clened.shape[0]-100, 20):
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)  
        neigh.fit(X_clened, y_cleaned) 
        y_pred_1 = neigh.predict(X_test_1)
        y_pred_2 = neigh.predict(X_test_2)
        y_pred_pair = np.less(y_pred_1, y_pred_2).astype(int)
        print("n_neighbors",n_neighbors, "accuracy_score", accuracy_score(y_test_pair, y_pred_pair))

    return global_shapley_arr
    
 

if __name__ == "__main__":
    knn_baseline()
    knn_smooth_scores()
    do_knn_shapley()