import numpy as np
import pandas as pd
import math
from lenskit.metrics import topn as tnmetrics
from pathlib import Path

from sklearn import metrics
import sys


def get_relevants_by_user(df, threshold):
    df = df[df.rating >= threshold]

    return df.item


def get_top_n(items_scores, n):
    items_scores = items_scores.sort_values(by=['score'], ascending=False)

    items_scores_n = items_scores.head(n)

    return items_scores_n


# Precision


def precision(recomendations, relevant):
    mask = np.isin(recomendations, relevant)
    P = len(recomendations[mask]) / len(recomendations)

    return P


# def precision(recommendations, relevant):
#     precision = tnmetrics.precision(recommendations, relevant)
#
#     if math.isnan(precision):
#         precision = 0
#
#     return precision


# Recall


def recall(recomendations, relevant):
    mask = np.isin(recomendations, relevant)
    if len(relevant != 0):
        R = len(recomendations[mask]) / len(relevant)

    else:
        R = 0

    return R


# def recall(recommendations, relevant):
#     recall = tnmetrics.recall(recommendations, relevant)
#
#     if math.isnan(recall):
#         recall = 0
#
#     return recall


# F-Measure

def fmeasure(precision, recall):
    if precision == 0 and recall == 0:

        fm = 0

    else:
        fm = 2 * ((precision * recall) / (precision + recall))

    return fm


# dcg


def get_real_item_rating(rank, user_ratings):
    # map the items to the rating given by the user

    rank["rating"] = rank["item"].map(user_ratings.set_index('item')["rating"]).fillna(0)

    return rank


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def reciprocal_rank(rs):
    if len(rs[rs != 0] > 0):

        first_nonzero_position = rs.to_numpy().nonzero()[0][0] + 1
        #print(first_nonzero_position)
        rr = 1 / first_nonzero_position

    else:
        rr = 0

    return rr


def false_positive_rate(list_of_test_items, relevants, rank):
    '''

    :param list_of_test_items: all candidates items to be recommended
    :param relevants: all relevants items in the test set
    :param rank: listed ranked by the algorithm
    :return: false positive rate
    '''

    mask = np.isin(list_of_test_items, relevants)

    all_negatives = list_of_test_items[~mask]

    mask2 = np.isin(rank.item, relevants)

    fp = rank.item[~mask2]

    if len(all_negatives) != 0:

        fpr = len(fp) / len(all_negatives)

    else:
        fpr = 0

    return fpr




def get_auc(recall, fpr):


    # add point 1,1

    recall.append(1)
    fpr.append(1)

    if np.sum(np.array(fpr))==0:
        auc = 1

    else:

        auc = metrics.auc(fpr, recall)
    del recall[-1]
    del fpr[-1]

    return auc


def auc_(recall, fpr, max_n):

    auc_list = []

    for a in range(1, max_n):

        recall_ = recall[0:a+1]

        fpr_ = fpr[0:a+1]


        auc = metrics.auc(fpr_, recall_)

        auc_list.append(auc)

    print(auc_list)

    return np.array(auc_list)





def topk_metrics_sum(P, R, F, rr, nDCG, n):

    #print(P, R, F, n)

    my_file = Path("temp" + str(n) + ".csv")
    if my_file.is_file():

        df = pd.read_csv(my_file, sep=',',header=None)


        df_array = np.array(df)


        df_array[0] = np.add(np.array([P, R, F, rr, nDCG]), df_array[0])
        print(df_array)

        #print(df_array)
        np.savetxt("temp" + str(n) + ".csv", df_array, delimiter=",")



        #line = pd.DataFrame(np.sum([df_array, [P, R, F]], axis=0))
        #print(line)

        #line.to_csv("mlData/temp" + str(n) + ".csv")


    else:
        #print("first top k")
        line = np.array([[P, R, F, rr, nDCG]])

        np.savetxt("temp" + str(n) + ".csv", line, delimiter=",")
        #line.to_csv("mlData/temp" + str(n) + ".csv")




# RMSE

def rmse(predictions_list, real_list):
    rmse = predict.rmse(predictions_list, real_list, missing='ignore')

    return rmse


# MAE


def mae(predictions_list, real_list):
    mae = predict.mae(predictions_list, real_list, missing='ignore')

    return mae


def normalize_between_range(array, a, b):
    array = ((b - a) * ((array - array.min()) / (array.max() - array.min()))) + a

    return array
