import implicit
from recommender_evaluation import *
from cross_val import *
import mysql.connector
from itertools import product
from semsimcalculus import *
import gc
import numpy as np
import pandas as pd
#pd.set_option('display.max_rows', 1000000)

def connect(host, user, password, database):
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    return mydb


def recommendations(model, train_data, test_items, user):
    user_items = train_data.T.tocsr()  # user, item, rating

    ranking_items = model.rank_items(user, user_items, test_items)

    return ranking_items


def map_original_id_to_system_id(item_score, original_item_id):
    """
    map the original id to the system ids
    :param item_score:
    :param original_item_id:
    :return:
    """

    item_score_ontology = item_score.rename(columns={"item": "item_chebi"})
    item_score_ontology["item"] = item_score_ontology["item_chebi"].map(
        original_item_id.set_index('item')["new_index"]).fillna(0)

    return item_score_ontology


def map_system_id_to_original_id(item_score, original_item_id):
    """
    map the id to the original ids
    :param item_score:
    :param original_item_id:
    :return:
    """

    #item_score_ontology = item_score.rename(columns={"item": "item_chebi"})
    item_score["item_chebi"] = item_score["item"].map(
        original_item_id.set_index('new_index')["item"]).fillna(0)

    return item_score

def select_metric(scores_by_item, metric, n):
    """
    select the column with the metric to use
    :param scores_by_item: pd DataFrame with all compounds and metrics
    :param metric: metric to select to calculate the mean of the similarities
    :return: pd DataFrame with columns item, score
    """

    item_score = scores_by_item[['comp_1', metric]]

    #item_score = item_score.groupby(['comp_1']).sum().reset_index().sort_values(metric, ascending=False).head(5).mean()

    item_score = item_score.groupby('comp_1').apply(
        lambda x: x.sort_values((metric), ascending=False).head(n).mean())

    item_score = item_score.rename(columns={"comp_1": "item", metric: "score"})
    item_score.item = item_score.item.astype(int)

    return item_score


def onto_algorithm(train_ratings_for_t_us, mydb, test_items_chebi_id, n):
    """

    :param train_ratings_for_t_us:
    :param mydb:
    :param test_items_chebi_id:
    :param host:
    :param user:
    :param password:
    :param database:
    :param path_to_ontology:
    :return: pandas dataframe: columns = item, score (item with chebi_id)
    """

    # get just the IDs of the items in the train set

    train_items_for_t_us = train_ratings_for_t_us.item.unique()

    ####################training items for this user to be used for finding the similarity

    # get the score for each item in the test set
    scores_by_item = get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us)

    item_score_lin = select_metric(scores_by_item, 'sim_lin', n)
    item_score_resnik = select_metric(scores_by_item, 'sim_resnik', n)
    item_score_jc = select_metric(scores_by_item, 'sim_jc', n)

    return item_score_lin, item_score_resnik, item_score_jc


def all_evaluation_metrics(k, item_score, ratings_t_us, test_items, relevant, metrics_dict):
    user_r = [0.0]
    user_fpr = [0.0]

    for i in range(1, k + 1):

        top_n = get_top_n(item_score, i)

        top_n.item = top_n.item.astype(int)

        topn_real_ratings = get_real_item_rating(top_n, ratings_t_us).rating

        fpr = false_positive_rate(test_items, relevant, top_n)

        recs = np.array(top_n.item).astype(int)
        P = precision(recs, np.array(relevant))
        R = recall(recs, np.array(relevant))
        F = fmeasure(P, R)
        rr = reciprocal_rank(topn_real_ratings)

        user_r.append(R)
        user_fpr.append(fpr)

        nDCG = ndcg_at_k(topn_real_ratings, i, method=0)

        # auc = metrics.auc(user_fpr, user_r)

        auc = get_auc(user_r, user_fpr)

        if len(metrics_dict) != k:
            metrics_dict.update({'top' + str(i): [P, R, F, fpr, rr, nDCG, auc]})

        else:
            old = np.array(metrics_dict['top' + str(i)])
            new = np.array([P, R, F, fpr, rr, nDCG, auc])

            to_update = old + new

            metrics_dict.update({'top' + str(i): to_update})

    return metrics_dict


def get_evaluation(test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                   ratings_train_sparse_CF, test_items, k, all_ratings, original_item_id, mydb, n):
    # CB
    metrics_dict_onto_lin = {}
    metrics_dict_onto_resnik = {}
    metrics_dict_onto_jc = {}

    # CF
    metrics_dict_ALS = {}
    metrics_dict_BPR = {}
    metrics_dict_item_item = {}

    # Hybrid
    metrics_dict_ALS_ONTO_lin_metric1 = {}
    metrics_dict_ALS_ONTO_resnik_metric1 = {}
    metrics_dict_ALS_ONTO_jc_metric1 = {}
    metrics_dict_BPR_ONTO_lin_metric1 = {}
    metrics_dict_BPR_ONTO_resnik_metric1 = {}
    metrics_dict_BPR_ONTO_jc_metric1 = {}

    metrics_dict_ALS_ONTO_lin_metric2 = {}
    metrics_dict_ALS_ONTO_resnik_metric2 = {}
    metrics_dict_ALS_ONTO_jc_metric2 = {}
    metrics_dict_BPR_ONTO_lin_metric2 = {}
    metrics_dict_BPR_ONTO_resnik_metric2 = {}
    metrics_dict_BPR_ONTO_jc_metric2 = {}

    model_bayes = implicit.bpr.BayesianPersonalizedRanking(factors=150, num_threads=10, use_gpu=False)
    model_als = implicit.als.AlternatingLeastSquares(factors=150, num_threads=10, use_gpu=False)

    model_als.fit(ratings_train_sparse_CF)
    model_bayes.fit(ratings_train_sparse_CF)

    progress = 0
    users_to_remove = 0
    relevant_items_sum = 0

    # test items chebi id!!!  what i'm rating. Array is equal for all users

    # to use in onto algorithm
    test_items_chebi_id = all_ratings[all_ratings.index_item.isin(
        test_items)].item.unique()

    for t_us in test_users:

        # print("user id = ", t_us)

        progress += 1
        print(progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r")
        # print(progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items)

        sys.stdout.flush()

        # all ratings for user t_us (index_user)
        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]

        # train ratings for user t_us
        train_ratings_for_t_us_CB = all_ratings_for_t_us[
            ~(all_ratings_for_t_us.index_item.isin(ratings_test.index_item))]

        # verify it user has condition to be evaluated, i.e., it has al least one item in the test set
        ratings_test_t_us = all_ratings_for_t_us[(all_ratings_for_t_us.index_item.isin(ratings_test.index_item))]

        if np.sum(ratings_test_t_us.rating) == 0:
            users_to_remove += 1

            continue

        if len(train_ratings_for_t_us_CB) == 0:
            users_to_remove += 1

            continue


        ####
        item_score_lin, item_score_resnik, item_score_jc = onto_algorithm(train_ratings_for_t_us_CB, mydb,
                                                                          test_items_chebi_id, n)


        item_score_lin = map_original_id_to_system_id(item_score_lin, original_item_id)
        item_score_resnik = map_original_id_to_system_id(item_score_resnik, original_item_id)
        item_score_jc = map_original_id_to_system_id(item_score_jc, original_item_id)
        item_score_implicit_als = get_score_by_implicit(model_als, ratings_train_sparse_CF, test_items, t_us)
        item_score_implicit_als = map_system_id_to_original_id(item_score_implicit_als, original_item_id)
        item_score_implicit_bpr = get_score_by_implicit(model_bayes, ratings_train_sparse_CF, test_items, t_us)
        item_score_implicit_bpr = map_system_id_to_original_id(item_score_implicit_bpr, original_item_id)

        # print('onto lin')
        # print(item_score_lin.sort_values(by=['score'], ascending=False).head(20))
        # print('onto resnik')
        # print(item_score_resnik.sort_values(by=['score'], ascending=False).head(20))
        # print('onto jc')
        # print(item_score_jc.sort_values(by=['score'], ascending=False).head(20))
        # print('als')
        # print(item_score_implicit_als.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr')
        # print(item_score_implicit_bpr.sort_values(by=['score'], ascending=False).head(20))

        item_score_ALS_ONTO_lin_metric1 = merge_algorithms_scores(item_score_lin, item_score_implicit_als, 1)
        # print("item_score_ALS_ONTO_lin_metric1 ", item_score_ALS_ONTO_lin_metric1)

        item_score_ALS_ONTO_lin_metric2 = merge_algorithms_scores(item_score_lin, item_score_implicit_als, 2)
        item_score_BPR_ONTO_lin_metric1 = merge_algorithms_scores(item_score_lin, item_score_implicit_bpr, 1)
        item_score_BPR_ONTO_lin_metric2 = merge_algorithms_scores(item_score_lin, item_score_implicit_bpr, 2)

        # print('ALS_ONTO_lin_metric1')
        # print(item_score_ALS_ONTO_lin_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('ALS_ONTO_lin_metric2')
        # print(item_score_ALS_ONTO_lin_metric2.sort_values(by=['score'], ascending=False).head(20))
        # print('BPR_ONTO_lin_metric1')
        # print(item_score_BPR_ONTO_lin_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('BPR_ONTO_lin_metric2')
        # print(item_score_BPR_ONTO_lin_metric2.sort_values(by=['score'], ascending=False).head(20))

        item_score_ALS_ONTO_resnik_metric1 = merge_algorithms_scores(item_score_resnik, item_score_implicit_als, 1)
        item_score_ALS_ONTO_resnik_metric2 = merge_algorithms_scores(item_score_resnik, item_score_implicit_als, 2)
        item_score_BPR_ONTO_resnik_metric1 = merge_algorithms_scores(item_score_resnik, item_score_implicit_bpr, 1)
        item_score_BPR_ONTO_resnik_metric2 = merge_algorithms_scores(item_score_resnik, item_score_implicit_bpr, 2)

        # print('ALS_ONTO_resnik_metric1')
        # print(item_score_ALS_ONTO_resnik_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('ALS_ONTO_resnik_metric2')
        # print(item_score_ALS_ONTO_resnik_metric2.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_ONTO_resnik_metric1')
        # print(item_score_BPR_ONTO_resnik_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_ONTO_resnik_metric2')
        # print(item_score_BPR_ONTO_resnik_metric2.sort_values(by=['score'], ascending=False).head(20))


        item_score_ALS_ONTO_jc_metric1 = merge_algorithms_scores(item_score_jc, item_score_implicit_als, 1)

        item_score_ALS_ONTO_jc_metric2 = merge_algorithms_scores(item_score_jc, item_score_implicit_als, 2)

        item_score_BPR_ONTO_jc_metric1 = merge_algorithms_scores(item_score_jc, item_score_implicit_bpr, 1)

        item_score_BPR_ONTO_jc_metric2 = merge_algorithms_scores(item_score_jc, item_score_implicit_bpr, 2)

        # print('ALS_ONTO_jc_metric1')
        # print(item_score_ALS_ONTO_jc_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('ALS_ONTO_jc_metric2')
        # print(item_score_ALS_ONTO_jc_metric2.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_ONTO_jc_metric1')
        # print(item_score_BPR_ONTO_jc_metric1.sort_values(by=['score'], ascending=False).head(20))
        # print('bpr_ONTO_jc_metric2')
        # print(item_score_BPR_ONTO_jc_metric2.sort_values(by=['score'], ascending=False).head(20))



        relevant = get_relevants_by_user(ratings_test_t_us, 0)
        # print("relevant: ", relevant)



        relevant_items_sum += len(relevant)  # so esta a fazer media

        metrics_dict_onto_lin = all_evaluation_metrics(k, item_score_lin, ratings_test_t_us, test_items,
                                                       relevant.index_item,
                                                       metrics_dict_onto_lin)
        metrics_dict_onto_resnik = all_evaluation_metrics(k, item_score_resnik, ratings_test_t_us, test_items,
                                                          relevant.index_item,
                                                          metrics_dict_onto_resnik)
        metrics_dict_onto_jc = all_evaluation_metrics(k, item_score_jc, ratings_test_t_us, test_items,
                                                      relevant.index_item,
                                                      metrics_dict_onto_jc)

        metrics_dict_ALS = all_evaluation_metrics(k, item_score_implicit_als, ratings_test_t_us, test_items,
                                                  relevant.index_item,
                                                  metrics_dict_ALS)
        metrics_dict_BPR = all_evaluation_metrics(k, item_score_implicit_bpr, ratings_test_t_us, test_items,
                                                  relevant.index_item,
                                                  metrics_dict_BPR)
        metrics_dict_ALS_ONTO_lin_metric1 = all_evaluation_metrics(k, item_score_ALS_ONTO_lin_metric1,
                                                                   ratings_test_t_us, test_items, relevant.index_item,
                                                                   metrics_dict_ALS_ONTO_lin_metric1)
        metrics_dict_BPR_ONTO_lin_metric1 = all_evaluation_metrics(k, item_score_BPR_ONTO_lin_metric1,
                                                                   ratings_test_t_us, test_items, relevant.index_item,
                                                                   metrics_dict_BPR_ONTO_lin_metric1)

        metrics_dict_ALS_ONTO_resnik_metric1 = all_evaluation_metrics(k, item_score_ALS_ONTO_resnik_metric1,
                                                                      ratings_test_t_us, test_items,
                                                                      relevant.index_item,
                                                                      metrics_dict_ALS_ONTO_resnik_metric1)
        metrics_dict_BPR_ONTO_resnik_metric1 = all_evaluation_metrics(k, item_score_BPR_ONTO_resnik_metric1,
                                                                      ratings_test_t_us, test_items,
                                                                      relevant.index_item,
                                                                      metrics_dict_BPR_ONTO_resnik_metric1)

        metrics_dict_ALS_ONTO_jc_metric1 = all_evaluation_metrics(k, item_score_ALS_ONTO_jc_metric1,
                                                                  ratings_test_t_us, test_items,
                                                                  relevant.index_item,
                                                                  metrics_dict_ALS_ONTO_jc_metric1)
        metrics_dict_BPR_ONTO_jc_metric1 = all_evaluation_metrics(k, item_score_BPR_ONTO_jc_metric1,
                                                                  ratings_test_t_us, test_items,
                                                                  relevant.index_item,
                                                                  metrics_dict_BPR_ONTO_jc_metric1)

        # hybrid metric 2
        metrics_dict_ALS_ONTO_lin_metric2 = all_evaluation_metrics(k, item_score_ALS_ONTO_lin_metric2,
                                                                   ratings_test_t_us, test_items, relevant.index_item,
                                                                   metrics_dict_ALS_ONTO_lin_metric2)
        metrics_dict_BPR_ONTO_lin_metric2 = all_evaluation_metrics(k, item_score_BPR_ONTO_lin_metric2,
                                                                   ratings_test_t_us, test_items, relevant.index_item,
                                                                   metrics_dict_BPR_ONTO_lin_metric2)

        metrics_dict_ALS_ONTO_resnik_metric2 = all_evaluation_metrics(k, item_score_ALS_ONTO_resnik_metric2,
                                                                      ratings_test_t_us, test_items,
                                                                      relevant.index_item,
                                                                      metrics_dict_ALS_ONTO_resnik_metric2)
        metrics_dict_BPR_ONTO_resnik_metric2 = all_evaluation_metrics(k, item_score_BPR_ONTO_resnik_metric2,
                                                                      ratings_test_t_us, test_items,
                                                                      relevant.index_item,
                                                                      metrics_dict_BPR_ONTO_resnik_metric2)

        metrics_dict_ALS_ONTO_jc_metric2 = all_evaluation_metrics(k, item_score_ALS_ONTO_jc_metric2,
                                                                  ratings_test_t_us, test_items,
                                                                  relevant.index_item,
                                                                  metrics_dict_ALS_ONTO_jc_metric2)
        metrics_dict_BPR_ONTO_jc_metric2 = all_evaluation_metrics(k, item_score_BPR_ONTO_jc_metric2,
                                                                  ratings_test_t_us, test_items,
                                                                  relevant.index_item,
                                                                  metrics_dict_BPR_ONTO_jc_metric2)

    test_users_size = test_users_size - users_to_remove

    print("n users removed: ", users_to_remove)

    relevant_items_mean = relevant_items_sum / test_users_size

    print("mean of relevant items: ", relevant_items_mean)
    metrics_dict_onto_lin = calculate_dictionary_mean(metrics_dict_onto_lin, float(test_users_size))
    metrics_dict_onto_resnik = calculate_dictionary_mean(metrics_dict_onto_resnik, float(test_users_size))
    metrics_dict_onto_jc = calculate_dictionary_mean(metrics_dict_onto_jc, float(test_users_size))

    metrics_dict_ALS = calculate_dictionary_mean(metrics_dict_ALS, float(test_users_size))
    metrics_dict_BPR = calculate_dictionary_mean(metrics_dict_BPR, float(test_users_size))

    metrics_dict_ALS_ONTO_lin_metric1 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_lin_metric1,
                                                                  float(test_users_size))
    metrics_dict_ALS_ONTO_resnik_metric1 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_resnik_metric1,
                                                                     float(test_users_size))
    metrics_dict_ALS_ONTO_jc_metric1 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_jc_metric1,
                                                                 float(test_users_size))

    metrics_dict_BPR_ONTO_lin_metric1 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_lin_metric1,
                                                                  float(test_users_size))
    metrics_dict_BPR_ONTO_resnik_metric1 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_resnik_metric1,
                                                                     float(test_users_size))
    metrics_dict_BPR_ONTO_jc_metric1 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_jc_metric1,
                                                                 float(test_users_size))

    metrics_dict_ALS_ONTO_lin_metric2 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_lin_metric2,
                                                                  float(test_users_size))
    metrics_dict_ALS_ONTO_resnik_metric2 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_resnik_metric2,
                                                                     float(test_users_size))
    metrics_dict_ALS_ONTO_jc_metric2 = calculate_dictionary_mean(metrics_dict_ALS_ONTO_jc_metric2,
                                                                 float(test_users_size))

    metrics_dict_BPR_ONTO_lin_metric2 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_lin_metric2,
                                                                  float(test_users_size))
    metrics_dict_BPR_ONTO_resnik_metric2 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_resnik_metric2,
                                                                     float(test_users_size))
    metrics_dict_BPR_ONTO_jc_metric2 = calculate_dictionary_mean(metrics_dict_BPR_ONTO_jc_metric2,
                                                                 float(test_users_size))
    del model_bayes
    del model_als
    gc.collect()

    return metrics_dict_onto_lin, metrics_dict_onto_resnik, metrics_dict_onto_jc, metrics_dict_ALS, metrics_dict_BPR, \
           metrics_dict_ALS_ONTO_lin_metric1, metrics_dict_ALS_ONTO_resnik_metric1, metrics_dict_ALS_ONTO_jc_metric1, \
           metrics_dict_BPR_ONTO_lin_metric1, metrics_dict_BPR_ONTO_resnik_metric1, metrics_dict_BPR_ONTO_jc_metric1, \
           metrics_dict_ALS_ONTO_lin_metric2, metrics_dict_ALS_ONTO_resnik_metric2, metrics_dict_ALS_ONTO_jc_metric2,\
           metrics_dict_BPR_ONTO_lin_metric2, metrics_dict_BPR_ONTO_resnik_metric2, metrics_dict_BPR_ONTO_jc_metric2


def get_all_metrics_by_cv_implicit(test_users, test_users_size, count_cv, count_cv_items, ratings_test, model_als,
                                   ratings_sparse, test_items, k, train_ratings):
    metrics_dict = {}

    progress = 0

    users_to_remove = 0
    relevant_items_sum = 0

    for t_us in test_users:

        progress += 1
        print(progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r")

        sys.stdout.flush()

        ratings_t_us = ratings_test[ratings_test.user == t_us]
        train_ratings_t_us = train_ratings[train_ratings.user == t_us]

        if np.sum(ratings_t_us.rating) == 0:
            users_to_remove += 1

            continue

        if len(train_ratings_t_us) == 0:
            users_to_remove += 1

            continue

        relevant = get_relevants_by_user(ratings_t_us, 0)
        relevant_items_sum += len(relevant)

        item_score = recommendations(model_als, ratings_sparse, test_items, t_us)
        item_score = pd.DataFrame(np.array(item_score), columns=["item", "score"])

        user_r = [0.0]
        user_fpr = [0.0]

        for i in range(1, k + 1):

            top_n = get_top_n(item_score, i)

            top_n.item = top_n.item.astype(int)

            topn_real_ratings = get_real_item_rating(top_n, ratings_t_us).rating

            fpr = false_positive_rate(test_items, relevant, top_n)

            recs = np.array(top_n.item).astype(int)
            P = precision(recs, np.array(relevant))
            R = recall(recs, relevant)
            F = fmeasure(P, R)
            rr = reciprocal_rank(topn_real_ratings)

            user_r.append(R)
            user_fpr.append(fpr)

            nDCG = ndcg_at_k(topn_real_ratings, i, method=0)

            # auc = metrics.auc(user_fpr, user_r)
            auc = get_auc(user_r, user_fpr)

            if len(metrics_dict) != k:
                metrics_dict.update({'top' + str(i): [P, R, F, fpr, rr, nDCG, auc]})

            else:
                old = np.array(metrics_dict['top' + str(i)])
                new = np.array([P, R, F, fpr, rr, nDCG, auc])

                to_update = old + new

                metrics_dict.update({'top' + str(i): to_update})

    test_users_size = test_users_size - users_to_remove

    relevant_items_mean = relevant_items_sum / test_users_size

    print("mean of relevant items: ", relevant_items_mean)
    metrics_dict = calculate_dictionary_mean(metrics_dict, float(test_users_size))
    print(metrics_dict)
    print("n users removed: ", users_to_remove)

    return metrics_dict


# def calculate_semantic_similarity(test_items_chebi, train_items_chebi):
#     score_list = []
#     count = 0
#     for it1 in test_items_chebi:
#         it1 = "CHEBI_" + str(it1)
#         print(it1)
#         e1 = ssmpy.get_id(it1)
#         it2_sim = 0
#         for it2 in train_items_chebi:
#             it2 = "CHEBI_" + str(it2)
#             print(it2)
#             e2 = ssmpy.get_id(it2)
#             it2_sim += ssmpy.ssm_resnik(e1, e2)
#             sys.stdout.flush()
#             count += 1
#         score_list.append(it2_sim / len(train_items_chebi))
#
#     df = pd.DataFrame(test_items_chebi, columns=['item'])
#     df['score'] = score_list
#     print(df)
#
#     df = df.sort_values(by=['score'], ascending=False)
#     print(df)
#
#     return df



def get_sims(mydb, list1, list2):
    list1 = list1.tolist()

    list2 = list2.tolist()

    format_strings1 = ','.join(['%s'] * len(list1))
    format_strings2 = ','.join(['%s'] * len(list2))
    sql = "select * from similarity where comp_1 in (%s) and comp_2 in (%s)"
    format_strings1 = format_strings1 % tuple(list1)
    format_strings2 = format_strings2 % tuple(list2)
    sql = sql % (format_strings1, format_strings2)


    myresult = pd.read_sql_query(sql, con=mydb)

    return myresult




# def get_sims(mydb, list1, list2):
#     list1 = list1.tolist()
#
#     list2 = list2.tolist()
#
#     my_cursor = mydb.cursor()
#     format_strings1 = ','.join(['%s'] * len(list1))
#     format_strings2 = ','.join(['%s'] * len(list2))
#     sql = "select * from similarity where comp_1 in (%s) and comp_2 in (%s)"
#     format_strings1 = format_strings1 % tuple(list1)
#     format_strings2 = format_strings2 % tuple(list2)
#     sql = sql % (format_strings1, format_strings2)
#     my_cursor.execute(sql)
#
#     myresult = my_cursor.fetchall()
#
#     my_cursor.close()
#
#     if len(myresult) != 0:
#
#         myresult = pd.DataFrame(np.array(myresult),
#                                 columns=['id', 'comp_1', 'comp_2', 'sim_resnik', 'sim_lin', 'sim_jc'])
#
#     else:
#         myresult = pd.DataFrame(columns=['id', 'comp_1', 'comp_2', 'sim_resnik', 'sim_lin', 'sim_jc'])
#
#     return myresult


def confirm_all_test_train_similarities(test_items_chebi_id, train_items_for_t_us, scores_by_item, host, user, password,
                                        database, path_to_ontology):
    # test_items_chebi_id = np.insert(test_items_chebi_id, 1, 10000)
    print("items in train", len(train_items_for_t_us))

    # check if all item-item pair was found in the database
    test_train_items_combinations = pd.DataFrame(list(product(test_items_chebi_id, train_items_for_t_us)),
                                                 columns=['l1', 'l2'])

    ss = test_train_items_combinations.l1.isin(
        scores_by_item.comp_1.astype('int64').tolist()) & test_train_items_combinations.l2.isin(
        scores_by_item.comp_2.astype('int64').tolist())

    ss2 = test_train_items_combinations.l2.isin(
        scores_by_item.comp_1.astype('int64').tolist()) & test_train_items_combinations.l1.isin(
        scores_by_item.comp_2.astype('int64').tolist())

    test_train_sim_not_found = test_train_items_combinations[(~ss) & (~ss2)]

    unique_test = test_train_sim_not_found.l1.unique()
    unique_train = test_train_sim_not_found.l2.unique()

    # chebi_ids = np.concatenate((test_train_sim_not_found.l1.unique(), test_train_sim_not_found.l2.unique()))
    # print("items that will be added: ", len(chebi_ids))

    if len(unique_test) > 0:

        calculate_semantic_similarity(unique_test, unique_train, host, user, password, database, path_to_ontology)

    else:
        print("all items in DB")


def get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us):

    sims = get_sims(mydb, test_items_chebi_id, train_items_for_t_us)

    sims_inverse = get_sims(mydb, train_items_for_t_us, test_items_chebi_id)

    sims_inverse = sims_inverse.rename(columns={"comp_1": "comp_2", "comp_2": "comp_1"})

    sims_concat = pd.concat([sims, sims_inverse], axis=0, join='outer', ignore_index=True, sort=False)

    if len(sims_concat) > 0:

        #scores_by_item = sims_concat.groupby(['comp_1']).mean().reset_index()
        scores_by_item = sims_concat

    else:
        scores_by_item = []

    return scores_by_item


def get_all_metrics_by_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                                test_items, k, all_ratings, mydb, host, user, password, database, path_to_ontology):
    metrics_dict = {}
    progress = 0
    users_to_remove = 0
    relevant_items_sum = 0

    test_items_chebi_id = all_ratings[all_ratings.index_item.isin(
        test_items)].item.unique()  ###################### test items chebi id!!!  what i'm rating. Array is equal for all users

    for t_us in test_users:

        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]

        test_ratings_for_t_us = all_ratings_for_t_us[all_ratings_for_t_us.index_item.isin(test_items)]

        train_ratings_for_t_us = all_ratings_for_t_us[~(all_ratings_for_t_us.index_item.isin(test_items))]

        train_items_for_t_us = train_ratings_for_t_us.item.unique()  ####################training items for this user to be used for finding the similarity

        progress += 1
        print(progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r")

        sys.stdout.flush()

        ratings_t_us = ratings_test[ratings_test.user == t_us]

        if np.sum(ratings_t_us.rating) == 0:
            users_to_remove += 1

            continue

        if len(train_ratings_for_t_us) == 0:
            users_to_remove += 1

            continue

        scores_by_item = get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us, host, user, password,
                                           database, path_to_ontology)

        relevant = get_relevants_by_user(test_ratings_for_t_us, 0)

        relevant_items_sum += len(relevant)

        if len(scores_by_item) > 0:

            # item_score = recommendations(model_als, ratings_train_sparse_CF, test_items, t_us)
            # item_score = pd.DataFrame(np.array(item_score), columns=["item", "score"])

            item_score = scores_by_item[['comp_1', 'sim_lin']]
            item_score = item_score.rename(columns={"comp_1": "item", "sim_lin": "score"})
            item_score.item = item_score.item.astype(int)

            user_r = [0.0]
            user_fpr = [0.0]

            for i in range(1, k + 1):

                top_n = get_top_n(item_score, i)

                top_n.item = top_n.item.astype(int)

                topn_real_ratings = get_real_item_rating(top_n, test_ratings_for_t_us).rating

                fpr = false_positive_rate(test_items_chebi_id, relevant, top_n)

                recs = np.array(top_n.item).astype(int)
                P = precision(recs, np.array(relevant))
                R = recall(recs, relevant)
                F = fmeasure(P, R)
                rr = reciprocal_rank(topn_real_ratings)

                user_r.append(R)
                user_fpr.append(fpr)

                nDCG = ndcg_at_k(topn_real_ratings, i, method=0)

                # auc = metrics.auc(user_fpr, user_r)
                auc = get_auc(user_r, user_fpr)

                if len(metrics_dict) != k:
                    metrics_dict.update({'top' + str(i): [P, R, F, fpr, rr, nDCG, auc]})

                else:
                    old = np.array(metrics_dict['top' + str(i)])
                    new = np.array([P, R, F, fpr, rr, nDCG, auc])

                    to_update = old + new

                    metrics_dict.update({'top' + str(i): to_update})

    test_users_size = test_users_size - users_to_remove

    relevant_items_mean = relevant_items_sum / test_users_size

    print("mean of relevant items: ", relevant_items_mean)
    metrics_dict = calculate_dictionary_mean(metrics_dict, float(test_users_size))

    print("n users removed: ", users_to_remove)

    return metrics_dict


def merge_algorithms_scores(item_score_ontology, item_score_implicit, metric):
    """
    calculates the scores for each test item with hybrid algorithm
    :param item_score_ontology: item scoer from CB
    :param item_score_implicit: item score from CF
    :param metric: 1: multiplication of the scores; 2: mean of the scores
    :return: item score dataframe order descending
    """

    merged_item_scores = pd.merge(item_score_implicit, item_score_ontology, on='item')

    if metric == 1:
        merged_item_scores['score'] = merged_item_scores.score_x * merged_item_scores.score_y

    elif metric == 2:
        merged_item_scores['score'] = (merged_item_scores.score_x + merged_item_scores.score_y) / 2

    item_score = merged_item_scores[['item', 'score', 'item_chebi_x']].sort_values(by=['score'], ascending=False)

    return item_score


def hybrid_implicit_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test, model_als,
                             ratings_sparse, test_items, k, train_ratings, all_ratings, original_item_id, mydb, host,
                             user, password, database, path_to_ontology):
    metrics_dict = {}

    progress = 0

    users_to_remove = 0
    relevant_items_sum = 0
    # ssmpy.semantic_base("/mlData/chebi.db")

    test_items_chebi_id = all_ratings[all_ratings.index_item.isin(
        test_items)].item.unique()  ###################### test items chebi id!!!  what i'm rating. Array is equal for all users

    for t_us in test_users:

        progress += 1
        print(progress, ' of ', test_users_size, "cv ", count_cv, "-", count_cv_items, end="\r")

        sys.stdout.flush()

        # implicit
        ratings_t_us = ratings_test[ratings_test.user == t_us]
        train_ratings_t_us = train_ratings[train_ratings.user == t_us]

        # ontology
        all_ratings_for_t_us = all_ratings[all_ratings.index_user == t_us]
        test_ratings_for_t_us = all_ratings_for_t_us[all_ratings_for_t_us.index_item.isin(test_items)]
        train_ratings_for_t_us = all_ratings_for_t_us[~(all_ratings_for_t_us.index_item.isin(test_items))]
        train_items_for_t_us = train_ratings_for_t_us.item.unique()  ####################training items for this user to be user for fiding the similarity

        if np.sum(ratings_t_us.rating) == 0:
            users_to_remove += 1

            continue

        if len(train_ratings_t_us) == 0:
            users_to_remove += 1

            continue

        item_score_implicit = get_score_by_implicit(model_als, ratings_sparse, test_items, t_us)
        item_score_ontology = get_score_by_ontology(mydb, test_items_chebi_id, train_items_for_t_us, host, user,
                                                    password, database, path_to_ontology)

        item_score = merge_algorithms_scores(item_score_ontology, original_item_id, item_score_implicit)

        relevant = get_relevants_by_user(ratings_t_us, 0)
        relevant_items_sum += len(relevant)

        user_r = [0.0]
        user_fpr = [0.0]

        for i in range(1, k + 1):

            top_n = get_top_n(item_score, i)

            top_n.item = top_n.item.astype(int)

            topn_real_ratings = get_real_item_rating(top_n, ratings_t_us).rating

            fpr = false_positive_rate(test_items, relevant, top_n)

            recs = np.array(top_n.item).astype(int)
            P = precision(recs, np.array(relevant))
            R = recall(recs, relevant)
            F = fmeasure(P, R)
            rr = reciprocal_rank(topn_real_ratings)

            user_r.append(R)
            user_fpr.append(fpr)

            nDCG = ndcg_at_k(topn_real_ratings, i, method=0)

            # auc = metrics.auc(user_fpr, user_r)
            auc = get_auc(user_r, user_fpr)

            if len(metrics_dict) != k:
                metrics_dict.update({'top' + str(i): [P, R, F, fpr, rr, nDCG, auc]})

            else:
                old = np.array(metrics_dict['top' + str(i)])
                new = np.array([P, R, F, fpr, rr, nDCG, auc])

                to_update = old + new

                metrics_dict.update({'top' + str(i): to_update})

    test_users_size = test_users_size - users_to_remove

    relevant_items_mean = relevant_items_sum / test_users_size

    print("mean of relevant items: ", relevant_items_mean)
    metrics_dict = calculate_dictionary_mean(metrics_dict, float(test_users_size))
    print(metrics_dict)
    print("n users removed: ", users_to_remove)

    return metrics_dict


def get_score_by_implicit(model, ratings_sparse, test_items, t_us):
    item_score = recommendations(model, ratings_sparse, test_items, t_us)
    item_score = pd.DataFrame(np.array(item_score), columns=["item", "score"])
    item_score.item = item_score.item.astype(int)

    return item_score


def get_score_by_ontology(mydb, test_items_chebi_id, train_items_for_t_us, host, user, password, database,
                          path_to_ontology):
    scores_by_item = get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us, host, user, password, database,
                                       path_to_ontology)

    item_score = scores_by_item[['comp_1', 'sim_lin']]
    item_score = item_score.rename(columns={"comp_1": "item", "sim_lin": "score"})
    item_score.item = item_score.item.astype(int)

    return item_score


def neural_network_recommender():
    return
