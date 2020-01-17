import implicit
from recommender_evaluation import *
from cross_val import *
import ssmpy
import mysql.connector
from itertools import product


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

    # recommendations = model.recommend(16, user_items, N=10)

    ranking_items = model.rank_items(user, user_items, test_items)

    return ranking_items


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


def calculate_semantic_similarity(test_items_chebi, train_items_chebi):
    score_list = []
    count = 0
    for it1 in test_items_chebi:
        it1 = "CHEBI_" + str(it1)
        print(it1)
        e1 = ssmpy.get_id(it1)
        it2_sim = 0
        for it2 in train_items_chebi:
            it2 = "CHEBI_" + str(it2)
            print(it2)
            e2 = ssmpy.get_id(it2)
            it2_sim += ssmpy.ssm_resnik(e1, e2)
            sys.stdout.flush()
            count += 1
        score_list.append(it2_sim / len(train_items_chebi))

    df = pd.DataFrame(test_items_chebi, columns=['item'])
    df['score'] = score_list
    print(df)

    df = df.sort_values(by=['score'], ascending=False)
    print(df)

    return df


def get_sims(mydb, list1, list2):
    list1 = list1.tolist()

    list2 = list2.tolist()

    #mydb = connect(db_name)
    my_cursor = mydb.cursor()
    format_strings1 = ','.join(['%s'] * len(list1))
    format_strings2 = ','.join(['%s'] * len(list2))
    sql = "select * from similarity where comp_1 in (%s) and comp_2 in (%s)"
    format_strings1 = format_strings1 % tuple(list1)
    format_strings2 = format_strings2 % tuple(list2)
    sql = sql % (format_strings1, format_strings2)
    my_cursor.execute(sql)

    myresult = my_cursor.fetchall()

    if len(myresult) != 0:

        myresult = pd.DataFrame(np.array(myresult),
                                columns=['id', 'comp_1', 'comp_2', 'sim_resnik', 'sim_lin', 'sim_jc'])

    else:
        myresult = pd.DataFrame(columns=['id', 'comp_1', 'comp_2', 'sim_resnik', 'sim_lin', 'sim_jc'])

    return myresult


def confirm_all_test_train_similarities(test_items_chebi_id, train_items_for_t_us, scores_by_item):
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

    # if test_train_sim_not_found not empty:


def get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us):
    sims = get_sims(mydb, test_items_chebi_id, train_items_for_t_us)

    sims_inverse = get_sims(mydb, train_items_for_t_us, test_items_chebi_id)

    sims_inverse = sims_inverse.rename(columns={"comp_1": "comp_2", "comp_2": "comp_1"})

    sims_concat = pd.concat([sims, sims_inverse], axis=0, join='outer', ignore_index=True, sort=False)

    scores_by_item = sims_concat.groupby(['comp_1']).mean().sort_values(by=['sim_lin'],
                                                                        ascending=False).reset_index()

    return scores_by_item


def get_all_metrics_by_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                                test_items, k, all_ratings, mydb):
    metrics_dict = {}

    progress = 0

    users_to_remove = 0
    relevant_items_sum = 0
    #ssmpy.semantic_base("/mlData/chebi.db")


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

        scores_by_item = get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us)

        relevant = get_relevants_by_user(test_ratings_for_t_us, 0)

        relevant_items_sum += len(relevant)

        # item_score = recommendations(model_als, ratings_sparse, test_items, t_us)
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
    print(metrics_dict)
    print("n users removed: ", users_to_remove)

    return metrics_dict


def merge_algorithms_scores(item_score_ontology, original_item_id, item_score_implicit):
    item_score_ontology = item_score_ontology.rename(columns={"item": "item_chebi"})
    item_score_ontology["item"] = item_score_ontology["item_chebi"].map(
        original_item_id.set_index('item')["new_index"]).fillna(0)

    merged_item_scores = pd.merge(item_score_implicit, item_score_ontology, on='item')

    merged_item_scores['score'] = merged_item_scores.score_x * merged_item_scores.score_y

    item_score = merged_item_scores[['item', 'score']].sort_values(by=['score'], ascending=False)

    return item_score


def hybrid_implicit_ontology(test_users, test_users_size, count_cv, count_cv_items, ratings_test, model_als,
                             ratings_sparse, test_items, k, train_ratings, all_ratings, original_item_id, mydb):
    metrics_dict = {}

    progress = 0

    users_to_remove = 0
    relevant_items_sum = 0
    #ssmpy.semantic_base("/mlData/chebi.db")

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
        item_score_ontology = get_score_by_ontology(mydb, test_items_chebi_id, train_items_for_t_us)

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


def get_score_by_implicit(model_als, ratings_sparse, test_items, t_us):
    item_score = recommendations(model_als, ratings_sparse, test_items, t_us)
    item_score = pd.DataFrame(np.array(item_score), columns=["item", "score"])
    item_score.item = item_score.item.astype(int)

    return item_score


def get_score_by_ontology(mydb, test_items_chebi_id, train_items_for_t_us):
    scores_by_item = get_score_by_item(mydb, test_items_chebi_id, train_items_for_t_us)

    item_score = scores_by_item[['comp_1', 'sim_lin']]
    item_score = item_score.rename(columns={"comp_1": "item", "sim_lin": "score"})
    item_score.item = item_score.item.astype(int)

    return item_score


def neural_network_recommender():
    return
