import numpy as np
import pandas as pd
import configargparse
import implicit
import scipy
import sklearn
from scipy import sparse
from scipy.sparse import coo_matrix
import sys
from data import *
from algorithms import *
from recommender_evaluation import *
from cross_val import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
import cffi
import ssmpy
import os
import gc

if __name__ == '__main__':

    print(implicit.__version__)
    print(np.__version__)
    # print(configargparse.__version__)
    print(pd.__version__)
    print(scipy.__version__)
    print(sklearn.__version__)
    print(cffi.__version__)

    # ratings = pd.read_csv('mlData/u.data', sep='\t',
    #                     names=['user', 'item', 'rating', 'timestamp'])

    p = configargparse.ArgParser(default_config_files=['../config/config.ini'])
    p.add('-mc', '--my-config', is_config_file=True, help='alternative config file path')

    p.add("-ds", "--path_to_dataset", required=False, help="path to dataset", type=str)

    p.add("-cv", "--cv", required=False, help="cross validation folds",
          type=int)

    p.add("-k", "--topk", required=False, help="k for topk", type=int)

    p.add("-host", "--host", required=False, help="db host", type=str)
    p.add("-user", "--user", required=False, help="db user", type=str)
    p.add("-pwd", "--password", required=False, help="db password", type=str)
    p.add("-db_name", "--database", required=False, help="db name", type=str)


    options = p.parse_args()

    path_to_dataset = options.path_to_dataset

    k = options.topk

    cv_folds = options.cv

    host = options.host
    user = options.user
    password = options.password
    database = options.database

    ##################################################################################################################
    # connect db
    mydb = connect(host, user, password, database)

    ################################################################################################################
    # get the dataset in user-item-rating format
    ratings_original = upload_dataset(path_to_dataset)

    ratings2, original_item_id, original_user_id = id_to_index(ratings_original)  # are not unique

    ratings = ratings2.drop(columns=["user", "item"])
    ratings = ratings.rename(columns={"index_item": "item", "index_user": "user"})

    users_size = len(ratings.user.unique())
    items_size = len(ratings.item.unique())
    shuffle_users = get_shuffle_users(ratings)
    shuffle_items = get_shuffle_items(ratings)

    '''
    ## test Dishin

    if os.path.isfile('/mlData/chebi.db') == False:
        print("do not exists")
        ssmpy.create_semantic_base("/mlData/chebi_lite.owl", "/mlData/chebi.db",
                                   "http://purl.obolibrary.org/obo/",
                                   "http://www.w3.org/2000/01/rdf-schema#subClassOf", "")

    else:
        print("file already exists")

    ###############################################################################################################

    '''


    count_cv = 0

    # dictionary for saving the results of each cross validation for each model
    metrics_all_cv_als = {}
    metrics_all_cv_bayes = {}
    metrics_all_cv_ontology = {}
    metrics_all_cv_hyb_onto_bayes = {}
    metrics_all_cv_hyb_onto_als = {}

    for test_users in np.array_split(shuffle_users, cv_folds):


        test_users_size = len(test_users)
        print("number of test users: ", test_users_size)
        sys.stdout.flush()

        count_cv_items = 0
        for test_items in np.array_split(shuffle_items, cv_folds):

            # models to be used
            model_bayes = implicit.bpr.BayesianPersonalizedRanking(factors=150, num_threads=10, use_gpu=False)
            model_als = implicit.als.AlternatingLeastSquares(factors=150, num_threads=10, use_gpu=False)

            test_items_size = len(test_items)
            print("number of test items: ", test_items_size)
            sys.stdout.flush()

            ### prepare the data for implicit models
            ratings_test, ratings_train = prepare_train_test(ratings, test_users, test_items)
            #ratings_test, ratings_train = prepare_train_test_(ratings, test_users, test_items) # removes all the ratings from the training set for the test_items # does not work

            test_items = check_items_in_model(ratings_train.item.unique(), test_items)
            ratings_sparse = three_columns_matrix_to_csr(ratings_train)  # item, user, rating
            ###
            # print(ratings_sparse.toarray().shape)
            sys.stdout.flush()

            ####################################################################################################
            # fit models



            model_als.fit(ratings_sparse)
            model_bayes.fit(ratings_sparse)

            metrics_dict = get_all_metrics_by_cv_implicit(test_users, test_users_size, count_cv, count_cv_items,
                                                          ratings_test,
                                                          model_als, ratings_sparse, test_items, k, ratings_train)
            metrics_dict_bayes = get_all_metrics_by_cv_implicit(test_users, test_users_size, count_cv, count_cv_items,
                                                                ratings_test,
                                                                model_bayes, ratings_sparse, test_items, k,
                                                                ratings_train)

            metrics_dict_hyb_onto_bayes = hybrid_implicit_ontology(test_users, test_users_size, count_cv,
                                                                   count_cv_items, ratings_test, model_bayes,
                                                                   ratings_sparse, test_items, k, ratings_train,
                                                                   ratings2, original_item_id, mydb)

            metrics_dict_hyb_onto_als = hybrid_implicit_ontology(test_users, test_users_size, count_cv, count_cv_items,
                                                                 ratings_test, model_als,
                                                                 ratings_sparse, test_items, k, ratings_train, ratings2,
                                                                 original_item_id, mydb)
                                                                 


            metrics_dict_onto = get_all_metrics_by_ontology(test_users, test_users_size, count_cv, count_cv_items,
                                                            ratings_test,
                                                            test_items, k, ratings2, mydb)


            #### als
            if len(metrics_all_cv_als) == 0:
                metrics_all_cv_als = metrics_dict

            else:

                metrics_all_cv_als = {key: metrics_all_cv_als.get(key, 0) + metrics_dict.get(key, 0) for key in
                                      set(metrics_all_cv_als) | set(metrics_dict)}

            ##### bayes
            if len(metrics_all_cv_bayes) == 0:
                metrics_all_cv_bayes = metrics_dict_bayes

            else:

                metrics_all_cv_bayes = {key: metrics_all_cv_bayes.get(key, 0) + metrics_dict_bayes.get(key, 0) for key
                                        in
                                        set(metrics_all_cv_bayes) | set(metrics_dict_bayes)}

            

            ##### onto_bayes
            if len(metrics_all_cv_hyb_onto_bayes) == 0:
                metrics_all_cv_hyb_onto_bayes = metrics_dict_hyb_onto_bayes

            else:

                metrics_all_cv_hyb_onto_bayes = {
                    key: metrics_all_cv_hyb_onto_bayes.get(key, 0) + metrics_dict_hyb_onto_bayes.get(key, 0)
                    for key in set(metrics_all_cv_hyb_onto_bayes) | set(metrics_dict_hyb_onto_bayes)}

            ##### onto_als
            if len(metrics_all_cv_hyb_onto_als) == 0:
                metrics_all_cv_hyb_onto_als = metrics_dict_hyb_onto_als

            else:

                metrics_all_cv_hyb_onto_als = {
                    key: metrics_all_cv_hyb_onto_als.get(key, 0) + metrics_dict_hyb_onto_als.get(key, 0)
                    for key in set(metrics_all_cv_hyb_onto_als) | set(metrics_dict_hyb_onto_als)}


            ##### onto
            if len(metrics_all_cv_ontology) == 0:
                metrics_all_cv_ontology = metrics_dict_onto

            else:

                metrics_all_cv_ontology = {key: metrics_all_cv_ontology.get(key, 0) + metrics_dict_onto.get(key, 0)
                                           for key in set(metrics_all_cv_ontology) | set(metrics_dict_onto)}

            sys.stdout.flush()
            count_cv_items += 1

            del model_bayes
            del model_als
            gc.collect()
        count_cv += 1

    metrics_all_cv_als = calculate_dictionary_mean(metrics_all_cv_als, float(cv_folds * cv_folds))
    metrics_all_cv_bayes = calculate_dictionary_mean(metrics_all_cv_bayes, float(cv_folds * cv_folds))
    metrics_all_cv_ontology = calculate_dictionary_mean(metrics_all_cv_ontology, float(cv_folds * cv_folds))
    metrics_all_cv_hyb_onto_bayes = calculate_dictionary_mean(metrics_all_cv_hyb_onto_bayes, float(cv_folds * cv_folds))
    metrics_all_cv_hyb_onto_als = calculate_dictionary_mean(metrics_all_cv_hyb_onto_als, float(cv_folds * cv_folds))

    save_final_data(metrics_all_cv_als, "/mlData/cherRM_final_results_als.csv")
    save_final_data(metrics_all_cv_bayes, "/mlData/cherRM_final_results_bayes.csv")
    save_final_data(metrics_all_cv_ontology, "/mlData/cherRM_final_results_onto.csv")
    save_final_data(metrics_all_cv_hyb_onto_bayes, "/mlData/cherRM_final_results_onto_bayes.csv")
    save_final_data(metrics_all_cv_hyb_onto_als, "/mlData/cherRM_final_results_onto_als.csv")
