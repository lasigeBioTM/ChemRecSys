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
import cffi
from semsimcalculus import *
import ssmpy

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
    p.add("-n", "--n", required=False, help="n most similar items", type=int)

    p.add("-host", "--host", required=False, help="db host", type=str)
    p.add("-user", "--user", required=False, help="db user", type=str)
    p.add("-pwd", "--password", required=False, help="db password", type=str)
    p.add("-db_name", "--database", required=False, help="db name", type=str)
    p.add("-owl", "--path_to_owl", required=False, help="path to owl ontology", type=str)
    p.add("-db_onto", "--path_to_ontology_db", required=False, help="path to ontology db", type=str)

    options = p.parse_args()

    path_to_dataset = options.path_to_dataset

    k = options.topk
    n = options.n

    cv_folds = options.cv

    host = options.host
    user = options.user
    password = options.password
    database = options.database
    path_to_owl = options.path_to_owl
    path_to_ontology = options.path_to_ontology_db

    #################################################################################################################

    df_dataset = pd.read_csv(path_to_dataset, names=['user', 'item', 'rating'], sep=',')

    if os.path.isfile(path_to_ontology) == False:
        print("chebi DB does not exit. Creating...")
        ssmpy.create_semantic_base(path_to_owl, path_to_ontology,
                                   "http://purl.obolibrary.org/obo/",
                                   "http://www.w3.org/2000/01/rdf-schema#subClassOf", "")

    else:
        print("file already exists")

    # check_database(host, user, password, database)

    #################################################################################################################

    ##################################################################################################################
    # connect db
    mydb = connect( user, password, database)

    ################################################################################################################
    # get the dataset in user-item-rating format
    ratings_original = upload_dataset(path_to_dataset)

    ratings2, original_item_id, original_user_id = id_to_index(ratings_original)  # are not unique


    # ratings = ratings2.drop(columns=["user", "item"])
    # ratings = ratings.rename(columns={"index_item": "item", "index_user": "user"})

    users_size = len(ratings2.index_user.unique())
    items_size = len(ratings2.index_item.unique())
    shuffle_users = get_shuffle_users(ratings2)
    shuffle_items = get_shuffle_items(ratings2)

    count_cv = 0

    # dictionary for saving the results of each cross validation for each model
    metrics_all_cv_dict_onto_lin = {}
    metrics_all_cv_dict_onto_resnik = {}
    metrics_all_cv_dict_onto_jc = {}
    metrics_all_cv_dict_ALS = {}
    metrics_all_cv_dict_BPR = {}
    metrics_all_cv_dict_ALS_ONTO_lin_metric1 = {}
    metrics_all_cv_dict_ALS_ONTO_resnik_metric1 = {}
    metrics_all_cv_dict_ALS_ONTO_jc_metric1 = {}
    metrics_all_cv_dict_BPR_ONTO_lin_metric1 = {}
    metrics_all_cv_dict_BPR_ONTO_resnik_metric1 = {}
    metrics_all_cv_dict_BPR_ONTO_jc_metric1 = {}
    metrics_all_cv_dict_ALS_ONTO_lin_metric2 = {}
    metrics_all_cv_dict_ALS_ONTO_resnik_metric2 = {}
    metrics_all_cv_dict_ALS_ONTO_jc_metric2 = {}
    metrics_all_cv_dict_BPR_ONTO_lin_metric2 = {}
    metrics_all_cv_dict_BPR_ONTO_resnik_metric2 = {}
    metrics_all_cv_dict_BPR_ONTO_jc_metric2 = {}

    for test_users in np.array_split(shuffle_users, cv_folds):

        test_users_size = len(test_users)
        print("number of test users: ", test_users_size)
        sys.stdout.flush()

        count_cv_items = 0
        for test_items in np.array_split(shuffle_items, cv_folds):
            # models to be used

            test_items_size = len(test_items)
            print("number of test items: ", test_items_size)
            sys.stdout.flush()

            ### prepare the data for implicit models
            ratings_test, ratings_train = prepare_train_test(ratings2, test_users, test_items)

            # ratings_test, ratings_train = prepare_train_test_(ratings, test_users, test_items) # removes all the ratings from the training set for the test_items # does not work

            test_items = check_items_in_model(ratings_train.index_item.unique(), test_items)
            ratings_sparse = three_columns_matrix_to_csr(ratings_train)  # item, user, rating

            metrics_cv_dict_onto_lin, metrics_cv_dict_onto_resnik, metrics_cv_dict_onto_jc, metrics_cv_dict_ALS, \
            metrics_cv_dict_BPR, metrics_cv_dict_ALS_ONTO_lin_metric1, metrics_cv_dict_ALS_ONTO_resnik_metric1, \
            metrics_cv_dict_ALS_ONTO_jc_metric1, metrics_cv_dict_BPR_ONTO_lin_metric1, metrics_cv_dict_BPR_ONTO_resnik_metric1, \
            metrics_cv_dict_BPR_ONTO_jc_metric1, metrics_cv_dict_ALS_ONTO_lin_metric2, metrics_cv_dict_ALS_ONTO_resnik_metric2, \
            metrics_cv_dict_ALS_ONTO_jc_metric2, metrics_cv_dict_BPR_ONTO_lin_metric2, metrics_cv_dict_BPR_ONTO_resnik_metric2, \
            metrics_cv_dict_BPR_ONTO_jc_metric2 = get_evaluation(
                test_users, test_users_size, count_cv, count_cv_items, ratings_test,
                ratings_sparse, test_items, k, ratings2, original_item_id, mydb, n)

            # add to dictionary
            metrics_all_cv_dict_onto_lin = add_dict(metrics_all_cv_dict_onto_lin, metrics_cv_dict_onto_lin, count_cv,
                                                    count_cv_items)
            metrics_all_cv_dict_onto_resnik = add_dict(metrics_all_cv_dict_onto_resnik, metrics_cv_dict_onto_resnik,
                                                       count_cv,
                                                       count_cv_items)
            metrics_all_cv_dict_onto_jc = add_dict(metrics_all_cv_dict_onto_jc, metrics_cv_dict_onto_jc, count_cv,
                                                   count_cv_items)
            metrics_all_cv_dict_ALS = add_dict(metrics_all_cv_dict_ALS, metrics_cv_dict_ALS, count_cv, count_cv_items)
            metrics_all_cv_dict_BPR = add_dict(metrics_all_cv_dict_BPR, metrics_cv_dict_BPR, count_cv, count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_lin_metric1 = add_dict(metrics_all_cv_dict_ALS_ONTO_lin_metric1,
                                                                metrics_cv_dict_ALS_ONTO_lin_metric1, count_cv,
                                                                count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_resnik_metric1 = add_dict(metrics_all_cv_dict_ALS_ONTO_resnik_metric1,
                                                                   metrics_cv_dict_ALS_ONTO_resnik_metric1, count_cv,
                                                                   count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_jc_metric1 = add_dict(metrics_all_cv_dict_ALS_ONTO_jc_metric1,
                                                               metrics_cv_dict_ALS_ONTO_jc_metric1, count_cv,
                                                               count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_lin_metric1 = add_dict(metrics_all_cv_dict_BPR_ONTO_lin_metric1,
                                                                metrics_cv_dict_BPR_ONTO_lin_metric1, count_cv,
                                                                count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_resnik_metric1 = add_dict(metrics_all_cv_dict_BPR_ONTO_resnik_metric1,
                                                                   metrics_cv_dict_BPR_ONTO_resnik_metric1, count_cv,
                                                                   count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_jc_metric1 = add_dict(metrics_all_cv_dict_BPR_ONTO_jc_metric1,
                                                               metrics_cv_dict_BPR_ONTO_jc_metric1, count_cv,
                                                               count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_lin_metric2 = add_dict(metrics_all_cv_dict_ALS_ONTO_lin_metric2,
                                                                metrics_cv_dict_ALS_ONTO_lin_metric2, count_cv,
                                                                count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_resnik_metric2 = add_dict(metrics_all_cv_dict_ALS_ONTO_resnik_metric2,
                                                                   metrics_cv_dict_ALS_ONTO_resnik_metric2, count_cv,
                                                                   count_cv_items)
            metrics_all_cv_dict_ALS_ONTO_jc_metric2 = add_dict(metrics_all_cv_dict_ALS_ONTO_jc_metric2,
                                                               metrics_cv_dict_ALS_ONTO_jc_metric2, count_cv,
                                                               count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_lin_metric2 = add_dict(metrics_all_cv_dict_BPR_ONTO_lin_metric2,
                                                                metrics_cv_dict_BPR_ONTO_lin_metric2, count_cv,
                                                                count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_resnik_metric2 = add_dict(metrics_all_cv_dict_BPR_ONTO_resnik_metric2,
                                                                   metrics_cv_dict_BPR_ONTO_resnik_metric2, count_cv,
                                                                   count_cv_items)
            metrics_all_cv_dict_BPR_ONTO_jc_metric2 = add_dict(metrics_all_cv_dict_BPR_ONTO_jc_metric2,
                                                               metrics_cv_dict_BPR_ONTO_jc_metric2, count_cv,
                                                               count_cv_items)

            sys.stdout.flush()
            count_cv_items += 1

        count_cv += 1

    # calculates mean
    metrics_all_cv_dict_onto_lin = calculate_dictionary_mean(metrics_all_cv_dict_onto_lin, float(cv_folds * cv_folds))
    metrics_all_cv_dict_onto_resnik = calculate_dictionary_mean(metrics_all_cv_dict_onto_resnik,
                                                                float(cv_folds * cv_folds))
    metrics_all_cv_dict_onto_jc = calculate_dictionary_mean(metrics_all_cv_dict_onto_jc, float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS = calculate_dictionary_mean(metrics_all_cv_dict_ALS, float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR = calculate_dictionary_mean(metrics_all_cv_dict_BPR, float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_lin_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_lin_metric1,
                                                                         float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_resnik_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_resnik_metric1,
                                                                            float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_jc_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_jc_metric1,
                                                                        float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_lin_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_lin_metric1,
                                                                         float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_resnik_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_resnik_metric1,
                                                                            float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_jc_metric1 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_jc_metric1,
                                                                        float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_lin_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_lin_metric2,
                                                                         float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_resnik_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_resnik_metric2,
                                                                            float(cv_folds * cv_folds))
    metrics_all_cv_dict_ALS_ONTO_jc_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_ALS_ONTO_jc_metric2,
                                                                        float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_lin_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_lin_metric2,
                                                                         float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_resnik_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_resnik_metric2,
                                                                            float(cv_folds * cv_folds))
    metrics_all_cv_dict_BPR_ONTO_jc_metric2 = calculate_dictionary_mean(metrics_all_cv_dict_BPR_ONTO_jc_metric2,
                                                                        float(cv_folds * cv_folds))

    #save to file
    save_final_data(metrics_all_cv_dict_onto_lin, "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_onto_lin.csv")
    save_final_data(metrics_all_cv_dict_onto_resnik, "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_onto_resnik.csv")
    save_final_data(metrics_all_cv_dict_onto_jc, "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_onto_jc.csv")
    save_final_data(metrics_all_cv_dict_ALS, "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS.csv")
    save_final_data(metrics_all_cv_dict_BPR, "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_lin_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_lin_metric1.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_resnik_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_resnik_metric1.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_jc_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_jc_metric1.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_lin_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR_ONTO_lin_metric1.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_resnik_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR_ONTO_resnik_metric1.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_jc_metric1,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR_ONTO_jc_metric1.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_lin_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_lin_metric2.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_resnik_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_resnik_metric2.csv")
    save_final_data(metrics_all_cv_dict_ALS_ONTO_jc_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_ALS_ONTO_jc_metric2.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_lin_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR_ONTO_lin_metric2.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_resnik_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_dict_BPR_ONTO_resnik_metric2.csv")
    save_final_data(metrics_all_cv_dict_BPR_ONTO_jc_metric2,
                    "/mlData/results_nfolds" + str(cv_folds) + "_nsimilar_" +  str(n) + "_BPR_ONTO_jc_metric2.csv")
