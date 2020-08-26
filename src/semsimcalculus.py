import ssmpy
import pandas as pd
import numpy as np
import sys
import mysql.connector
import os
import multiprocessing as mp
from numba import jit, cuda
import configargparse
import time
from sqlalchemy import create_engine



pd.set_option('display.max_columns', None)


def connect_(db_name):
    mydb = mysql.connector.connect(
        host='172.17.0.6',
        user="root",
        password='1234',
        database=db_name
    )

    return mydb


def connect__(host, user, password, database):
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
         use_pure=True
    )

    return mydb

def connect(user, password, database_name):
    engine = create_engine("mysql+pymysql://{user}:{pw}@172.17.0.2/{db}"
                           .format(user=user,
                                   pw=password,
                                   db=database_name))

    return engine


def calculate_sim_it1_it2(it1, it2, e1, host, user, password, database):
    if check_if_pair_exist(host, user, password, database, it1, it2) == False:

        it2_ = "CHEBI_" + str(it2)

        # print("   ", it2_)
        try:
            start = time.time()
            e2 = ssmpy.get_id(it2_)
            items_sim_resnik = ssmpy.ssm_resnik(e1, e2)
            items_sim_lin = ssmpy.ssm_lin(e1, e2)
            items_sim_jc = ssmpy.ssm_jiang_conrath(e1, e2)
            end = time.time()
            print("unique calc: ", end - start)
            insert_row(host, user, password, database, it1.item(), it2.item(), items_sim_resnik, items_sim_lin,
                       items_sim_jc)

            sys.stdout.flush()

        except TypeError:
            print(it1, " or ", it2, " not found.")


    else:
        print(it1, it2, " pair already exists")


@jit(target="cuda")
def calculate_sim_it1_it2_test_gpu(it1, e1, host, user, password, database, chebi_ids):
    for it2 in chebi_ids:
        if check_if_pair_exist(host, user, password, database, it1, it2) == False:

            it2_ = "CHEBI_" + str(it2)

            # print("   ", it2_)
            try:
                e2 = ssmpy.get_id(it2_)
                items_sim_resnik = ssmpy.ssm_resnik(e1, e2)
                items_sim_lin = ssmpy.ssm_lin(e1, e2)
                items_sim_jc = ssmpy.ssm_jiang_conrath(e1, e2)
                insert_row(database, it1.item(), it2.item(), items_sim_resnik, items_sim_lin, items_sim_jc)

                sys.stdout.flush()
            except TypeError:
                print(it1, " or ", it2, " not found.")


        else:
            print(it1, it2, " pair already exists")
            continue


def calculate_semantic_similarity(chebi_ids, train, host, user, password, database, path_to_ontology):
    ssmpy.semantic_base(path_to_ontology)

    print("test size: ", len(chebi_ids))
    print("train size: ", len(train))

    count = 0
    for it1 in chebi_ids:
        it1_ = "CHEBI_" + str(it1)
        print(it1_)
        e1 = ssmpy.get_id(it1_)

        # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(30)

        start = time.time()

        pool.starmap_async(calculate_sim_it1_it2,
                           [(it1, it2, e1, host, user, password, database) for it2 in train]).get()
        end = time.time()
        print(end - start)

        pool.close()

        # mask = np.where(chebi_ids == it1)
        # chebi_ids = np.delete(chebi_ids, mask)
        count += 1
        print(count)


def calculate_semantic_similarity_gpu(chebi_ids, host, user, password, database, path_to_ontology):
    ssmpy.semantic_base(path_to_ontology)

    count = 0
    for it1 in chebi_ids:
        it1_ = "CHEBI_" + str(it1)
        print(it1_)
        e1 = ssmpy.get_id(it1_)

        # pool = mp.Pool(mp.cpu_count())

        calculate_sim_it1_it2_test_gpu(it1, e1, host, user, password, database, chebi_ids)

        mask = np.where(chebi_ids == it1)
        chebi_ids = np.delete(chebi_ids, mask)
        count += 1
        print(count)


# def calculate_semantic_similarity(chebi_ids, db_name):
#     ssmpy.semantic_base("/mlData/chebi.db")
#
#     count = 0
#     for it1 in chebi_ids:
#         it1_ = "CHEBI_" + str(it1)
#         print(it1_)
#         e1 = ssmpy.get_id(it1_)
#
#         for it2 in chebi_ids:
#             it2_ = "CHEBI_" + str(it2)
#             print(count)
#             print("   ", it2_)
#             e2 = ssmpy.get_id(it2_)
#             items_sim_resnik = ssmpy.ssm_resnik(e1, e2)
#             items_sim_lin = ssmpy.ssm_lin(e1, e2)
#             items_sim_jc = ssmpy.ssm_jiang_conrath(e1, e2)
#
#             sys.stdout.flush()
#             insert_row(db_name, it1.item(), it2.item(), items_sim_resnik, items_sim_lin, items_sim_jc)
#             count += 1
#
#             mask = np.where(chebi_ids==it1)
#             chebi_ids = np.delete(chebi_ids, mask)


def get_sim_where_comp(host, user, password, database, it1, it2):
    mydb = connect(host, user, password, database)

    my_cursor = mydb.cursor()
    sql = "select * from similarity where comp_1=%s and comp_2=%s"
    sql = sql % (it1, it2)
    my_cursor.execute(sql)

    my_cursor = my_cursor.fetchall()

    return len(my_cursor)


def check_if_pair_exist(host, user, password, database, it1, it2):
    exist = get_sim_where_comp(host, user, password, database, it1, it2)
    exist_reverse = get_sim_where_comp(host, user, password, database, it2, it1)

    if exist == 0 and exist_reverse == 0:
        return False

    else:

        return True


def insert_row(host, user, password, database, chebi_a, chebi_b, sim_res, sim_l, sim_j):
    mydb = connect(host, user, password, database)

    my_cursor = mydb.cursor()

    sql = "INSERT INTO similarity (comp_1, comp_2, sim_resnik, sim_lin, sim_jc) VALUES (%s,%s,%s,%s,%s)"

    val = (chebi_a, chebi_b, sim_res, sim_l, sim_j)
    my_cursor.execute(sql, val)

    mydb.commit()


def get_chebi_ids(dataset):
    chebi_ids = dataset.item.unique()

    return chebi_ids


def check_database(host, user, passwd, db_name):
    check = False

    mydb = mysql.connector.connect(
        host=host,
        user=user,
        passwd=passwd
    )
    mycursor = mydb.cursor()

    mycursor.execute("SHOW DATABASES")

    for x in mycursor:
        x = x[0].decode("unicode-escape")
        if x == db_name:
            check = True

    if check == False:

        print("will create db")
        mycursor.execute("CREATE DATABASE " + db_name)
        create_table(host, user, passwd, db_name)

    else:
        print("Database already exists")


def create_table(host, user, passwd, db_name):
    mydb = mysql.connector.connect(
        host=host,
        user=user,
        passwd=passwd,
        database=db_name
    )

    mycursor = mydb.cursor()

    # mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

    mycursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    mycursor.execute("DROP TABLE IF EXISTS `similarity`")

    mycursor.execute(
        " CREATE TABLE `similarity` (`id` INT NOT NULL AUTO_INCREMENT, `comp_1` INT NOT NULL,  `comp_2` INT NOT NULL, "
        "`sim_resnik` FLOAT NOT NULL, `sim_lin` FLOAT NOT NULL, `sim_jc` FLOAT NOT NULL, PRIMARY KEY (`id`), "
        "INDEX sim (`comp_1`,`comp_2`) ) ENGINE=InnoDB")

    mycursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    '''
if __name__ == '__main__':

    p = configargparse.ArgParser(default_config_files=['../config/config.ini'])
    p.add('-mc', '--my-config', is_config_file=True, help='alternative config file path')

    p.add("-ds", "--path_to_dataset", required=False, help="path to dataset", type=str)

    p.add("-host", "--host", required=False, help="db host", type=str)
    p.add("-user", "--user", required=False, help="db user", type=str)
    p.add("-pwd", "--password", required=False, help="db password", type=str)
    p.add("-db_name", "--database", required=False, help="db name", type=str)

    p.add("-owl", "--path_to_owl", required=False, help="path to owl ontology", type=str)
    p.add("-db_onto", "--path_to_ontology_db", required=False, help="path to ontology db", type=str)

    options = p.parse_args()

    dataset = options.path_to_dataset

    host = options.host
    user = options.user
    password = options.password
    database = options.database
    path_to_owl = options.path_to_owl
    path_to_ontology = options.path_to_ontology_db

    check_database(host, user, password, database)

    df_dataset = pd.read_csv(dataset, names=['user', 'item', 'rating'], sep=',')

    if os.path.isfile(path_to_ontology) == False:
        print("chebi DB does not exit. Creating...")
        ssmpy.create_semantic_base(path_to_owl, path_to_ontology,
                                   "http://purl.obolibrary.org/obo/",
                                   "http://www.w3.org/2000/01/rdf-schema#subClassOf", "")

    else:
        print("file already exists")

    chebi_ids = get_chebi_ids(df_dataset)
    print(chebi_ids)

    calculate_semantic_similarity(chebi_ids, host, user, password, database, path_to_ontology)

    #calculate_semantic_similarity_gpu(chebi_ids, host, user, password, database, path_to_ontology)
    '''
