import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def upload_dataset(path_csv):
    '''

    :param csvPath: user, item, rating csv file
    :return: user, item, rating pandas dataframe
    '''

    matrix = pd.read_csv(path_csv, sep=",", names=['user', 'item', 'rating'])

    return matrix


def three_columns_matrix_to_csr(matrix):
    '''

    :param matrix: pandas dataframe of user, item, rating
    :return: (item, user) rating sparse matrix
    '''

    print(len(matrix.item.unique()), len(matrix.user.unique()))

    ratings_sparse = coo_matrix((matrix.rating, (matrix.item, matrix.user)))
    # print(ratings_sparse.toarray().shape)

    return ratings_sparse


def save_final_data(data, path_csv):
    df = pd.DataFrame.from_dict(data)
    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(path_csv)


def id_to_index(df):
    """
    maps the values to the lowest consecutive values
    :param df: pandas Dataframe with columns user, item, rating
    :return: pandas Dataframe with the columns index_item and index_user
    """

    original_id_item = df.item
    original_id_user = df.user
    index_item = np.arange(0, len(df.item.unique()))
    index_user = np.arange(0, len(df.user.unique()))

    df_item_index = pd.DataFrame(df.item.unique(), columns=["item"])
    df_item_index["new_index"] = index_item
    df_user_index = pd.DataFrame(df.user.unique(), columns=["user"])
    df_user_index["new_index"] = index_user

    df["index_item"] = df["item"].map(df_item_index.set_index('item')["new_index"]).fillna(0)
    df["index_user"] = df["user"].map(df_user_index.set_index('user')["new_index"]).fillna(0)
    print(df)

    return df, df_item_index, df_user_index
