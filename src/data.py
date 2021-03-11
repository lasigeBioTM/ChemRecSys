###############################################################################
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License"); you may     #
# not use this file except in compliance with the License. You may obtain a   #
# copy of the License at http://www.apache.org/licenses/LICENSE-2.0           #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
###############################################################################
#                                                                             #
# @author: Márcia Barros                                                      #
# @email: marcia.c.a.barros@gmail.com                                         #
# @date: 17 Jan 2020                                                          #
# @version: 1.0                                                               #
# Lasige - FCUL                                                               #
#                                                                             #
# @last update:                                                               #
#   version 1.1: 12 Feb 2021                                                  #
#   (author: Matilde Pato, matilde.pato@gmail.com)                            #
#                                                                             #
#                                                                             #
###############################################################################
#
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix

# ----------------------------------------------------------------------------------------------------- #

def three_columns_matrix_to_csr(matrix):
    """

    :param matrix: pandas dataframe of user, item, rating
    :return: (item, user) rating sparse matrix
    """

    print( len( matrix.index_item.unique() ), len( matrix.index_user.unique() ) )
    ratings_sparse = coo_matrix( (matrix.rating, (matrix.index_item, matrix.index_user)) )
    return ratings_sparse

# ----------------------------------------------------------------------------------------------------- #

def save_final_data(data, path_csv):
    """
    Save data to csv file
    :param data: pandas Dataframe with columns <user, item, rating>
    :param path_csv: path to the csv file
    :return
    """
    df = pd.DataFrame.from_dict( data )
    df = df.reindex( sorted( df.columns ), axis=1 )
    df.to_csv( path_csv )

# ----------------------------------------------------------------------------------------------------- #

def create_directory(path):
    """
    Create directory to csv file
    :param path: path to save data
    :return
    """
    return Path(path).mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------- #

def id_to_index(df):
    """
    maps the values to the lowest consecutive values
    :param df: pandas Dataframe with columns user, item, rating
    :return: pandas Dataframe with the columns index_item and index_user
    """
    index_item = np.arange( 0, len( df.item.unique() ) )
    index_user = np.arange( 0, len( df.user.unique() ) )

    df_item_index = pd.DataFrame( df.item.unique(), columns=["item"] )
    df_item_index["new_index"] = index_item
    df_user_index = pd.DataFrame( df.user.unique(), columns=["user"] )
    df_user_index["new_index"] = index_user

    df["index_item"] = df["item"].map( df_item_index.set_index( 'item' )["new_index"] ).fillna( 0 )
    df["index_user"] = df["user"].map( df_user_index.set_index( 'user' )["new_index"] ).fillna( 0 )

    return df, df_item_index, df_user_index
