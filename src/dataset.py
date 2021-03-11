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
# @author Matilde Pato                                                        #  
# @email: matilde.pato@gmail.com                                              #
# ISEL - IPL and Lasige - FCUL                                                #
# @date: 12 Feb 2021                                                          #
# @version: 1.0                                                               #  
# @last update:                                                               #   
#                                                                             #  
# This file must be adapted to the dataset format, at the end you must ensure #
# the dataframe has 3 columns with <user, item, rating>                       #  
###############################################################################
# 

import pandas as pd
import numpy as np

def upload_dataset(csv_path, name_prefix):
    '''

    :param csv_path: <user, item, rating, ... > csv file
    :param name_prefix: prefix of the concepts to be extracted from the ontology
    :type name_prefix: string
    :return: user, item, rating pandas dataframe
    '''
       
    matrix = pd.read_csv( csv_path, sep=',' )
    
    if ( len(matrix.columns) > 3 ):
        matrix.columns = ['user_name', 'item', 'rating', 'user', 'item_label']
        matrix = matrix[['user', 'item', 'rating']]
    else:
        matrix.columns = ['user', 'item', 'rating']

    if ( matrix.dtypes['item'] == np.object ):
        # filter rows for specific ontology
        matrix = matrix[matrix['item'].astype(str).str.startswith( name_prefix )]
        # filter by number of users
        #matrix = matrix.groupby( 'user' ).filter( lambda x: len( x ) > 19 )
        # remove acronym of ontology and convert as int
        matrix['item'] = matrix['item'].str.replace( name_prefix, '' ).astype( int )
    
    return matrix