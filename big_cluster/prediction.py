# %%
import os,sys,inspect
currentdir = "D:/GPN_KIP-master/big_cluster"
# parentdir = os.path.dirname(currentdir)
parentdir = "D:/GPN_KIP-master"
sys.path.insert(0,parentdir)

# %%
import pandas as pd
import numpy as np
from preprocess.discretization import get_nodes_type, discretization
import matplotlib.pyplot as plt
from bayesian.train_bn import structure_learning, parameter_learning
from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from bayesian.sampling import generate_synthetics, predict
from visualization.visualization import draw_BN
from preprocess.graph import edges_to_dict
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from copy import copy

import time

import json

import traceback


import random
import pathlib


# %%
import json
from external.libpgm.graphskeleton import GraphSkeleton
import pandas as pd
import random
import numpy as np
from bayesian.train_bn import structure_learning, parameter_learning
from preprocess.discretization import get_nodes_type, discretization, code_categories, get_nodes_sign
import time
from copy import copy
import traceback
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import cdist

# %%
def su_dist(x, y): 
    return 1.0 - normalized_mutual_info_score(x, y)

# %%
def zip_code(data_discrete: pd.DataFrame, cluster_columns: list, alpha: float = 0.95, low_limit: int = 5):
    group = copy(data_discrete).groupby(cluster_columns)
    comb = group.size().reset_index().rename(columns={0:'count'})
    comb['count'] = comb['count'] / len(data_discrete)
    comb.sort_values(by=['count'], inplace=True, ascending=False)
    sum = 0.0
    ind = 0
    while (sum < alpha) & (comb.iloc[ind]['count'] >= low_limit / len(data_discrete)):
    #while (sum < alpha):
        sum += comb.iloc[ind]['count']
        ind += 1
        if ind == len(comb):
            break

    x, _ = code_categories(comb, 'onehot', cluster_columns)
    x = x.values
    if ind < len(comb):
        dists = cdist(x[ind:, :], x[:ind, :], metric='hamming')
        neighbors_map = [list(np.where(row == row.min())[0]) for row in dists]
        neighbors_map = [args[-1] for args in neighbors_map]
        better = [i if (i < ind) else neighbors_map[i - ind] for i in range(len(comb))]
    else:
        better = [i for i in range(len(comb))]

    code_col = [None for _ in range(len(data_discrete))]
    for i in range(len(comb)):
        merged_values = tuple(comb[cluster_columns].iloc[i])
        for j in group.indices[merged_values]:
            code_col[j] = better[i]

    return code_col

# %%
def zip_code_hamm(data_discrete: pd.DataFrame, cluster_columns: list, alpha: float = 0.95):
    group = copy(data_discrete).groupby(cluster_columns, as_index=True) 
    x = list(group.groups.keys())
    if len(x) > 1:
        dists = cdist(x, x, metric='hamming')
        model = AgglomerativeClustering(distance_threshold=alpha, n_clusters=None, affinity='precomputed', linkage='single')
        model = model.fit_predict(dists)
    else:
        model = [0]
    
    code_col = [None for _ in range(len(data_discrete))]
    for i in range(len(data_discrete)):
        merged_values = tuple(data_discrete[cluster_columns].iloc[i])
        code_col[i] = model[x.index(merged_values)]
    return code_col

# %%
name = 'hepar2'
bad_option = ['k2', 'K2', '_extra']

# %%
file_list = []
for root, dirs, files in os.walk(f"{parentdir}/models/parameter_bn/"):
    for file in files:
        if (name in file) and ('_simple' not in file) and ('coded' in file or 'uncoded' in file or 'true' in file) and all(opt not in file for opt in bad_option):
            file_list.append(file)

# %%
data = pd.read_csv(f'{currentdir}/my_{name}.csv', index_col='Unnamed: 0')
# data = pd.read_csv(f'{currentdir}/my_{name}.csv')

data.reset_index(drop=True, inplace=True)

col_inter = list(data.columns)
data_save=data[col_inter]
col_new = col_inter

# %%
data = data_save[col_new]
data_types = get_nodes_type(data)
colums_for_code = []
columns_for_disc = []
for c in data.columns:
    if data_types[c] == 'disc':
        colums_for_code.append(c)
    else:
        columns_for_disc.append(c)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)


# %%
data_coded, label_coder = code_categories(data, 'label', colums_for_code)
if len(columns_for_disc) > 0:
    data_discrete, coder = discretization(data_coded, 'equal_frequency', columns_for_disc)
else:
    data_discrete = data_coded
column = list(data.columns)
D = [[0.0 for i in column] for _ in column]
for i, var1 in enumerate(column):
    for j, var2 in enumerate(column):
        D[i][j] = su_dist(data_discrete[var1].values, data_discrete[var2].values)

data_discrete = data_discrete.astype(str)

# %%
df = pd.DataFrame(columns=['file'] + list(data.columns))
df.to_csv(f'{currentdir}/{name}_acc.csv', index=False)
df = pd.read_csv(f'{currentdir}/{name}_acc.csv')


for file in file_list:
    with open (f'{parentdir}/models/structure_bn/{file}', 'r') as f:
        old_data = json.load(f)
    for key in data_discrete.columns:
        old_data['V'] = [node.replace(f'\'{key}\'', f'{key}') for node in old_data['V']]
        old_data['E'] = [[e.replace(f'\'{key}\'', f'{key}') for e in edge] for edge in old_data['E']]
    save_structure(old_data, file.replace('.txt', ''))
   

    with open (f'{parentdir}/models/parameter_bn/{file}', 'r') as f:
        old_data = json.load(f)
    for key in data_discrete.columns:
        for node, val in old_data['Vdata'].items():
            if old_data['Vdata'][node]['parents']:
                old_data['Vdata'][node]['parents'] = [par.replace(f'\'{key}\'', f'{key}') for par in old_data['Vdata'][node]['parents']]
            if old_data['Vdata'][node]['children']:
                old_data['Vdata'][node]['children'] = [par.replace(f'\'{key}\'', f'{key}') for par in old_data['Vdata'][node]['children']]

        node_list = list(old_data['Vdata'].keys())
        for node in node_list:
            if f'\'{key}\'' in node:
                old_data['Vdata'][node.replace(f'\'{key}\'', f'{key}') ] = old_data['Vdata'][node]
                del old_data['Vdata'][node]
    save_params(old_data, file.replace('.txt', ''))

# %%
for file in file_list:
# for file in [file_list[10]]:
    try:
        
        bn = read_structure(file.replace('.txt', ''))
        
        
        dict_net = dict()
        
        if ('true' in file) or ('uncoded' in file):
            params = read_params(file.replace('.txt', ''))
            bn_h = HyBayesianNetwork(bn, params)

            result_dict = dict()
            for key in data_discrete.columns:
                res = 0
                for k in range(len(data_discrete)):
                    row = data_discrete.iloc[k].to_dict()
                    evidence = copy(row)
                    del evidence[key]
                    sample = predict(bn_h, 'simple', 100, evidence=evidence, regime='mode')
                    if sample[key] == row[key]:
                        res += 1
                res = res/len(data_discrete)
                result_dict[key] = res
            df = df.append({'file': file,  **result_dict}, ignore_index=True)
            df.to_csv(f'{currentdir}/{name}_acc.csv', index=False)
            df = pd.read_csv(f'{currentdir}/{name}_acc.csv')
                    
        elif '_simple' not in file:
            if file.replace('.txt', '_simple.txt') not in file_list:
                simple_nodes = [node for node in bn.V if ')_' not in node]
                simple_edges = [edge for edge in bn.E if all([')_' not in e for e in edge])]
                save_structure({'V': simple_nodes, 'E': simple_edges}, file.replace('.txt', '_simple'))

            bn_simple = read_structure(file.replace('.txt', '_simple'))

            params = read_params(file.replace('.txt', ''))
            params_simple = read_params(file.replace('.txt', '_simple'))
            


            bn_h = HyBayesianNetwork(bn, params)
            bn_h_simple = HyBayesianNetwork(bn_simple, params_simple)
           
            
            att = file.replace('.txt', '').split('_')
            large = len(col_new)
            alpha = float('0.' + att[2])
            option = att[3]
            size = att[4]
            
            res_dict = {}
            time_start = time.time()
            model = AgglomerativeClustering(distance_threshold=alpha, n_clusters=None, affinity='precomputed', linkage='single')
            model = model.fit_predict(D)
        

            for i, val in enumerate(model):
                if val in res_dict:
                    res_dict[val].append(column[i])
                else:
                    res_dict[val] = [column[i]]

            ind_dict = {}
            for i, val in enumerate(model):
                if val in ind_dict:
                    ind_dict[val].append(i)
                else:
                    ind_dict[val] = [i]
            count = 0
            flag_all_var = False
            for key, val in ind_dict.items():
                if len(val) > 1:
                    count += 1
                if len(val) == len(col_new):
                    flag_all_var = True
            if (count == 0) or flag_all_var:
                continue
            
            df2 = pd.DataFrame()
            try:
                for key, val in res_dict.items():
                    if len(val) > 1:
                        if option == 'mostly':
                            code_col = zip_code(data_discrete, val)
                        elif option == 'hamming':
                            code_col = zip_code_hamm(data_discrete, val)
                        df2[tuple(val)] = code_col
                    else:
                        df2[tuple(val)] = data_discrete[val[0]]
            except:
                continue
            
           
            # %%
            tuple_columns = df2.columns
            rename_dict = {tuple_columns[i]: i  for i in range(len(tuple_columns))}
            df2.rename(rename_dict, axis = 1, inplace=True)
          

            # %%
            rerename_dict = {val: str(key) for key, val in rename_dict.items()}
          
            
            data_discrete_new = copy(data_discrete)
        
            for node in bn.V:
                if ')_out' in node:
                    node_corr = node
                    node_corr = node_corr.replace('(', '(\"').replace(',', '\",').replace(', ', ', \"').replace(')', '\")')
                    
                    node_list = json.loads(node_corr.split(")_")[0].replace('(', '[') + ']')
                    for key, val in rename_dict.items():
                        if set(key) == set(node_list):
                            data_discrete_new[node.split(")_")[0] +')_out'] = df2[val]
                            data_discrete_new[node.split(")_")[0] +')_in'] = df2[val]
            
           
            result_dict = dict()
            for key in data_discrete.columns:
                res = 0
                for k in range(len(data_discrete_new)):
                    row = data_discrete_new.iloc[k].to_dict()
                
                    evidence = copy(row)
                    del evidence[key]
                    for node in data_discrete_new.columns:
                        filter = [f'({key},', f', {key},', f', {key})']
                        if any([filt in node for filt in filter]) and ')_' in node:
                            del evidence[node]
                    
                    sample = predict(bn_h, 'simple', 100, evidence=evidence, regime='mode')
                    if sample[key] == row[key]:
                        res += 1
                res = res/len(data_discrete_new)
                result_dict[key] = res
            df = df.append({'file': file,  **result_dict}, ignore_index=True)
            df.to_csv(f'{currentdir}/{name}_acc.csv', index=False)
            df = pd.read_csv(f'{currentdir}/{name}_acc.csv')

            result_dict = dict()
            for key in data_discrete.columns:
                res = 0
                for k in range(len(data_discrete)):
                    row = data_discrete.iloc[k].to_dict()
                    evidence = copy(row)
                    del evidence[key]
            
                            
                    sample = predict(bn_h_simple, 'simple', 100, evidence=evidence, regime='mode')
                    
                    if sample[key] == row[key]:
                        res += 1
                res = res/len(data_discrete)
                result_dict[key] = res
            df = df.append({'file': file.replace('.txt', '_simple.txt'),  **result_dict}, ignore_index=True)
            df.to_csv(f'{currentdir}/{name}_acc.csv', index=False)
            df = pd.read_csv(f'{currentdir}/{name}_acc.csv')
            
            
    except:
        with open(f'{currentdir}/{name}_pred_error.txt', 'a', encoding='utf-8') as log:
            log.write(f'{file}\n')
            log.write(f'{row}\n')
            log.write(f'{evidence}\n')
            log.write(traceback.format_exc() + '\n')

# %%
df.to_csv(f'{currentdir}/{name}_acc.csv', index=False)


