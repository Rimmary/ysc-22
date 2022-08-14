# %%
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentparentdir)

# %%
import pandas as pd
import random
import numpy as np
from bayesian.train_bn import structure_learning, parameter_learning
from preprocess.discretization import get_nodes_type, discretization, code_categories, get_nodes_sign

from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from copy import copy
from sklearn.cluster import AgglomerativeClustering

from core.core_utils import project_root

from scipy.cluster.hierarchy import dendrogram, linkage


import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

from scipy.spatial.distance import cdist
import time
import traceback



# %%
def su_dist(x, y): 
    return 1.0 - normalized_mutual_info_score(x, y)


# %%
def zip_code(data_discrete: pd.DataFrame, cluster_columns: list, alpha: float = 0.95, low_limit: int = 30):
    group = copy(data_discrete).groupby(cluster_columns)
    comb = group.size().reset_index().rename(columns={0:'count'})
    comb['count'] = comb['count'] / len(data_discrete)
    comb.sort_values(by=['count'], inplace=True, ascending=False)
    sum = 0.0
    ind = 0
    while (sum < alpha) & (comb.iloc[ind]['count'] >= low_limit / len(data_discrete)):
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

name = 'pathfinder'
data = pd.read_csv(f'{project_root()}/data/my_{name}.csv')
data.reset_index(drop=True, inplace=True)

col_inter = list(data.columns)
data_save=data[col_inter]


repeat = 1
# for option in ['hamming']:

#for size in [1500, 3000, 4500, 6000]:
for size in [3000]:
    # for large in [len(col_inter)]:
    # for large in range(15, len(col_inter)):
    for large in range(10, 11):
        for iter_sample in range(repeat):
            if size < len(data):
                data = data.sample(n=size, random_state=1)
            col_new = random.sample(col_inter, large)
            data = data_save[col_new]

            data_types = get_nodes_type(data)

            # %%
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

            time_start = time.time()
            bn_without = structure_learning(data_discrete, 'HC', get_nodes_type(data), 'BIC', cont_disc = False)
            time_long = time.time() - time_start
            save_structure(bn_without, f'uncoded_{name}')
            with open(f'{currentdir}/log_full_{name}.txt', 'a', encoding="utf-8") as log:
                log.write(f'{large}\n')
                log.write(f'Sample {col_new}\n')
                log.write(f'Time {time_long}\n')
            # %%
            column = list(data.columns)
            D = [[0.0 for i in column] for _ in column]
            for i, var1 in enumerate(column):
                for j, var2 in enumerate(column):
                    D[i][j] = su_dist(data_discrete[var1].values, data_discrete[var2].values)
            #D = np.where(np.isnan(D), 0.0, D)    


                # %%
            for option in ['mostly', 'hamming']:

                for alpha in [step/10 for step in range(1, 11)]:
                #for alpha in [0.5]:
                    try:
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
                        node_type = {}
                        for key, val in rename_dict.items():
                            if len(key) > 1:
                                node_type[val] = 'dict'
                            else:
                                node_type[val] = data_types[key[0]]



                        # %%
                        bn = structure_learning(df2, 'HC', node_type, 'BIC', cont_disc = False)

                        # %%
                        rerename_dict = {val: str(key) for key, val in rename_dict.items()}
                        bn_rename = copy(bn)
                        bn_rename['V'] = [rerename_dict[var] for var in bn_rename['V']]
                        bn_rename['E'] = [[rerename_dict[var1], rerename_dict[var2]]  for var1, var2 in bn_rename['E']]


                        # %%
                        bn_dict = dict()
                        for key, val in res_dict.items():
                            if len(val) > 1:
                                bn_loc = structure_learning(data_discrete[val], 'HC', {key_t: val_t for key_t, val_t in data_types.items() if key_t in val}, 'BIC', cont_disc = True)
                                bn_dict[str(tuple(val))] = bn_loc
                
                        # %%
                        bn_full = copy(bn_rename)
                        for key, bn_loc in bn_dict.items():
                            bn_full['V'].extend(bn_loc['V'])
                            bn_full['V'].remove(key)
                            bn_full['V'].extend([key+"_in", key+'_out'])
                            for i in range(len(bn_full['E'])):
                                e0, e1 = bn_full['E'][i]
                                if e0 == key:
                                    bn_full['E'][i] = [key+'_out', e1]
                                if e1 == key:
                                    bn_full['E'][i] = [e0, key+'_in']
                            bn_full['E'].extend(bn_loc['E'])
                            bn_full['E'].extend([[key+'_in', var] for var in bn_loc['V']])
                            bn_full['E'].extend([[var, key+'_out'] for var in bn_loc['V']])

                        # %%
                        rerename_dict_new = dict() 
                        for var in bn_full['V']:
                            if (',)' in var):
                                rerename_dict_new[var] = eval(var)[0]
                            else:
                                rerename_dict_new[var] = var
                        bn_full_rename = copy(bn_full)
                        bn_full_rename['V'] = [rerename_dict_new[var] for var in bn_full_rename['V']]
                        bn_full_rename['E'] = [[rerename_dict_new[var1], rerename_dict_new[var2]]  for var1, var2 in bn_full_rename['E']]
                    

                        # %%
                        rename_dict = {str(key): val for key, val in rename_dict.items()}

                        # %%
                        #data_add = copy(data)
                        data_add = copy(data_discrete)
                        for key in bn_full_rename['V']:
                            if ')_in' in key:
                                data_add[key] = df2[rename_dict[key[:-3]]]
                            if ')_out' in key:
                                data_add[key] = df2[rename_dict[key[:-4]]]
                                
                        # %%
                        rename_again = {var: str(i) for i, var in enumerate(data_add.columns)}

                        # %%
                        rename_again_inv = {i: var for var, i in rename_again.items()}

                        # %%
                        data_add.rename(rename_again, axis=1, inplace=True)

                                

                        # %%
                        bn_full_rename_again = copy(bn_full_rename)
                        bn_full_rename_again['V'] = [rename_again[var] for var in bn_full_rename['V']]
                        bn_full_rename_again['E'] = [[rename_again[var1], rename_again[var2]]  for var1, var2 in bn_full_rename['E']]
                        #bn_full_rename_again
                                

                        # %%
                        nodes_loc_net = {}
                        full_net = copy(bn_full_rename_again)
                        for var in full_net['V']:
                            if ')_' not in rename_again_inv[var]:
                                nodes_loc_net[var] = {'V': [var], 'E': []}
                                nodes_loc = set([var])
                                for e0, e1 in full_net['E']:
                                    if e1 == var:
                                        nodes_loc_net[var]['E'].append([e0, e1])
                                        nodes_loc.add(e0)
                                        if (')_out' in rename_again_inv[e0]) or (')_in' in rename_again_inv[e0]):
                                            for e2, e3 in full_net['E']:
                                                if (e3 == e0) and (e2 != var):
                                                    nodes_loc_net[var]['E'].append([e2, e3])
                                                    nodes_loc.add(e2) 
                                nodes_loc_net[var]['V'] = list(nodes_loc)



                        # %%
                        time_long = time.time() - time_start
                        hyparam = str(alpha).replace('0.', '')
                        save_structure(bn_full_rename, f'coded_{name}_{hyparam}_{option}_{size}')
                        
                        with open(f'{currentdir}/log_coded_{name}.txt', 'a', encoding="utf-8") as log:
                            log.write(f'{large}\n')
                            log.write(f'{alpha}\n')
                            log.write(f'{option}\n')
                            log.write(f'Sample {col_new}\n')
                            log.write(f'Time {time_long}\n')
                    except:
                        with open(f'{currentdir}/{name}_error.txt', 'a', encoding='utf-8') as log:
                            log.write(f'{large}\n')
                            log.write(f'{alpha}\n')
                            log.write(f'{option}\n')
                            log.write(f'{col_new}\n')
                            log.write(traceback.format_exc() + '\n')
       
