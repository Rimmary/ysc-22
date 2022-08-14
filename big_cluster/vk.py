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
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from copy import copy
from visualization.visualization import draw_BN, draw_comparative_hist, get_probability
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from core.core_utils import project_root
from external.pyBN.utils.independence_tests import mutual_information, entropy, mi_from_en
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
import seaborn as sns
import sklearn.metrics as skm
import scipy.stats as ss
from scipy.spatial.distance import cdist
from external.libpgm.sampleaggregator import SampleAggregator
from sklearn.metrics import accuracy_score, mean_squared_error
import operator
from typing import Tuple
import json
from bayesian.sampling import generate_synthetics
from graph.precision_recall import child_dict
from visualization.visualization import get_probability, grouped_barplot
from bayesian.calculate_accuracy import parall_accuracy
import time
import itertools

# %%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# %%
def su_dist(x, y): 
    #print(x)
    #print(y)
    #z = np.concatenate((np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)), axis=1)
    #z = np.concatenate((x, y), axis=1)
    """if (entropy(x) + entropy(y)) == 0.0:
        print(mi_from_en(z))
        return 0.0
    else:
        return - 2 * mi_from_en(z) / (entropy(x) + entropy(y))"""
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
# %%
data = pd.read_csv(f'{project_root()}/data/vk_data.csv')
data.reset_index(drop=True, inplace=True)

col_inter = ['sex', 'age', 'city_id', 'has_high_education', 'relation',
       'num_of_relatives', 'followers_count', 'status', 'mobile_phone',
       'twitter', 'facebook', 'instagram', 'about', 'about_topic',
       'activities', 'activities_topic', 'books', 'interests',
       'interests_topic', 'movies', 'music', 'quotes', 'personal_alcohol',
       'personal_life_main', 'personal_people_main', 'personal_political',
       'HLength', 'max_tr', 'min_tr', 'mean_tr', 'median_tr', '90_perc',
       'sum_am', 'tr_per_month', 'cash_sum', 'cash_usage', 'top1', 'top1_mean',
       'top2', 'top2_mean', 'top3', 'top3_mean', 'game_sum', 'is_gamer',
       'parent_sum', 'is_parent', 'driver_sum', 'is_driver', 'pets_sum',
       'has_pets']

data_save=data[col_inter]
data_save['city_id'] = data_save['city_id'].apply(lambda x: int(x)).apply(lambda x: str(x))

# if 'Onshore/offshore' in data_save.columns:
#     data_save.rename({'Onshore/offshore': 'OnshoreOffshore'}, axis = 1, inplace = True)

# col_inter = ['Country', 'Region', 'Tectonic regime',
#        'OnshoreOffshore', 'Hydrocarbon type', 'Productive area', 'Period',
#        'Depositional system', 'Depositional environment', 'Lithology',
#        'Porosity type', 'Gross', 'Netpay', 'Porosity', 'Permeability',
#        'Structural setting', 'Trapping mechanism', 'Depth']
repeat = 5
#for size in [1500, 3000, 4500, 6000]:
for size in [1500]:
    for large in range(15, len(col_inter)):
    #for large in range(10, 11):
        for iter_sample in range(repeat):
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
            data_discrete, coder = discretization(data_coded, 'equal_frequency', columns_for_disc)
            time_start = time.time()
            bn_without = structure_learning(data_discrete, 'HC', get_nodes_type(data), 'BIC', cont_disc = False)
            time_long = time.time() - time_start
            with open(f'{currentdir}/log_full_vk.txt', 'a', encoding="utf-8") as log:
                log.write(f'{large}\n')
                log.write(f'Sample {col_new}\n')
                log.write(f'Time {time_long}\n')
            # %%
            column = list(data.columns)
            D = [[0.0 for i in column] for j in column]
            for i, var1 in enumerate(column):
                for j, var2 in enumerate(column):
                    D[i][j] = su_dist(data_discrete[var1].values, data_discrete[var2].values)
            #D = np.where(np.isnan(D), 0.0, D)    


            # %%
            for alpha in [step/20 for step in range(1, 21)]:
            #for alpha in [0.5]:

                res_dict = {}

                time_start = time.time()
                model = AgglomerativeClustering(distance_threshold=alpha, n_clusters=None, affinity='precomputed', linkage='single')
                model = model.fit_predict(D)

                for i, val in enumerate(model):
                    if val in res_dict:
                        res_dict[val].append(column[i])
                    else:
                        res_dict[val] = [column[i]]
                # for key, val in res_dict.items():
                #     if len(val) > 1:
                #         print(val)


                # # %%
                # ax = sns.heatmap(D)

                # # %%
                # label_dict = {var: var for var in column}
                # label_dict['Depositional system'] = 'Dep. sys.'
                # label_dict['Depositional environment'] = 'Dep. env.'
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
                


                #g = [None for _ in range(count)]
                # ind = 0
                # for key, val in ind_dict.items():
                #     if len(val) > 1:
                #         loc_D = [[D[i][j] for j in val] for i in val]
                #         fig, ax = plt.subplots()
                #         ax = sns.heatmap(loc_D, xticklabels=[label_dict[column[i]] for i in val], yticklabels=[label_dict[column[i]] for i in val])
                #         plt.tight_layout()
                #         plt.savefig(str([column[i] for i in val]))
                #         #fig.clear()
                #         ind += 1


                # %%\

                df2 = pd.DataFrame()
                try:
                    for key, val in res_dict.items():
                        if len(val) > 1:
                            code_col = zip_code(data_discrete, val)
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
                #draw_BN(bn_rename, {rerename_dict[key]: val for key, val in node_type.items()}, 'coded_vars')

                # %%
                bn_dict = dict()
                for key, val in res_dict.items():
                    if len(val) > 1:
                        bn_loc = structure_learning(data_discrete[val], 'HC', {key_t: val_t for key_t, val_t in data_types.items() if key_t in val}, 'BIC', cont_disc = True)
                        bn_dict[str(tuple(val))] = bn_loc
                #bn_dict

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
                #print(bn_full['V'])
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
                #rename_again

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
                #print(nodes_loc_net)




                # %%
                # for key, val in nodes_loc_net.items():  
                #     draw_BN(val, {var: 'disc' for var in val['V']}, f'local_net_for_{rename_again_inv[key]}')

                # %%
                save_structure(bn_full_rename, 'coded_vars_full_vk_net')
                time_long = time.time() - time_start
                with open(f'{currentdir}/log_coded_vk.txt', 'a', encoding="utf-8") as log:
                    log.write(f'{large}\n')
                    log.write(f'{alpha}\n')
                    log.write(f'Sample {col_new}\n')
                    log.write(f'Time {time_long}\n')
            # %%
            #draw_BN(bn_full_rename, {var: 'disc' for var in bn_full_rename['V']}, 'coded_full_net')

            # # %%
            # example = dict()
            # example['V'] = ["('parent_sum', 'is_parent')_in", "('parent_sum', 'is_parent')_out", 'parent_sum', 'is_parent']
            # example['E'] = [["('parent_sum', 'is_parent')_in", 'is_parent'], ["('parent_sum', 'is_parent')_in", 'parent_sum'], ['is_parent', 'parent_sum'],
            # ['parent_sum', "('parent_sum', 'is_parent')_out"], ['is_parent', "('parent_sum', 'is_parent')_out"] ]
            #draw_BN(example, {var: 'disc' for var in example['V']}, 'example_block')

            # # %%
            # nodes_type_add = get_nodes_type(data_add)

            # # %%
            # ind = 1 
            # nodes_loc_param = {}
            # for key, val in nodes_loc_net.items():
            #     if any([')_' in rename_again_inv[var] for var in val['V']]):
            #         try:
            #             params = parameter_learning(data_add[val['V']], {key1: val1 for key1, val1 in nodes_type_add.items() if key1 in val['V']}, val, 'simple', 'kNN')
            #             nodes_loc_param[key] = params
            #             save_params(params, f'coded_vars_full_param_{rename_again_inv[key]}')
            #             print('success')
            #         except:
            #             print(ind)
            #             ind += 1


            # # %%
            # nodes_loc_net[rename_again['is_parent']]['E']

            # # %%
            # key = 'has_pets'
            # rename_key = rename_again[key]
            # save_structure(nodes_loc_net[rename_key], f'coded_vars_full_net_{key}')
            # skelet = read_structure(f'coded_vars_full_net_{key}')
            # params = read_params(f'coded_vars_full_param_{key}')
            # bn_h = HyBayesianNetwork(skelet, params)
            # sample1 = generate_synthetics(bn_h, get_nodes_sign(data_add[nodes_loc_net[rename_key]['V']]), 'simple', 200)

            # # %%
            # sample1[[rename_key]].head()

            # # %%
            # data_add[[rename_key]].head()

            # %%
            # draw_comparative_hist(rename_key, data_add[[rename_key]], sample1[[rename_key]], {rename_key: 'disc'})

            # # %%
            # bn_without_rename_again = copy(bn_without)
            # bn_without_rename_again['V'] = [rename_again[var] for var in bn_without['V']]
            # bn_without_rename_again['E'] = [[rename_again[var1], rename_again[var2]]  for var1, var2 in bn_without['E']]

            # # %%
            # nodes_without_loc_net = {}
            # full_net = copy(bn_without_rename_again)
            # for var in full_net['V']:
            #     if ')_' not in rename_again_inv[var]:
            #         nodes_without_loc_net[var] = {'V': [var], 'E': []}
            #         nodes_loc = set([var])
            #         for e0, e1 in full_net['E']:
            #             if e1 == var:
            #                 nodes_without_loc_net[var]['E'].append([e0, e1])
            #                 nodes_loc.add(e0)
            #                 if (')_out' in rename_again_inv[e0]) or (')_in' in rename_again_inv[e0]):
            #                     for e2, e3 in full_net['E']:
            #                         if (e3 == e0) and (e2 != var):
            #                             nodes_without_loc_net[var]['E'].append([e2, e3])
            #                             nodes_loc.add(e2)
                    
            #         nodes_without_loc_net[var]['V'] = list(nodes_loc)

            # # # %%
            # # for key, val in nodes_without_loc_net.items():  
            # #     draw_BN(val, {var: 'disc' for var in val['V']}, f'local_without_net_for_{rename_again_inv[key]}')

            # # %%
            # save_structure(bn_without, 'without_full_geo_net')

            # # %%
            # #data_without = copy(data)
            # data_without = copy(data_coded)
            # data_without.rename(rename_again, axis=1, inplace=True)


            # # %%
            # nodes_type_without = get_nodes_type(data_without)

            # # %%
            # ind = 1 
            # nodes_without_loc_param = {}
            # for key, val in nodes_without_loc_net.items():
            #     if key in nodes_loc_param:
            #         try:
            #             params = parameter_learning(data_without[val['V']], {key1: val1 for key1, val1 in nodes_type_without.items() if key1 in val['V']}, val, 'simple', 'kNN')
            #             nodes_without_loc_param[key] = params
            #             save_params(params, f'without_param_{rename_again_inv[key]}')
            #             #print('success')
            #         except:
            #             print(ind)
            #             ind += 1

            # # %%
            # def draw_comparative_hist(parameter: str, original_data: pd.DataFrame,
            #                         synthetic_data: pd.DataFrame, synthetic_data2: pd.DataFrame,  node_type: dict, rename_again_inv: dict):
                

            #     if node_type[parameter] == 'disc':
            #         plt.clf()
            #         df1 = pd.DataFrame()
            #         probs = get_probability(sample=original_data, initial_data=original_data,parameter=parameter)

            #         df1[parameter] = list(probs.keys())

            #         df1['Probability'] = [p[1] for p in probs.values()]
            #         df1['Error'] = [p[2] - p[1] for p in probs.values()]
            #         df1['Data'] = 'Original data'

            #         df2 = pd.DataFrame()
            #         probs = get_probability(sample=synthetic_data, initial_data=original_data, parameter=parameter)
            #         df2[parameter] = list(probs.keys())
            #         df2['Probability'] = [p[1] for p in probs.values()]
            #         df2['Error'] = [p[2] - p[1] for p in probs.values()]
            #         df2['Data'] = 'Clustered'

            #         df3 = pd.DataFrame()
            #         probs = get_probability(sample=synthetic_data2, initial_data=original_data, parameter=parameter)
            #         df3[parameter] = list(probs.keys())
            #         df3['Probability'] = [p[1] for p in probs.values()]
            #         df3['Error'] = [p[2] - p[1] for p in probs.values()]
            #         df3['Data'] = 'Without clustering'

            #         final_df = pd.concat([df1, df2, df3])

            #         grouped_barplot(final_df, parameter, 'Data', 'Probability', 'Error')
            #     # else:
            #     #     sns.distplot(processor.data[parameter], hist=False, label='Исходные данные')
            #     #     sns.distplot(data_without_restore[parameter], hist=False, label='Данные из сети с изучаемым узлом')
            #     #     ax = sns.distplot(data_with_restore[parameter], hist=False, label='Данные из сети без изучаемого узла')
            #     #     ax.legend()
                
            #     plt.savefig(f'barplot_{rename_again_inv[parameter]}')
            #     #plt.show()
            #     plt.close()
