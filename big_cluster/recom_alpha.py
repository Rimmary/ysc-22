# %%
import os,sys,inspect
currentdir = 'D:/BAMT/plan/big_cluster'
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentparentdir)

# %%
import pandas as pd
import random
import numpy as np

from preprocess.discretization import get_nodes_type, discretization, code_categories, get_nodes_sign

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import normalized_mutual_info_score





# %%
def su_dist(x, y): 
    return 1.0 - normalized_mutual_info_score(x, y)

result = pd.DataFrame(columns=['name', 'alpha', 'max'])
name_dict = {'healthcare': 4500, 'sangiovese': 4500, 'mehra': 4500, 'hepar2': 3000, 'diabetes': 3000, 'link': 3000, 'pathfinder': 3000}

for name, sample in name_dict.items():
    if sample < 4500:
        data = pd.read_csv(f'{currentdir}/my_{name}.csv', index_col='Unnamed: 0')
    else:
        data = pd.read_csv(f'{currentdir}/my_{name}.csv')

    data.reset_index(drop=True, inplace=True)

    col_inter = list(data.columns)
    data_save=data[col_inter]


    repeat = 1
    # for option in ['hamming']:

    #for size in [1500, 3000, 4500, 6000]:
    target_value = {}
    for size in [sample]:
        for large in [len(col_inter)]:
            for iter_sample in range(repeat):
                if size < len(data):
                    data = data.sample(n=size, random_state=1)
                col_new = random.sample(col_inter, large)
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
                #D = np.where(np.isnan(D), 0.0, D)    


                for alpha in [step/10 for step in range(1, 11)]:
                        res_dict = {}
                
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
                        max_value = len(ind_dict)
                        if min(ind_dict.keys(), key=(lambda new_k: len(ind_dict[new_k]))) == 1:
                            print(f'{name} {alpha}')
#                         max_key = max(ind_dict.keys(), key=(lambda new_k: len(ind_dict[new_k])))

#                         if max_value <= len(ind_dict[max_key]):
#                             max_value = len(ind_dict[max_key])

#                         target_value[str(alpha)] = max_value
#     target_value = dict(sorted(target_value.items(), key=lambda item: (item[1], -float(item[0]))))
#     result = result.append({'name': name, 'alpha': str(list(target_value.keys())), 'max': str(list(target_value.values()))}, ignore_index=True)
# print(result)
# result.to_csv(f'{currentdir}/com_alpha.csv', index=False)
       
