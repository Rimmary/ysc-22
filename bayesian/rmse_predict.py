import pandas as pd
import numpy as np
from bayesian.train_bn import parameter_learning
from preprocess.discretization import get_nodes_type
from copy import copy
from sklearn.metrics import mean_squared_error
import operator

def one_acc(params: dict, data: pd.DataFrame, res: str, act: str, normed: str = 'none'):
    """Function for rapid assessment of rmse
    Args:
        params (dict): part of BN with parametric learning data
        data (pd.DataFrame): test dataset
        res (str): resource name
        act (str): activity name
        normed (str, optional): parameter controlling whether and how rmse will be normalised ('none', 'range', 'std')
    Returns:
        rmse (float): rapid assessment of rmse
    """
    pred_param = [0 for j in range(len(data))]
    real_param = [0 for j in range(len(data))]
    for i in range(len(data)):
        if res:
            prob = params['Vdata'][act]['cprob'][str(list(data[[res]].values[i]))]
        else:
            prob = params['Vdata'][act]['cprob']
        if params['Vdata'][act]['vals'][np.argmax(prob)] == '0.0':
            pred_param[i] = 0.0
        else:
            q = 1 - prob[params['Vdata'][act]['vals'].index('0.0')]
            pred_param[i] = np.dot(prob, np.array(params['Vdata'][act]['vals'], dtype=np.float16)) / q
        real_param[i] = float(data[act][i])
    
    rmse = 0.0
    if normed == 'range':
        rmse = round(mean_squared_error(real_param, pred_param, squared=False) / (np.max(data[act].values) - np.min(data[act].values)), 3)
    elif normed == 'std':
        std = np.std(data[act].values)
        rmse = round(mean_squared_error(real_param, pred_param, squared=False) / std, 3)
    elif normed == 'none':
        rmse = round(mean_squared_error(real_param, pred_param, squared=False), 3)
    return rmse
    
def rmse_predict(data: pd.DataFrame, ind_res: list, ind_act: list, rmse_option: str = 'none'):
    """Function which evaluates rmse and arranges the resource lists for each activity in ascending order of error
    Args:
        data (pd.DataFrame): test dataset
        ind_res (list): resource names
        ind_act (list): activity names
        rmse_option (str, optional): parameter controlling whether and how rmse will be normalised ('none', 'range', 'std')
    Returns:
        final_dict (dict): for each activity contains a sub-vocabulary in which 'res' corresponds to an ordered list of resources 
        and 'rmse' to an ascending list of errors
    """
    df = copy(data)
    for var in df.columns:
        df[var] = df[var].apply(float)
    for var in df.columns:
        df[var] = df[var].apply(str)
    ind_columns = copy(ind_res)
    ind_columns.extend(ind_act)
    # I see no contradiction if the lists intersect
    ind_columns = list(set(ind_columns))
    df = df[ind_columns]
    data_types = get_nodes_type(df) # Equivalent to {var: 'disc' for var in ind_columns}
    train = copy(df)
    test = copy(df)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
        
    res_result = {act: [] for act in ind_act}
    for act in ind_act:
        for res in ind_res:
            if res != act:
                nodes = [res, act]
                geo_types_new = {key: data_types[key] for key in nodes}
                bn = {'V': nodes, 'E': [[res, act]]}
                params = parameter_learning(train[nodes], geo_types_new, bn, 'simple', classifier = 'logit')
                rmse = one_acc(params, test, res, act, rmse_option)
                res_result[act].append([rmse, res])
    final_dict = {act: {'res': [], 'rmse': []}  for act in ind_act}
    for act, value in res_result.items():
        res_result[act].sort(key=operator.itemgetter(0))
        for pair in value:
            final_dict[act]['rmse'].append(pair[0])
            final_dict[act]['res'].append(pair[1])
    return final_dict


