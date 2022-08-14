
import itertools
from copy import copy
import math
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, K2Score
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
from sklearn import linear_model
from bayesian.structure_score import MIG, LLG, BICG, AICG
from sklearn.mixture import GaussianMixture
from bayesian.redef_HC import hc as hc_method
import random
from scipy.stats.distributions import chi2
from scipy import stats
from sklearn.linear_model import LinearRegression
import os, sys

from sklearn.metrics import mean_squared_error
import joblib
import xgboost as xgb
from core.core_utils import project_root
import time
from pathlib import Path
import catboost as cb

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)




random.seed(1)
np.random.seed(1)



def structure_learning(data: pd.DataFrame, search: str, node_type: dict, score: str = 'MI', init_nodes: list = None,
                       white_list: list = None, max_iter=1e6,
                       init_edges: list = None, remove_init_edges: bool = True, black_list: list = None, cont_disc = False) -> dict:
    """Function for bayesian networks structure learning

    Args:
        data (pd.DataFrame): input encoded and discretized data
        search (str): search strategy (HC, evo)
        score (str): algorith of learning (K2, MI, MI_mixed)
        node_type (dict): dictionary with node types (discrete or continuous)
        init_nodes (list, optional): nodes with no parents. Defaults to None.
        white_list (list, optional): allowable edges. Defaults to None.
        init_edges (list, optional): start edges of graph (set user). Defaults to None.
        remove_init_edges (bool, optional): flag that allow to delete start edges (or not allow). Defaults to True.
        black_list (list, optional): forbidden edges. Defaults to None.
        cont_disc (bool, optional): flag that allow to delete start edges (or not allow). Defaults to False.

    Returns:
        dict: dictionary with structure (values are lists of nodes and edges)
    """
    blacklist = []
    datacol = data.columns.to_list()
    if init_nodes:
        blacklist = [(x, y) for x in datacol for y in init_nodes if x != y]
    if not cont_disc:    
        for x, y in itertools.product(datacol, repeat=2):
            if x != y:
                if (node_type[x] == 'cont') & (node_type[y] == 'disc'):
                    blacklist.append((x, y))
    if black_list:
        blacklist = blacklist + black_list

    skeleton = dict()
    skeleton['V'] = datacol

    if search == 'HC':
        if (score == "MI") | (score == "LL") | (score == "BIC") | (score == "AIC"):
            column_name_dict = dict([(n, i) for i, n in enumerate(datacol)])
            blacklist_new = []
            for pair in blacklist:
                blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if white_list:
                white_list_old = copy(white_list)
                white_list = []
                for pair in white_list_old:
                    white_list.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if init_edges:
                init_edges_old = copy(init_edges)
                init_edges = []
                for pair in init_edges_old:
                    init_edges.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            bn = hc_method(data, metric=score, restriction=white_list, init_edges=init_edges, remove_geo_edges=remove_init_edges, black_list=blacklist_new)
            structure = []
            nodes = sorted(list(bn.nodes()))
            for rv in nodes:
                for pa in bn.F[rv]['parents']:
                    structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                  list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
            skeleton['E'] = structure
            
        if score == "K2":
            hc_K2Score = HillClimbSearch(data)
            if init_edges == None:
                best_model_K2Score = hc_K2Score.estimate(
                    scoring_method=K2Score(data),
                    black_list=blacklist, white_list=white_list,
                    max_iter=max_iter, show_progress=False)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag, show_progress=False)
                else:
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges, show_progress=False)
            structure = [list(x) for x in list(best_model_K2Score.edges())]
            skeleton['E'] = structure


        if score == 'MI_mixed':
            hc_mi_mixed = HillClimbSearch(data)
            if init_edges == None:
                best_model_mi_mixed = hc_mi_mixed.estimate(scoring_method=MIG(data=data), black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_mi_mixed.edges())]
            skeleton['E'] = structure
        
        if (score == 'LL_mixed') | (score == 'BIC_mixed') | (score == 'AIC_mixed'):
            if score == 'LL_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=LLG(data=data))
            if score == 'BIC_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=BICG(data=data))
            if score == 'AIC_mixed':
                hc_mi_mixed = HillClimbSearch(data, scoring_method=AICG(data=data))

            if init_edges == None:
                best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_mi_mixed.edges())]
            skeleton['E'] = structure


    if search == 'evo':
        print('In process')
    return skeleton

def parameter_learning(data: pd.DataFrame, node_type: dict, skeleton: dict, method: str, regressor: str = 'cb', mono_dict: dict = None) -> dict:
    """Function for parameters learning in BN

    Args:
        data (pd.DataFrame): data for learning
        node_type (dict): types of nodes
        skeleton (dict): BN cstructure
        method (str): method of learning - simple, mix or reg
        regressor (str): if method is reg, this parameter indicates the type of regression - cb or xgb

    Raises:
        Exception: The specified type of parametric learning is not supported

    Returns:
        dict: distributions parameters
    """    
    params = dict()
    if method == 'simple':
        params = parameter_learning_simple(data, node_type, skeleton)
    elif method == 'mix':
        params = parameter_learning_mix(data, node_type, skeleton)
    elif method == 'reg':
        params = parameter_learning_reg(data, node_type, skeleton, regressor, mono_dict)
    else:
        raise Exception("The specified type of parametric learning is not supported")
    return params



def parameter_learning_simple(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
    """Function for parameter learning for hybrid BN
    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN
    Returns:
        dict: dictionary with parameters of distributions in nodes
    """
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])
        disc_parents = []
        cont_parents = []
        for parent in parents:
            if node_type[parent] == 'disc':
                disc_parents.append(parent)
            else:
                cont_parents.append(parent)

        # Discrete Node
        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}

      
        if (node_type[node] == "disc") & (len(parents) != 0):
            # Discrete Node
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                numoutcomes = int(len(data[node].unique()))
                dist = DiscreteDistribution.from_samples(data[node].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": children}
                else:
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": None}
            #Logit Node
  
            if (len(disc_parents) == 0):
                model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                model.fit(data[parents].values, data[node].values)
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": None}
            #Conditional Logit Node
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base = [np.nan]
                    classes = []
                    if new_data.shape[0] != 0:
                        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                        values = set(new_data[node])
                        if len(values) > 1:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            mean_base = model.intercept_
                            classes = model.classes_
                            coef = model.coef_
                        else:
                            mean_base = np.array([0.0])
                            coef = np.array([sys.float_info.max])
                            classes = list(values)
                            
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base),
                                                  'mean_scal': list(coef.reshape(1,-1)[0])}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base), 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": None,
                                                "hybcprob": hycprob}

            
        #Gaussian Node
        if (node_type[node] == "cont") & (len(parents) == 0):
            mean_base = np.mean(data[node].values)
            variance = np.var(data[node].values)
            mode = data[[node]].mode(axis=0, dropna=True).iloc[0]
            mode = mode[node]
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "mode": float(mode), "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "mode": float(mode), "type": "lg", "children": None}
        
        
        if (node_type[node] == "cont") & (len(parents) != 0):
            #Gaussian Node
            if (len(disc_parents) == 0):
                model = linear_model.LinearRegression()
                predict = []
                if len(parents) == 1:
                    model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                    predict = model.predict(np.transpose([data[parents[0]].values]))
                else:
                    model.fit(data[parents].values, data[node].values)
                    predict = model.predict(data[parents].values)
                variance = (RSE(data[node].values, predict)) ** 2
                mode = data[[node]].mode(axis=0, dropna=True).iloc[0]
                mode = mode[node]
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "mode": float(mode), "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "mode": float(mode), "type": "lg",
                                                "children": None}
            #Conditional Gaussian Node
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base = np.nan
                    variance = np.nan
                    predict = []
                    if new_data.shape[0] != 0:
                        model = linear_model.LinearRegression()
                        if len(cont_parents) == 1:
                            model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                        else:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            predict = model.predict(new_data[cont_parents].values)
                        key_comb = [str(x) for x in comb]
                        mean_base = model.intercept_
                        variance = (RSE(new_data[node].values, predict)) ** 2
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base,
                                                  'mean_scal': list(model.coef_)}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}
            # Conditional Gaussian Node
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    if new_data.shape[0] != 0:
                        mean_base = np.mean(new_data[node].values)
                        variance = np.var(new_data[node].values)
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}

    return node_data


def parameter_learning_mix(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
    """Function for parameter learning for hybrid BN

    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN

    Returns:
        dict: dictionary with parameters of distributions in nodes
    """
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])
        disc_parents = []
        cont_parents = []
        for parent in parents:
            if node_type[parent] == 'disc':
                disc_parents.append(parent)
            else:
                cont_parents.append(parent)

        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                numoutcomes = int(len(data[node].unique()))
                dist = DiscreteDistribution.from_samples(data[node].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": children}
                else:
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": None}

            if (len(disc_parents) == 0):
                model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                model.fit(data[parents].values, data[node].values)
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": None}

            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base = [np.nan]
                    classes = []
                    if new_data.shape[0] != 0:
                        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                        values = set(new_data[node])
                        if len(values) > 1:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            mean_base = model.intercept_
                            classes = model.classes_
                            coef = model.coef_
                        else:
                            mean_base = np.array([0.0])
                            coef = np.array([sys.float_info.max])
                            classes = list(values)
                            
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base),
                                                  'mean_scal': list(coef.reshape(1,-1)[0])}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base), 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": None,
                                                "hybcprob": hycprob}
       
       
       #Mixture Gaussian Node
        if (node_type[node] == "cont") & (len(parents) == 0):
            
            n_comp = int((component(data, [node], 'aic') + component(data, [node], 'bic')) / 2)#component(data, [node], 'LRTS')#
            #n_comp = 3
            #n_comp = 4
            gm_sklearn = GaussianMixture(n_components=n_comp, max_iter=500).fit(np.transpose([data[node].values]))
            #gmm = GMM(n_components=n_comp)
            #gmm.from_samples(np.transpose([data[node].values]))
            means = gm_sklearn.means_.tolist()#gmm.means.tolist()
            cov = gm_sklearn.covariances_.tolist()#gmm.covariances.tolist()
            #weigts = np.transpose(gmm.to_responsibilities(np.transpose([data[node].values])))
            w = gm_sklearn.weights_.tolist()#gmm.priors.tolist()#[]
            # for row in weigts:
            #     w.append(np.mean(row))
            mode = data[[node]].mode(axis=0, dropna=True).iloc[0]
            mode = mode[node]
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
                                            "variance": cov, "mode": float(mode), "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
                                            "variance": cov, "mode": float(mode), "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            #Mixture Gaussian Node
            if (len(disc_parents) == 0) & (len(cont_parents) != 0):
                nodes = [node] + cont_parents
                new_data = copy(data[nodes])
                new_data.reset_index(inplace=True, drop=True)
                n_comp = int((component(new_data, nodes, 'aic') + component(new_data, nodes, 'bic')) / 2)#component(new_data, nodes, 'LRTS')#
                #n_comp = len(new_data[node].unique())
                gm_sklearn = GaussianMixture(n_components=n_comp, max_iter=500).fit(new_data[nodes].values)
                #gmm = GMM(n_components=n_comp)
                #gmm.from_samples(new_data[nodes].values)
                means = gm_sklearn.means_.tolist()#gmm.means.tolist()
                cov = gm_sklearn.covariances_.tolist()#gmm.covariances.tolist()
                #weigts = np.transpose(gmm.to_responsibilities(new_data[nodes].values))
                w = gm_sklearn.weights_.tolist()#gmm.priors.tolist()#[]
                # for row in weigts:
                #     w.append(np.mean(row))
                mode = new_data[[node]].mode(axis=0, dropna=True).iloc[0]
                mode = mode[node]
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w,
                                                "parents": parents, "variance": cov, "mode": float(mode), "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w,
                                                "parents": parents, "variance": cov, "mode": float(mode), "type": "lg",
                                                "children": None}
    return node_data

def parameter_learning_reg(data: pd.DataFrame, node_type: dict, skeleton: dict, regressor: str = 'cb', mono_dict: dict = None) -> dict:
    """Function for parameter learning for hybrid BN

    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN
        regressor (str): indicates the type of regression - cb or xgb

    Returns:
        dict: dictionary with parameters of distributions in nodes
    """
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])
        disc_parents = []
        cont_parents = []
        for parent in parents:
            if node_type[parent] == 'disc':
                disc_parents.append(parent)
            else:
                cont_parents.append(parent)

        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                numoutcomes = int(len(data[node].unique()))
                dist = DiscreteDistribution.from_samples(data[node].values)
                vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
                dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
                params = dist.parameters[0]
                cprob = dict()
                for i in range(0, len(params), len(vals)):
                    probs = []
                    for j in range(i, (i + len(vals))):
                        probs.append(params[j][-1])
                    combination = [str(x) for x in params[i][0:len(parents)]]
                    cprob[str(combination)] = probs
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": children}
                else:
                    node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                                "vals": vals, "type": "discrete", "children": None}

            if (len(disc_parents) == 0):
                model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                model.fit(data[parents].values, data[node].values)
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": list(model.intercept_), "mean_scal": list(model.coef_.reshape(1,-1)[0]),
                                                "parents": parents, 'classes': list(model.classes_), "type": "logit",
                                                "children": None}

            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base = [np.nan]
                    classes = []
                    if new_data.shape[0] != 0:
                        model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=100)
                        values = set(new_data[node])
                        if len(values) > 1:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            mean_base = model.intercept_
                            classes = model.classes_
                            coef = model.coef_
                        else:
                            mean_base = np.array([0.0])
                            coef = np.array([sys.float_info.max])
                            classes = list(values)
                            
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base),
                                                  'mean_scal': list(coef.reshape(1,-1)[0])}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'classes': list(classes), 'mean_base': list(mean_base), 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "logitandd", "children": None,
                                                "hybcprob": hycprob}
       
       
       #Mixture Gaussian Node
        if (node_type[node] == "cont") & (len(parents) == 0):
            
            n_comp = int((component(data, [node], 'aic') + component(data, [node], 'bic')) / 2)#component(data, [node], 'LRTS')#
            #n_comp = 3
            #n_comp = 4
            gm_sklearn = GaussianMixture(n_components=n_comp, max_iter=500).fit(np.transpose([data[node].values]))
            #gmm = GMM(n_components=n_comp)
            #gmm.from_samples(np.transpose([data[node].values]))
            means = gm_sklearn.means_.tolist()#gmm.means.tolist()
            cov = gm_sklearn.covariances_.tolist()#gmm.covariances.tolist()
            #weigts = np.transpose(gmm.to_responsibilities(np.transpose([data[node].values])))
            w = gm_sklearn.weights_.tolist()#gmm.priors.tolist()#[]
            # for row in weigts:
            #     w.append(np.mean(row))
            mode = data[[node]].mode(axis=0, dropna=True).iloc[0]
            mode = mode[node]

            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
                                            "variance": cov, "mode": float(mode), "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": means, "mean_scal": w, "parents": None,
                                            "variance": cov, "mode": float(mode), "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            #Regressian Node
            if (len(disc_parents) == 0) & (len(cont_parents) != 0):
                nodes = [node] + cont_parents
                new_data = copy(data[nodes])
                new_data.reset_index(inplace=True, drop=True)
                if regressor == 'xgb':
                    xgb_reg = xgb.XGBRegressor()
                    xgb_reg.fit(new_data[cont_parents].values, new_data[node].values)
                    y_pred = xgb_reg.predict(new_data[cont_parents].values)
                    #regressor_path = os.path.join(f"{project_root()}","joblib_compressed",f"{time.time()}.joblib.compressed")
                    regressor_path = os.path.join("joblib_compressed",f"{time.time()}.joblib.compressed")
                    full_regressor_path = os.path.join(Path(__file__).parent.parent, regressor_path)
                    _ = joblib.dump(xgb_reg, full_regressor_path, compress=True, protocol=4)
                    variance = mean_squared_error(new_data[node].values, y_pred)

                    mode = new_data[[node]].mode(axis=0, dropna=True).iloc[0]
                    mode = mode[node]

                    node_data['Vdata'][node] = {"regressor": "xgb", "regressor_path": regressor_path, "mode": float(mode),
                                                "parents": parents, "variance": variance, "type": "reg",
                                                "children": children if len(children) != 0 else None}
                elif regressor == 'cb':
                    if mono_dict is None or len(mono_dict) == 0 or node not in mono_dict:
                        mono_list = None
                    else:
                        mono_list = [mono_dict[node][key] if key in mono_dict[node] else 0 for key in cont_parents]
                    model = cb.CatBoostRegressor(loss_function='RMSE', monotone_constraints = mono_list)
                    train_dataset = cb.Pool(new_data[cont_parents].values, new_data[node].values) 
                    model.fit(train_dataset, silent=True)
                    y_pred = model.predict(new_data[cont_parents].values)
                    #regressor_path = os.path.join(f"{project_root()}","joblib_compressed",f"{time.time()}.joblib.compressed")
                    regressor_path = os.path.join("joblib_compressed",f"{time.time()}.joblib.compressed")
                    full_regressor_path = os.path.join(Path(__file__).parent.parent, regressor_path)
                    _ = joblib.dump(model, full_regressor_path, compress=True, protocol=4)
                    variance = mean_squared_error(new_data[node].values, y_pred)

                    model_90 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.9', monotone_constraints = mono_list)
                    model_90.fit(train_dataset, silent=True)
                    regressor_path_90 = os.path.join("joblib_compressed",f"{time.time()}.joblib.compressed")
                    full_regressor_path_90 = os.path.join(Path(__file__).parent.parent, regressor_path_90)
                    _ = joblib.dump(model_90, full_regressor_path_90, compress=True, protocol=4)

                    model_50 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.5', monotone_constraints = mono_list)
                    model_50.fit(train_dataset, silent=True)
                    regressor_path_50 = os.path.join("joblib_compressed",f"{time.time()}.joblib.compressed")
                    full_regressor_path_50 = os.path.join(Path(__file__).parent.parent, regressor_path_50)
                    _ = joblib.dump(model_50, full_regressor_path_50, compress=True, protocol=4)

                    model_10 = cb.CatBoostRegressor(loss_function='Quantile:alpha=0.1', monotone_constraints = mono_list)
                    model_10.fit(train_dataset, silent=True)
                    regressor_path_10 = os.path.join("joblib_compressed",f"{time.time()}.joblib.compressed")
                    full_regressor_path_10 = os.path.join(Path(__file__).parent.parent, regressor_path_10)
                    _ = joblib.dump(model_10, full_regressor_path_10, compress=True, protocol=4)

                    mode = new_data[[node]].mode(axis=0, dropna=True).iloc[0]
                    mode = mode[node]

                    node_data['Vdata'][node] = {"regressor": "cb", "regressor_path": regressor_path,
                                                "regressor_path_90": regressor_path_90, "regressor_path_50": regressor_path_50, 
                                                "regressor_path_10": regressor_path_10, "mode": float(mode),
                                                "parents": parents, "variance": variance, "type": "reg",
                                                "children": children if len(children) != 0 else None}
            
    return node_data



def RSE(y_true, y_predicted):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true)))
    return rse
def lrts_comp(data):
    n = 0
    biggets_p = -1*np.infty
    comp_biggest = 0
    max_comp = 10
    if len(data) < max_comp:
        max_comp = len(data)
    for i in range (1, max_comp+1, 1):
        gm1 = GaussianMixture(n_components=i, random_state=0)
        gm2 = GaussianMixture(n_components=i+1, random_state=0)
        gm1.fit(data)
        ll1 = np.mean(gm1.score_samples(data))
        gm2.fit(data)
        ll2 = np.mean(gm2.score_samples(data))
        LR = 2*(ll2 - ll1)
        p = chi2.sf(LR, 1)    
        if p > biggets_p:
            biggets_p = p
            comp_biggest = i
        n = comp_biggest
    return n

def mix_norm_cdf(x, weights, means, covars):
    mcdf = 0.0
    for i in range(len(weights)):
        mcdf += weights[i] * stats.norm.cdf(x, loc=means[i][0], scale=covars[i][0][0])
    return mcdf

def theoretical_quantile (data, n_comp):
    model = GaussianMixture(n_components=n_comp,random_state=0)
    model.fit(data)
    q = []
    x = []
    #step =  ((np.max(model.sample(100000)[0])) - (np.min(model.sample(100000)[0])))/1000
    step =  (np.max(data) - np.min(data))/1000
    d = np.arange(np.min(data),np.max(data), step)
    for i in d:
        x.append(i)
        q.append(mix_norm_cdf(i,model.weights_, model.means_, model.covariances_))
    return x, q
def quantile_mix(p, vals, q):
    ind = q.index(min(q, key=lambda x:abs(x-p)))
    return vals[ind]
def probability_mix(val, vals, q):
    ind = vals.index(min(vals, key=lambda x:abs(x-val)))
    return(q[ind])
def sum_dist(data, vals, q):
    percs = np.linspace(1, 100, 10)
    x = np.quantile(data, percs/100)
    y = []
    for p in percs:
        y.append(quantile_mix(p/100, vals, q))
    dist = 0
    for xi,yi in zip(x,y):
        dist = dist + (abs(-1*xi + yi)) / math.sqrt(2)
    return dist


def component (data, columns, method):
    n = 1
    max_comp = 10
    x = []
    if data.shape[0] < max_comp:
        max_comp = data.shape[0]
    if len(columns) == 1:
        x = np.transpose([data[columns[0]].values])
    else:
        x = data[columns].values
    if method == 'aic':
        lowest_aic = np.infty
        comp_lowest = 0
        for i in range (1, max_comp+1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            aic1 = gm1.aic(x)
            if aic1 < lowest_aic:
                lowest_aic = aic1
                comp_lowest = i
            n = comp_lowest


    if method == 'bic':
        lowest_bic = np.infty
        comp_lowest = 0
        for i in range (1, max_comp+1, 1):
            gm1 = GaussianMixture(n_components=i, random_state=0)
            gm1.fit(x)
            bic1 = gm1.bic(x)
            if bic1 < lowest_bic:
                lowest_bic = bic1
                comp_lowest = i
            n = comp_lowest


    if method == 'LRTS':
        n = lrts_comp(x)
    if method == 'quantile':
        biggest_p = -1*np.infty
        comp_biggest = 0
        for i in range(1, max_comp, 1):
            vals, q = theoretical_quantile(x, i)
            dist = sum_dist(x, vals, q)
            p = probability_mix(dist, vals, q)
            if p > biggest_p:
                biggest_p = p
                comp_biggest = i
        n = comp_biggest
    return n
def n_component(data: pd.DataFrame, columns: list):
    n = 1
    max_comp = 10
    size = data.shape[0]
    d = len(columns)
    if data.shape[0] < max_comp:
        max_comp = data.shape[0]
    if len(columns) == 1:
        x = np.transpose([data[columns[0]].values])
    else:
        x = data[columns].values
    n1 = n_BIC(x, max_comp)
    n2 = n_AIC(x, max_comp)
    # n3 = n_comp_sample(size, d)
    n = int((n1 + n2) / 2)
    #n = round ((n + n3)/2)
    return n

def n_BIC(data: np.ndarray, max_comp: int):
    n = 1
    bic = 100000000000000000
    for i in range(1, max_comp+1, 1):
        gm = GaussianMixture(n_components=i, random_state=0)
        gm.fit(data)
        bic_current = gm.bic(data)
        if bic_current < bic:
            bic = bic_current
            n = i
        else:
            break
    return n
def n_AIC(data: np.ndarray, max_comp: int):
    n = 1
    aic = 100000000000000000
    for i in range(1, max_comp+1, 1):
        gm = GaussianMixture(n_components=i, random_state=0)
        gm.fit(data)
        aic_current = gm.aic(data)
        if aic_current < aic:
            aic = aic_current
            n = i
        else:
            break
    return n
coef095 = [13.5, 43.5, 73.25, 101.25, 133.5, 161.75, 198.0, 217.25, 249.25, 275.25]
intercept095 = [-7.499999999999993,
 -29.66666666666662,
 -39.666666666666615,
 -56.49999999999994,
 -76.33333333333329,
 -78.33333333333329,
 -105.49999999999989,
 -94.99999999999994,
 -113.66666666666663,
 -92.49999999999977]

def func_for_ndim (n_dim):
    Zpred095 = []
    for j in range(10):
        Zpred095.append(n_dim*coef095[j] + intercept095[j])
    return Zpred095
def sample_n_comp (size, Z):
    m095 = LinearRegression().fit(np.transpose([Z]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return round(m095.predict([[size]])[0])



def quantile2d(x,y,Nbins,nth):
    from numpy import percentile
    from scipy.stats import binned_statistic
    def myperc(x,n=nth):
        return(percentile(x,n))
    t=binned_statistic(x,y,statistic=myperc,bins=Nbins)
    
    return t
