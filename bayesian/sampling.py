import pandas as pd
from scipy.fftpack import fftfreq

from external.libpgm.hybayesiannetwork import HyBayesianNetwork


def generate_synthetics(bn: HyBayesianNetwork, method: str, n: int = 1000, evidence: dict = None, as_json=False) -> pd.DataFrame:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        sign (dict): dictionary with nodes signs
        method (str): method of sampling - simple or mix
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = 0

    if as_json:
        sample = dict()
        sample_current = pd.DataFrame(bn.randomsample(n, method, 'sample', evidence))
        for i in range(n):
            for c in sample_current.columns:
                if sample_current.loc[i,c] < 0:
                    sample_current.loc[i,c] = 0
        for c in sample_current.columns:
            sample[c] = list(sample_current[c].values)
        




    else:
        sample = pd.DataFrame(bn.randomsample(n, method, 'sample', evidence))
        for i in range(n):
            for c in sample.columns:
                if sample.loc[i,c] < 0:
                    sample.loc[i,c] = 0
    return sample


def predict(bn: HyBayesianNetwork, method: str, quantile, evidence: dict = None, regime: str = '') -> dict:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        sign (dict): dictionary with nodes signs
        method (str): method of sampling - simple or mix
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = 0
    if regime == '':
        regime = quantile+'_predict'

    sample = bn.randomsample(1, method, regime, evidence)[0]
    for k in sample.keys():
        if not isinstance(sample[k], str):
            if sample[k] < 0:
                sample[k] = 0
    new_sample = dict()
    for v in sample.keys():
        flag = True
        if '_res_fact' in v:
            par = bn.getparents(v)
            for vi in par:
                if sample[vi] != 0 and 'act_fact' in vi:
                    flag=False
            if flag:
                sample[v] = 0
                new_sample[v] = 0
        if 'product' in v:
            par = bn.getparents(v)
            for vi in par:
                if 'act_fact' in vi and sample[vi] == 0:
                    sample[v] = 0
                    new_sample[v] = 0
                    break
    new_sample = {**new_sample,**evidence}
    sample = bn.randomsample(1, method, regime, new_sample)[0]
    for k in sample.keys():
        if not isinstance(sample[k], str):
            if sample[k] < 0:
                sample[k] = 0
    return sample
