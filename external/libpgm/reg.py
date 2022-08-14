# Copyright (c) 2012, CyberPoint International, LLC
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the CyberPoint International, LLC nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CYBERPOINT INTERNATIONAL, LLC BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
This module contains tools for representing regressian nodes -- those with a continuous linear Gaussian distribution of outcomes and a finite number of continuous parents -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''

import random
import math
from scipy.spatial import distance
import numpy as np
import xgboost as xgb
import warnings
import joblib
from pathlib import Path
import os
import catboost as cb

class Reg():
    '''
    This class represents a linear Gaussian node, as described above. It contains the *Vdataentry* attribute and the *choose* method.

    '''
    def __init__(self, Vdataentry):
        '''
        This class is constructed with the argument *Vdataentry* which must be a dict containing a dictionary entry for this particular node. The dict must contain entries of the following form::
        
            "mean_base": <float used for mean starting point
                          (\mu_0)>,
            "mean_scal": <array of scalars by which to
                          multiply respectively ordered 
                          continuous parent outcomes>,
            "variance": <float for variance>

        See :doc:`lgbayesiannetwork` for an explanation of linear Gaussian sampling.

        The *Vdataentry* attribute is set equal to this *Vdataentry* input upon instantiation.

        '''
        self.Vdataentry = Vdataentry
        '''A dict containing CPD data for the node.'''

    def choose(self, pvalues, method, regime, outcome):
        '''
        Randomly choose state of node from probability distribution conditioned on *pvalues*.
        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.
        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.
        The function creates a Gaussian distribution in the manner described in :doc:`lgbayesiannetwork`, and samples from that distribution, returning its outcome.
        
        '''
        warnings.filterwarnings("ignore", category=FutureWarning)
        pvalues = [outcome[t] for t in self.Vdataentry["parents"]] # ideally can we pull this from the skeleton so as not to store parent data at all?
        for pvalue in pvalues:
            assert pvalue != 'default', "Graph skeleton was not topologically ordered."
        
        if regime == 'mode':
            return self.Vdataentry["mode"]
        else:
            names = ['xgb', 'cb']
            regressor_list = [xgb.XGBRegressor(), cb.CatBoostRegressor(loss_function="RMSE")]
            classifier_dict = dict(zip(names, regressor_list))

            if self.Vdataentry["regressor"] in names:
                regressor = classifier_dict[self.Vdataentry["regressor"]]
            else:
                raise Exception('Incorrect regressor name')
            
            
            random.seed()
            sample = 0
            if regime == '50_predict':
                regressor_path = self.Vdataentry["regressor_path_50"]
            elif regime == '90_predict':
                regressor_path = self.Vdataentry["regressor_path_90"]
            elif regime == '10_predict':
                regressor_path = self.Vdataentry["regressor_path_10"]
            else:
                regressor_path = self.Vdataentry["regressor_path"]
            regressor_path = os.path.join(Path(__file__).parent.parent.parent, regressor_path)
            regressor = joblib.load(regressor_path)
            mean = regressor.predict(np.array(pvalues).reshape(1, -1))[0]
            variance = self.Vdataentry["variance"]
            if regime == 'sample':
                sample = random.gauss(mean, math.sqrt(variance))
            else:
                sample = mean
        return sample

    