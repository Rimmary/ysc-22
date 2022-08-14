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
This module contains tools for representing linear Gaussian nodes -- those with a continuous linear Gaussian distribution of outcomes and a finite number of linear Gaussian parents -- as class instances with their own *choose* method to choose an outcome for themselves based on parent outcomes.

'''

import random
import math
from scipy.spatial import distance
import numpy as np
from sklearn.mixture import GaussianMixture
from pomegranate import DiscreteDistribution
from gmr import GMM
class Lg():
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
        sample = 0
        if regime == 'sample' or regime == 'mean':
            random.seed()
            
            if method == 'simple':
                # calculate Bayesian parameters (mean and variance)
                mean = self.Vdataentry["mean_base"]
                if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                    for x in range(len(self.Vdataentry["parents"])):
                        if (pvalues[x] != "default"):
                            mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
                        else:
                            print ("Attempted to sample node with unassigned parents.")

                variance = self.Vdataentry["variance"]
                if regime == 'mean':
                    sample = mean
                elif regime == 'sample':
                    sample = random.gauss(mean, math.sqrt(variance))
            else:
                mean = self.Vdataentry["mean_base"]
                variance = self.Vdataentry["variance"]
                w = self.Vdataentry["mean_scal"]
                n_comp = len(self.Vdataentry["mean_scal"])
                if n_comp != 0:
                    if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                        indexes = [i for i in range (1, (len(self.Vdataentry["parents"])+1), 1)]
                        if not np.isnan(np.array(pvalues)).all():
                            gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                            cond_gmm = gmm.condition(indexes, [pvalues])
                            if regime == 'mean':
                                
                                sample = gmm.predict(indexes, [pvalues])[0][0]
                            else:
                                sample = cond_gmm.sample(1)[0][0]
                        else:
                            sample = np.nan
                    else:
                        gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                        if regime == 'mean':
                            sample = np.dot(w, mean)[0]
                        else:
                            sample = gmm.sample(1)[0][0]
                else:
                    sample = np.nan
        if regime == '50_predict':
            random.seed()
            
            if method == 'simple':
                # calculate Bayesian parameters (mean and variance)
                mean = self.Vdataentry["mean_base"]
                if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                    for x in range(len(self.Vdataentry["parents"])):
                        if (pvalues[x] != "default"):
                            mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
                        else:
                            print ("Attempted to sample node with unassigned parents.")

                #variance = self.Vdataentry["variance"]
                sample = mean#random.gauss(mean, math.sqrt(variance))
            else:
                mean = self.Vdataentry["mean_base"]
                variance = self.Vdataentry["variance"]
                w = self.Vdataentry["mean_scal"]
                n_comp = len(self.Vdataentry["mean_scal"])
                if n_comp != 0:
                    if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                        indexes = [i for i in range (1, (len(self.Vdataentry["parents"])+1), 1)]
                        if not np.isnan(np.array(pvalues)).all():
                            gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                            cond_gmm = gmm.predict(indexes, [pvalues])[0][0]
                            sample = cond_gmm
                        else:
                            sample = np.nan
                    else:
                        gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                        sample = mean[w.index(max(w))][0]
                       
                else:
                    sample = np.nan
        elif regime == 'mode':
            return self.Vdataentry["mode"]
        if regime == '10_predict':
            random.seed()
            
            if method == 'simple':
                # calculate Bayesian parameters (mean and variance)
                mean = self.Vdataentry["mean_base"]
                if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                    for x in range(len(self.Vdataentry["parents"])):
                        if (pvalues[x] != "default"):
                            mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
                        else:
                            print ("Attempted to sample node with unassigned parents.")

                variance = self.Vdataentry["variance"]
                sample = mean - 2*math.sqrt(variance)#random.gauss(mean, math.sqrt(variance))
            else:
                mean = self.Vdataentry["mean_base"]
                variance = self.Vdataentry["variance"]
                w = self.Vdataentry["mean_scal"]
                n_comp = len(self.Vdataentry["mean_scal"])
                if n_comp != 0:
                    if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                        indexes = [i for i in range (1, (len(self.Vdataentry["parents"])+1), 1)]
                        if not np.isnan(np.array(pvalues)).all():
                            gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                            cond_gmm = gmm.condition(indexes, [pvalues])
                            sample_pred = gmm.predict(indexes, [pvalues])[0][0]
                            sigma = 0
                            for l,m in enumerate(cond_gmm.covariances):
                                sigma += cond_gmm.priors[l] * m[0][0]
                            sample = sample_pred - 2*math.sqrt(sigma)
                        else:
                            sample = np.nan
                    else:
                        gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                        sample = mean[w.index(max(w))][0] - 2*math.sqrt(variance[w.index(max(w))][0][0])
                       
                else:
                    sample = np.nan


        if regime == '90_predict':
            random.seed()
            
            if method == 'simple':
                # calculate Bayesian parameters (mean and variance)
                mean = self.Vdataentry["mean_base"]
                if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                    for x in range(len(self.Vdataentry["parents"])):
                        if (pvalues[x] != "default"):
                            mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
                        else:
                            print ("Attempted to sample node with unassigned parents.")

                variance = self.Vdataentry["variance"]
                sample = mean + 2*math.sqrt(variance)#random.gauss(mean, math.sqrt(variance))
            else:
                mean = self.Vdataentry["mean_base"]
                variance = self.Vdataentry["variance"]
                w = self.Vdataentry["mean_scal"]
                n_comp = len(self.Vdataentry["mean_scal"])
                if n_comp != 0:
                    if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                        indexes = [i for i in range (1, (len(self.Vdataentry["parents"])+1), 1)]
                        if not np.isnan(np.array(pvalues)).all():
                            gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                            cond_gmm = gmm.condition(indexes, [pvalues])
                            sample = gmm.predict(indexes, [pvalues])[0][0]
                            sigma = 0
                            for l,m in enumerate(cond_gmm.covariances):
                                sigma += cond_gmm.priors[l] * m[0][0]
                            sample = sample + 2*math.sqrt(sigma)
                        else:
                            sample = np.nan
                    else:
                        gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                        sample = mean[w.index(max(w))][0] + 2*math.sqrt(variance[w.index(max(w))][0][0])
                       
                else:
                    sample = np.nan
            
            
        return sample

    def choose_gmm(self, pvalues, outcome):
        '''
        Randomly choose state of node from probability distribution conditioned on *pvalues*.
        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.
        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.
        The function creates a Gaussian distribution in the manner described in :doc:`lgbayesiannetwork`, and samples from that distribution, returning its outcome.
        
        '''
        random.seed()

        # calculate Bayesian parameters (mean and variance)
        s = 0
        mean = self.Vdataentry["mean_base"]
        variance = self.Vdataentry["variance"]
        w = self.Vdataentry["mean_scal"]
        n_comp = len(self.Vdataentry["mean_scal"])
        indexes = [i for i in range (1, (len(self.Vdataentry["parents"])+1), 1)]
        if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
            # for x in range(len(self.Vdataentry["parents"])):
            #     if (pvalues[x] != "default"):
            #         X.append(pvalues[x])
            #     else:
            #         print ("Attempted to sample node with unassigned parents.")
            if not np.isnan(np.array(pvalues)).any():
                gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
                s = gmm.predict(indexes, [pvalues])[0][0]
            else:
                s = np.nan
        else:
            gmm = GMM(n_components=n_comp, priors=w, means=mean, covariances=variance)
            s = gmm.sample(1)[0][0]
        

        

        # draw random outcome from Gaussian
        # note that this built in function takes the standard deviation, not the
        # variance, thus requiring a square root
        return s  

    def choose_mix(self, pvalues, outcome):
        '''
        Randomly choose state of node from probability distribution conditioned on *pvalues*.

        This method has two parts: (1) determining the proper probability
        distribution, and (2) using that probability distribution to determine
        an outcome.

        Arguments:
            1. *pvalues* -- An array containing the assigned states of the node's parents. This must be in the same order as the parents appear in ``self.Vdataentry['parents']``.

        The function creates a Gaussian distribution in the manner described in :doc:`lgbayesiannetwork`, and samples from that distribution, returning its outcome.
        
        '''
        random.seed()

        # calculate Bayesian parameters (mean and variance)
        #mean = self.Vdataentry["mean_base"]
        # mean_node = 0
        # mean_node = 1
        mean = self.Vdataentry["mean_base"]
        variance = self.Vdataentry["variance"]
        parents = []
        parents_mean = []
        if isinstance(mean, list):
            if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                for x in range(len(self.Vdataentry["parents"])):
                    if (pvalues[x] != "default"):
                        parents.append(pvalues[x])
                        parents_mean.append(self.Vdataentry["mean_scal"][x])
                    else:
                        print ("Attempted to sample node with unassigned parents.")
                if len(parents) == 1:
                    if str(parents[0]) != 'nan':
                        dists = []
                        for vector in parents_mean:
                            dists.append(distance.euclidean(parents, vector))
                        label = dists.index(min(dists))
                        mean_node = mean[label]
                        variance_node = variance[label]
                    else:
                        mean_node = mean[random.randint(0,4)]
                        variance_node = variance[random.randint(0,4)]
                else:
                    if np.nan in parents:
                        if parents.count(np.nan) < len(parents):
                            nan_index = [i for i,d in enumerate(parents) if str(d)=='nan']
                            dists = []
                            for vector in parents_mean:
                                dists.append(distance.euclidean([p for p in parents if parents.index(p) not in nan_index], [p for p in vector if vector.index(p) not in nan_index]))
                                label = dists.index(min(dists))
                                mean_node = mean[label]
                                variance_node = variance[label]
                        else:
                            mean_node = mean[random.randint(0,4)]
                            variance_node = variance[random.randint(0,4)]
                    else:
                        dists = []
                        for vector in parents_mean:
                            dists.append(distance.euclidean(parents, vector))
                        label = dists.index(min(dists))
                        mean_node = mean[label]
                        variance_node = variance[label]



            
        else:
            if (self.Vdataentry["parents"] != None) and (len(self.Vdataentry["parents"]) != 0):
                for x in range(len(self.Vdataentry["parents"])):
                    if (pvalues[x] != "default"):
                        mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
                    else:
                        print ("Attempted to sample node with unassigned parents.")
            mean_node = mean
            variance_node = variance


       

        # if (self.Vdataentry["parents"] != None):
        #     for x in range(len(self.Vdataentry["parents"])):
        #         if (pvalues[x] != "default"):
        #             mean += pvalues[x] * self.Vdataentry["mean_scal"][x]
        #         else:
        #             print ("Attempted to sample node with unassigned parents.")

        #variance = self.Vdataentry["variance"]

        # draw random outcome from Gaussian
        # note that this built in function takes the standard deviation, not the
        # variance, thus requiring a square root
        return random.gauss(mean_node, math.sqrt(variance_node))          
