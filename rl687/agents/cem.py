import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean theta and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """
    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):
        #TODO
        self._name = "Cross_Entropy_Method"
        self._theta = theta
        self._Sigma = (sigma)*np.identity(len(theta))

        self._parameters = theta
        self.initSigma = sigma
        self.initTheta = theta
        self.popSize = popSize
        self.numElite = numElite
        self.evaluationFunction = evaluationFunction
        self.numEpisodes = numEpisodes
        self.epsilon = epsilon

        self.bestJ = -np.inf

    @property
    def name(self)->str:
        #TODO
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._parameters

    def train(self)->np.ndarray:
        #TODO
        print("----Training----")
        returns = []
        for k in range(self.popSize):
            theta = np.random.multivariate_normal(self._theta, self._Sigma)
            # print (self._parameters, new_theta)
            J = self.evaluationFunction(theta, self.numEpisodes)
            returns.append((theta, J))
        
        # print("Returns", returns)
        topReturns = sorted(returns, key=lambda tup:tup[1], reverse = True)[:self.numElite]
        # print("Best Returns", topReturns)

        bestTheta, bestReturn = topReturns[0]
        if bestReturn > self.bestJ:
            self.bestJ = bestReturn
            self._parameters = bestTheta
        
        newTheta = np.zeros(self._theta.shape)
        for t, r in topReturns:
            newTheta += np.array(t)
        newTheta /= self.numElite
        # print (newTheta.shape)

        newSigma = self.epsilon*np.identity(len(self._theta))
        for t, r in topReturns:
            tmp = np.reshape(t-newTheta, (len(self._theta), 1))
            # print ("tmp", tmp.shape)
            tmp2 = tmp.dot(tmp.T)
            # print ("tmp2", tmp2.shape)
            # print (tmp2)
            newSigma += tmp.dot(tmp.T)
            # print ("newSigma", newSigma.shape)
        newSigma /= (self.numElite + self.epsilon)
        # print(newSigma)
        self._theta = newTheta
        self._Sigma = newSigma

        return bestTheta

    def reset(self)->None:
        #TODO
        self._parameters = self.initTheta
        self._theta = self.initTheta
        self._Sigma = (self.initSigma)*np.identity(len(self._theta))
