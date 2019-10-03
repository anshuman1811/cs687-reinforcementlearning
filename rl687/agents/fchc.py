import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        #TODO
        self._name = "first_choice_hill_climbing"
        self.initTheta = theta
        self._parameters = theta
        self._sigma = sigma
        self._evaluationFunction = evaluationFunction
        self._numEpisodes = numEpisodes
        self.bestReturn = self._evaluationFunction(theta, numEpisodes)

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
        new_theta = np.random.multivariate_normal(self._parameters, (self._sigma*self._sigma)*np.identity(len(self._parameters)))
        # print (self._parameters, new_theta)
        J_new = self._evaluationFunction(new_theta, self._numEpisodes)
        returns.append (J_new)
        if J_new > self.bestReturn:
            print("Updating policy")
            self._parameters = new_theta
            self.bestReturn = J_new
        print (J_new)
        return self._parameters

    def reset(self)->None:
        #TODO
        self._parameters = self.initTheta
        self.bestReturn = self._evaluationFunction(self.initTheta, self._numEpisodes)
