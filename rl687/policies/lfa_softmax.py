import numpy as np
from .skeleton import Policy
from typing import Union
from itertools import product

class SoftmaxWithLFA(Policy):
    """
    A Softmax with Linear Function Approximation Policy (bs)


    Parameters
    ----------
    stateSize (int): the number of elements describing the state in the policy
    numActions (int): the number of actions the policy has
    k (int): the order of the Fourier Basis used for linear function approximation
    """

    def __init__(self, stateSize:int, numActions: int, k:int=1):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._theta = np.zeros((numActions, np.power(k+1, stateSize)))
        
        #TODO
        self.stateSize = stateSize
        self.numActions = numActions
        self.k = k

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:np.ndarray, action=None)->Union[float, np.ndarray]:
        
        #TODO
        if action is None:
            return self.getActionProbabilities(state)
        else:
            return self.getActionProbabilities(state)[action]

    def getNormalizedState(self, state:np.ndarray):
        normalizedState = np.zeros_like(state)
        # x
        normalizedState[0] = (state[0]+3.0)/(6.0)
        # v
        normalizedState[1] = (state[1]+4.5)/(9.0)
        # theta
        normalizedState[2] = (state[2]+(np.pi/12.0))/(np.pi/6.0)
        # omega
        normalizedState[3] = (state[3]+4.0)/(8.0)
        # if np.any(normalizedState<0) or np.any(normalizedState>1):
        #     print ("State", state, "\nNormalized State", normalizedState)
        return normalizedState


    def getStateFeatures(self, state:np.ndarray):
        # print ("Getting Features for ", state)
        normalizedState = self.getNormalizedState(state)
        fourierBasisMask = np.array(list(product(np.arange(self.k+1), repeat=4)))
        # print (fourierBasisMask)
        # print ("Fourier Basis Mask ", fourierBasisMask.shape)
        fourierBasis = np.cos(np.sum(fourierBasisMask*normalizedState, axis=1))
        # print ("Fourier Basis ", fourierBasis.shape)
        return fourierBasis

    def samplAction(self, state:np.ndarray)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        
        #TODO
        return np.random.choice(self.numActions, p = self.getActionProbabilities(state))

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """

        #TODO
        features = self.getStateFeatures(state)
        # print("Features", features.shape)
        probabilities = self._theta.dot(features)
        probabilities -= np.max(probabilities)
        # print ("Action", probabilities)
        return np.exp(probabilities)/np.sum(np.exp(probabilities))
