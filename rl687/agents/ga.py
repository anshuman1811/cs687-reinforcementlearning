import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numTruncate:int=1, alpha:float=0.5, numEpisodes:int=10):
        print ("Init")
        print (populationSize, numElite, numEpisodes)
        self._name = "genetic_algorithm"
        self._population = initPopulationFunction(populationSize)
        self._parameters = np.zeros_like(self._population[1])
        self.bestReturn = -np.inf
    
        self.populationSize = populationSize
        self.evaluationFunction = evaluationFunction
        self.numElite = numElite
        self.numEpisodes = numEpisodes
        self.initPopulationFunction = initPopulationFunction

        # for candidate in self._population:
        #     print ("Evaluating Candidate")
        #     J = self.evaluationFunction(candidate, self.numEpisodes)
        #     print ("Return", J)
        #     if J > self.bestReturn:
        #         print ("Better Policy Found!")
        #         self.bestReturn = J
        #         self._parameters = candidate
        # print (self._population, self._parameters, self.bestReturn)

        self.numTruncate = numTruncate
        self.alpha = alpha

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._parameters

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        return parent + self.alpha*np.random.normal(0, 1, parent.shape)

    def getParents (self, Kp:int, sortedCandidates):
        return sortedCandidates[:Kp]

    def getChildren (self, alpha:float, parents):
        children = []
        for i in np.random.choice(len(parents), self.populationSize - self.numElite):
            children.append(self._mutate(parents[i]))
        return children

    def train(self)->np.ndarray:
        returns = []
        print ("Training")
        for candidate in self._population:
            print ("Evaluating Candidate")
            J = self.evaluationFunction(candidate, self.numEpisodes)
            print ("Return", J)
            returns.append ((candidate, J))
            if J > self.bestReturn:
                print ("Better Policy Found!")
                self.bestReturn = J
                self._parameters = candidate

        print("Sorting Candidates")
        sortedCandidates = [candidate[0] for candidate in sorted(returns, key=lambda tup:tup[1], reverse=True)]
        print("Getting Parents")
        parents = self.getParents(self.numTruncate, sortedCandidates)
        print("Extracting Elite Candidates")
        elite = sortedCandidates[:self.numElite]
        # print(len(elite))
        print("Getting Children")
        children = self.getChildren(self.alpha, parents)
        # print (len(children))
        self._population[:self.numElite] = elite
        self._population[self.numElite:] = children
        print ("New Population", len(self._population), self._population[0].shape)
        print ("Best params", np.array(sortedCandidates[0]).shape)
        return sortedCandidates[0]

    def reset(self)->None:
        print ("Resetting")
        self._population = self.initPopulationFunction(self.populationSize)
        self._parameters = np.zeros_like(self._population[1])
        self.bestReturn = -np.inf
        # for candidate in self._population:
        #     print ("Evaluating Candidate")
        #     J = self.evaluationFunction(candidate, self.numEpisodes)
        #     print ("Return", J)
        #     if J > self.bestReturn:
        #         print ("Better Policy Found!")
        #         self.bestReturn = J
        #         self._parameters = candidate