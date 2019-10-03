import numpy as np
from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
import matplotlib.pyplot as plt
from rl687.util import GridworldEvaluation, CartPoleEvaluation, GAInit
from rl687.agents.fchc import FCHC
from rl687.agents.cem import CEM
from rl687.agents.ga import GA

def problem1():
    """
    Apply the CEM algorithm to the More-Watery 687-Gridworld. Use a tabular 
    softmax policy. Search the space of hyperparameters for hyperparameters 
    that work well. Report how you searched the hyperparameters, 
    what hyperparameters you found worked best, and present a learning curve
    plot using these hyperparameters, as described in class. This plot may be 
    over any number of episodes, but should show convergence to a nearly 
    optimal policy. The plot should average over at least 500 trials and 
    should include standard error or standard deviation error bars. Say which 
    error bar variant you used. 
    """

    #TODO
    print ("Problem 1")

    numStates = 25
    numActions = 4

    numTrials = 5
    numIters = 50
    numEps = 20
    popSize = 50

    numElite = 10
    epsilon = 1.5
    sigma = 0.1
    policyEval = GridworldEvaluation()
    agent = CEM(np.zeros(numStates*numActions), sigma, popSize, numElite, numEps, policyEval, epsilon)

    for trial in range (numTrials):
        print("Trial ", trial)
        for it in range(numIters):  
            print("Iteration ", it)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot()

def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    
    #TODO
    print ("Problem 2")

    num_states = 25
    num_actions = 4

    numTrials = 1
    numIters = 1000
    numEps = 100

    sigma = 0.8
    policyEval = GridworldEvaluation()

    print("Trials: %d\nIterations: %d\nEpisodes: %d\nSigma: %f" % (numTrials, numIters, numEps, sigma))

    agent = FCHC(np.zeros(num_states*num_actions), sigma, policyEval, numEpisodes = numEps)
    for trial in range (numTrials):
        print("Trial ", trial)
        for it in range(numIters):            
            print("Iteration ", it)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot()

def problem3():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this assignment) on the More-Watery 687-Gridworld domain. Report the same 
    quantities.
    """

    #TODO
    print ("Problem 3")

    # Environment Params
    num_states = 25
    num_actions = 4

    # Policy Search Params
    numTrials = 50
    numGenerations = 100
    populationSize = 30
    numEpisodes = 20

    numElite = 20
    numTruncate = 5
    alpha = 1.25
    policyEval = GridworldEvaluation()
    initGA = GAInit(num_states*num_actions)

    # print("Trials: %d\nIterations: %d\nEpisodes: %d\nSigma: %f" % (numTrials, numIters, numEps, sigma))

    agent = GA(populationSize, policyEval, initGA, numElite=numElite, numTruncate=numTruncate, alpha=alpha, numEpisodes = numEpisodes)
    for trial in range(numTrials):
        print("Trial ", trial)
        for gen in range(numGenerations):
            print("Generation ", gen)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot()


def problem4():
    """
    Repeat the previous question, but using the cross-entropy method on the 
    cart-pole domain. Notice that the state is not discrete, and so you cannot 
    directly apply a tabular softmax policy. It is up to you to create a 
    representation for the policy for this problem. Consider using the softmax 
    action selection using linear function approximation as described in the notes. 
    Report the same quantities, as well as how you parameterized the policy. 
    
    """

    #TODO
    print ("Problem 4")

    numStates = 25
    numActions = 4

    numEps = 10
    numIters = 50
    popSize = 25
    numElite = 5
    epsilon = 1.5
    sigma = 1.25

    policyEval = CartPoleEvaluation()
    agent = CEM(np.zeros(numStates*numActions), sigma, popSize, numElite, numEps, policyEval, epsilon)

    returnsOverTrials = []
    numTrials = 1
    for trial in range (numTrials):
        print ("Trial ", trial)
        for it in range(numIters) :            
            print("Iteration ", it)
            agent.train()
        agent.reset()
    policyEval.plot()

def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    #TODO
    pass

def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    
    #TODO
    pass

def main():
    np.random.seed(123)
    # problem1()
    # problem2()
    problem3()
    # problem4()
    
    #TODO
    pass


if __name__ == "__main__":
    main()
