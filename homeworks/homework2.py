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

    numTrials = 50
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
    policyEval.plot('learningCurve_gridworld_CEM.png', "Learning Curve - Gridworld with CEM Agent")

def problem2():
    """
    Repeat the previous question, but using first-choice hill-climbing on the 
    More-Watery 687-Gridworld domain. Report the same quantities.
    """
    
    #TODO
    print ("Problem 2")

    num_states = 25
    num_actions = 4

    numTrials = 50
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
    policyEval.plot('learningCurve_gridworld_FCHC.png', "Learning Curve - Gridworld with FCHC Agent")

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
    policyEval.plot('learningCurve_gridworld_GA.png', "Learning Curve - Gridworld with GA Agent")


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

    m = 4
    numActions = 2

    numTrials = 50
    numEps = 10
    numIters = 50
    popSize = 20
    numElite = 5

    epsilon = 1.25
    sigma = 0.25
    k=3
    policyEval = CartPoleEvaluation(k=k)
    # print ("Size of theta = ", numActions*np.power(k+1, m))
    agent = CEM(np.zeros(numActions*np.power(k+1, m)), sigma, popSize, numElite, numEps, policyEval, epsilon)

    for trial in range (numTrials):
        print ("Trial ", trial)
        for it in range(numIters) :            
            print("Iteration ", it)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot('learningCurve_cartpole_CEM.png', "Learning Curve - Cartpole with CEM Agent")

def problem5():
    """
    Repeat the previous question, but using first-choice hill-climbing (as 
    described in class) on the cart-pole domain. Report the same quantities 
    and how the policy was parameterized. 
    
    """
    #TODO
    print ("Problem 5")

    m = 4
    numActions = 2

    numTrials = 50
    numIters = 200
    numEps = 10

    sigma = 0.8
    k=3
    policyEval = CartPoleEvaluation(k=k)

    print("Trials: %d\nIterations: %d\nEpisodes: %d\nSigma: %f" % (numTrials, numIters, numEps, sigma))

    agent = FCHC(np.zeros(numActions*np.power(k+1, m)), sigma, policyEval, numEpisodes = numEps)
    for trial in range (numTrials):
        print("Trial ", trial)
        for it in range(numIters):            
            print("Iteration ", it)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot('learningCurve_cartpole_FCHC.png', "Learning Curve - Cartpole with FCHC Agent")

def problem6():
    """
    Repeat the previous question, but using the GA (as described earlier in 
    this homework) on the cart-pole domain. Report the same quantities and how
    the policy was parameterized. 
    """
    
    #TODO
    print ("Problem 6")

    # Environment Params
    m = 4
    numActions = 2

    # Policy Search Params
    numTrials = 50
    numGenerations = 100
    populationSize = 30
    numEpisodes = 20

    numElite = 20
    numTruncate = 5
    alpha = 1.25
    k=3
    policyEval = CartPoleEvaluation(k=k)
    initGA = GAInit(numActions*np.power(k+1, m))

    # print("Trials: %d\nIterations: %d\nEpisodes: %d\nSigma: %f" % (numTrials, numIters, numEps, sigma))

    agent = GA(populationSize, policyEval, initGA, numElite=numElite, numTruncate=numTruncate, alpha=alpha, numEpisodes = numEpisodes)
    for trial in range(numTrials):
        print("Trial ", trial)
        for gen in range(numGenerations):
            print("Generation ", gen)
            agent.train()
        policyEval.endTrial()
        agent.reset()
    policyEval.plot('learningCurve_cartpole_GA.png', "Learning Curve - Cartpole with GA Agent")

def main():
    np.random.seed(123)
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
    problem6()
    
    #TODO
    pass


if __name__ == "__main__":
    main()
