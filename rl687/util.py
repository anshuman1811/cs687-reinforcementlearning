import numpy as np
from rl687.policies.tabular_softmax import TabularSoftmax
from rl687.policies.lfa_softmax import SoftmaxWithLFA
import matplotlib.pyplot as plt
from rl687.environments.gridworld import Gridworld
from rl687.environments.cartpole import Cartpole
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from multiprocessing import Condition
from multiprocessing.dummy import Pool

class GridworldEvaluation:
	def __init__ (self):
		self.returns = []
		self.curTrialReturns = []
		self.policy = TabularSoftmax(25, 4)
		self.numTrial = 0
		# self.lock = Condition()
		print("New Gridworld Eval")

	def endTrial(self):
		# print ("Incrementing Num Trial", self.numTrial)
		self.numTrial += 1
		self.returns.append(np.array(self.curTrialReturns))
		# self.lock.acquire()
		print("Number of returns collected", len(self.returns))
		# self.lock.release()
		self.curTrialReturns = []

	def __call__(self, policy:np.array, numEpisodes:int):
	    # print("Evaluating Gridworld")
	    G = []
	    policy=TabularSoftmax(25, 4)
	    policy.parameters = policy
	    env = Gridworld()
	    for ep in range (numEpisodes):
	        # print("Episode ", ep)
	        env.reset()
	        Gi = 0
	        timeStep = 0
	        while not env.isEnd:
	            state = env.state
	            action = policy.samplAction(state)
	            _, next_state, reward = env.step(action)
	            Gi += reward
	            timeStep += 1
	            if timeStep == 200:
	            	Gi += -50
	            	break
	        G.append(Gi)
	        self.curTrialReturns.append(Gi)
	    print("Mean Return ", np.mean(G))
	    return np.mean(G)

	def plot(self, filename:str="plot.png", title:str="plot", show:bool=False):
		fig, ax = plt.subplots()
		numEpisodes = len(self.returns[0])
		# print (self.numTrial, numEpisodes)
		self.returns = np.reshape(self.returns, (self.numTrial, numEpisodes))
		# print (self.returns.shape)
		ax.errorbar(np.arange(numEpisodes), np.mean(self.returns, axis=0), yerr=np.std(self.returns, axis=0), ecolor='gray', label='1 standard deviation')
		# plt.yscale('log')
		ax.set_xlabel('Num Episodes')
		ax.set_ylabel('Return')

		# ax.xaxis.set_major_locator(MultipleLocator(10000))
		ax.xaxis.set_minor_locator(AutoMinorLocator(5))
		ax.yaxis.set_major_locator(MultipleLocator(5))
		ax.yaxis.set_minor_locator(AutoMinorLocator(5))
		ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black')
		ax.grid(True, which='minor', linestyle=':', linewidth='0.2', color='red')
		ax.legend(loc=4)
		if show:
			ax.show()
		fig.suptitle(title, fontsize=10)
		fig.savefig(filename)

class CartPoleEvaluation:
	def __init__ (self, k:int):
		self.policy = SoftmaxWithLFA(4, 2, k)
		self.returns = []
		self.curTrialReturns = []
		self.numTrial = 0
		self.k = k
		self.lock = Condition()
		# print("New Cartpole Eval")

	def endTrial(self):
		print ("Incrementing Num Trial", self.numTrial)
		self.numTrial += 1
		self.returns.append(np.array(self.curTrialReturns))
		self.curTrialReturns = []
		# self.lock.acquire()
	    # self.lock.release()

	def __call__(self, parameters:np.array, numEpisodes:int):
	    # print("Evaluating Cartpole")
	    G = []
	    # self.policy.parameters = policy
	    policy = SoftmaxWithLFA(4, 2, self.k)
	    policy.parameters = parameters
	    # threadPool = Pool(numEpisodes)
	    # G = threadPool.map(self.runEpisode, [policy for i in range(numEpisodes)])
	    # print("G", G)
	    # self.curTrialReturns.extend(G)
	    env = Cartpole()
	    for ep in range (numEpisodes):
	        # print("Episode ", ep)
	        env.reset()
	        Gi = 0
	        while not env.isEnd:
	            state = env.state
	            action = policy.samplAction(state)
	            next_state, reward, _ = env.step(action)
	            Gi += reward
	        G.append(Gi)
	        self.lock.acquire()
	        self.curTrialReturns.append(Gi)
	        # print("Number of returns collected int trial", len(self.curTrialReturns))
	        self.lock.release()
	    # threadPool.close()
	    # threadPool.join()
	    print("Mean Return ", np.mean(G))
	    return np.mean(G)

	def runEpisode(self, policy):
		env = Cartpole()
		Gi = 0
		while not env.isEnd:
			state = env.state
			action = policy.samplAction(state)
			next_state, reward, _ = env.step(action)
			Gi += reward
		return Gi 

	def plot(self, filename:str="plot.png", title:str="plot", show:bool=False):
		fig, ax = plt.subplots()
		numEpisodes = len(self.returns[0])
		# print (self.numTrial, numEpisodes)
		returns = np.reshape(self.returns, (self.numTrial, numEpisodes))
		print ("Plotting Returns of shape ", returns.shape)
		ax.errorbar(np.arange(numEpisodes), np.mean(returns, axis=0), yerr=np.std(returns, axis=0), ecolor='gray', label='1 standard deviation')
		# plt.yscale('log')
		ax.set_xlabel('Num Episodes')
		ax.set_ylabel('Return')

		# ax.xaxis.set_major_locator(MultipleLocator(500))
		ax.xaxis.set_minor_locator(AutoMinorLocator(5))
		# ax.yaxis.set_major_locator(MultipleLocator(5))
		ax.yaxis.set_minor_locator(AutoMinorLocator(5))
		ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='black')
		ax.grid(True, which='minor', linestyle=':', linewidth='0.2', color='red')
		ax.legend(loc=4)
		if show:
			ax.show()
		fig.suptitle(title, fontsize=10)
		fig.savefig(filename)

class GAInit:
	def __init__(self, numParameters:int):
		self.numParameters = numParameters

	def __call__(self, populationSize:int):
	    print("Initializing GA Population")
	    return np.random.normal(0,1,(populationSize, self.numParameters))