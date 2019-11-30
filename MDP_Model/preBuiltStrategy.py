import copy, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
from parseDatasheet import encode_state, decode_state, str2list, state_match
sys.path.append("..")
sys.dont_write_bytecode = True
from Simulator2.gmm_mdp_simulator import GMMMDPSimulator
from Simulator1.simulator import Simulator1
from  naivePolicy import genNaivePolicy

class PreBuiltStrategy(object):
	def __init__(self, policy,  goal, num_kc, num_plevels, simulator = None):
		self.policy = policy
		self.i = 0 
		self.goal = goal
		self.num_kc = num_kc
		self.num_plevels = num_plevels
		self.simulator = simulator
		self.round = 0

	def genRandomInitialState(self, reset = True, p_level = None):
		if reset:
			self.state = np.zeros(self.num_plevels**self.num_kc)
			
			self.simulator.create_student(1000, self.goal)
			p_level = self.simulator.student.cur_state
		self.state[encode_state(p_level)] = 1.0
		self.initial_p  = p_level
		self.additive_finish = 0
		self.discount = 1.0
		self.budget = 0
		self.i = 0
		self.round = 0

	def reachGoal(self, p_levels):
		for i in range(self.num_kc):
			if p_levels[i] < self.goal[i]:
				return False
		return True


	def transition(self):
		action = self.policy[self.i]

		add =  (self.i + 1 == len(self.policy))
		self.round += add 
		self.i = (self.i + 1) % len(self.policy)

		observation = self.simulator.make_action(action)
		cont = True
		if self.simulator.student.cur_state == self.goal and self.round > 0:
			cont = False

		return cont, observation, add



	def GenerateLearningCurve(self,num_episodes):
		pass_steps = []
		for i in range(num_episodes):
			if i % 100 == 0:
				print "episode: %d" % i
			self.genRandomInitialState()
			store_state = copy.deepcopy(self.simulator.student.cur_state)
			cont = True
			observations = []
			actions = []
			cur_step = 0
			while cont:
				cur_step += 1
				cont, observation, add = self.transition()
				if add:
					cur_step += 1
				if cur_step > 100:
					break
			pass_steps.append(cur_step)
			self.simulator.student.cur_state = copy.deepcopy(store_state)
			self.simulator.student.budget = 100
			self.genRandomInitialState(reset = False, p_level= copy.deepcopy(store_state))
		
		res = dict()
		for val in set(pass_steps):
			res[val] = pass_steps.count(val)
		return res 






#simulator = GMMMDPSimulator(5, 3, "../Simulator2/gmm_pickled.txt", transition_matrix_path, action_index, revealKC)
simulator1 = Simulator1()

GOAL = [2,2,2,2,2]

policy = ["2", "AT12"]
generator = PreBuiltStrategy(policy,  GOAL, 5, 3, simulator = simulator1)
print generator.GenerateLearningCurve(10000)

