import numpy as np
import sklearn.mixture as mixture
import math, pickle, json, copy, random, ast
import sys

sys.dont_write_bytecode = True
#Note: We assume that each action takes 1 time unit.

class GMMMDPSimulator():

	"""
	Arguments:
	- gmm_pickled: Name of a txt file containing a pickled GMM
	- transition_matrix: Name of a npy file containing transition matrix
	- action_index: Dictionary mapping action names to indices in transition_matrix
	- revealKC: Dictionary mapping action names to lists of proficiency levels to reveal
	"""
	def __init__(self, num_kc, num_proficiency_level, gmm_pickled, transition_matrix, action_index, revealKC):
		self.num_kc = num_kc
		self.num_proficiency_level = num_proficiency_level

		#Set up gmm and transition matrix
		try:
			gmm_file = open(gmm_pickled, "rb")
			self.initial_prof_distribution = pickle.load(gmm_file)
			gmm_file.close()
		except:
			import initial_proficiency_gmm
		self.P = np.load(transition_matrix)

		#Set up actions and proficiencies to reveal
		self.action_index = copy.deepcopy(action_index)
		self.revealed_proficiencies = copy.deepcopy(revealKC)

		self.cur_proficiency = None
		self.initial_proficiency = None
		self.reset()

	def make_valid_prof(self, l):
		l = np.minimum(l, self.num_proficiency_level - 1)
		l = np.maximum(l, 0)
		return (l + 0.5).astype(int)

	def reset(self):
		self.cur_proficiency = self.initial_prof_distribution.sample()[0][0]
		self.cur_proficiency = self.make_valid_prof(self.cur_proficiency)
		self.initial_proficiency = copy.deepcopy(self.cur_proficiency)

	def reset_prof(self, initial_proficiency):
		self.cur_proficiency = copy.deepcopy(initial_proficiency)
		self.initial_proficiency = copy.deepcopy(self.cur_proficiency)

	def get_initial_proficiency(self):
		return self.initial_proficiency

	def get_action_set(self):
		return self.revealed_proficiencies.keys()

	#Wherever necessary, returns proficiencies in relevant KCs
	#Returns None otherwise
	def make_action(self, action_name):
		#Update current proficiency by sampling from P
		#The following two functions are from parseDatasheet.py.
		def encode_state(state):
			res = 0
			for i in range(len(state)):
				res += state[i] * self.num_proficiency_level **(len(state)-1-i)
			return res

		def decode_state(num):
			res = [0] * self.num_kc
			for i in range(self.num_kc):
				res[-1-i] = num % self.num_proficiency_level
				num = num/self.num_proficiency_level
			return res

		#The following function is copied from simulator.py.
		def gen_random_output_index(l):
			ACCURACY = 0.001
			if 1 - sum(l) > ACCURACY:
				print("Invalid Input for gen_random_output_index!")
				return
			else:
				p = random.random()
				index = 0
				while p > 0 and index < len(l):
					p -=l[index]
					index += 1
				return index - 1

		if action_name == 'Prior Assessment Test':
			raise Exception("Prior Assessment Test: Not a valid action for sim 2!")
		if action_name == 'Final Exam':
			#Immediately go to terminal state and restart episode
			full_proficiencies = copy.deepcopy(self.cur_proficiency)
			self.reset()
			return full_proficiencies

		action_id = self.action_index[action_name]
		transition_prof = self.P[action_id][encode_state(self.cur_proficiency)]
		new_state_index = gen_random_output_index(transition_prof)
		new_prof = decode_state(new_state_index)
		self.cur_proficiency = new_prof

		#Reveal certain proficiencies, in the case of assessment tests
		prof_list = self.revealed_proficiencies[action_name]
		if len(prof_list) == 0:
			return None
		else:
			prof_dict = {}
			for kc in prof_list:
				prof_dict[kc] = self.cur_proficiency[kc]
			return prof_dict