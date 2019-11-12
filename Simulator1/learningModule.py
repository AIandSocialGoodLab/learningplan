import random, copy, numpy as np

class LearningModule(object):
	def __init__(self, num_kc, name_code, related_kc, reveal_kc, kc_distribution_params, kc_minimum_requirement):
		self.num_kc = num_kc
		self.name_code = name_code
		self.related_kc = related_kc
		self.reveal_kc = reveal_kc
		
		self.kc_minimum_requirement = kc_minimum_requirement
		self.kc_distribution_params = kc_distribution_params
	
	def __repr__(self):
		return self.name_code

	def getInc(self, kc,c):
		inc = int(self.kc_distribution_params[kc][0] + c + self.kc_distribution_params[kc][1] * np.random.normal())
		if inc < 0:
			inc = 0
		return inc

	def list2str(self,state):
		s = ""
		for i in range(len(state)):
			s += str(state[i])
		return s 
	def IsAssessmentTest(self):
		return len(self.reveal_kc) > 0

	def Key(self):
		print(self.name_code, self.related_kc)
		res = []
		def subset(l):
			if l == []:
				return [[]]
			else:
				res1 = subset(l[1:])
				res2 = subset(l[1:])
				res2 = [[l[0]]+ res2[i] for i in range(len(res2))]
				return res1+res2

		subsetKC = subset(self.related_kc)
		for _ in range(self.num_kc/len(self.related_kc)):
			for subset in subsetKC:
				res.append(self.list2str(subset))
		return res

	def Name(self):
		return self.name_code

	def Transition(self,state,c):
		new_state = copy.deepcopy(state)
		for kc in self.related_kc:
			if state[kc] >= self.kc_minimum_requirement[kc]:
				new_state[kc] = min(self.num_plevels - 1, state[kc] + self.getInc(kc,c))
			else:
				new_state =  copy.deepcopy(state)
				break
		show_state = copy.deepcopy(new_state)
		for i in range(len(new_state)):
			if i not in self.reveal_kc:
				show_state[i] = -1 

		d = dict()
		for i in range(len(show_state)):
			if show_state[i]!=-1:
				d[i] = show_state[i]
		return new_state, d


	def setNumPlevels(self, num_plevels):
		self.num_plevels = num_plevels



