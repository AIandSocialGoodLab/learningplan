import random
import numpy as np

class LearningModule(object):
	def __init__(self, name_code, related_kc, reveal_kc, kc_distribution_params, kc_minimum_requirement):
		self.name_code = name_code
		self.related_kc = related_kc
		self.reveal_kc = reveal_kc
		self.key = self.list2str(related_kc)
		self.kc_minimum_requirement = kc_distribution_params
		self.kc_distribution_params = kc_distribution_params


	def getInc(self, kc):
		inc = round(kc_distribution_params[kc][0] +  kc_distribution_params[kc][1] * np.random.normal())
		if inc < 0:
			inc = 0
		return inc

	def list2str(self,state):
		s = ""
		for i in range(len(state)):
			s += str(state[i])
		return s 
	def IsAssessmentTest(self):
		return len(reveal_kc) > 0

	def Key(self):
		return self.key

	def Name(self):
		return self.name_code

	def Transition(self,state):
		new_state = state
		for kc in self.related_kc:
			if state[kc] >= kc_minimum_requirement[kc]:
				new_state[kc] = state[kc] + self.getInc(kc)
			else:
				return state
		return new_state


class PriorAssessmentTest(LearningModule):
	def __init__(self):
		super().__init__("Prior Assessment Test")

class FinalExam(LearningModule):
	def __init__(self):
		super().__init__("Final Exam")
