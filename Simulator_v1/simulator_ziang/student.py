from learningModule import LearningModule, PriorAssessmentTest, FinalExam
import numpy as np, random
class Student(object):
	def __init__(self, ability, variance, prior_plevel, budget, goal, all_actions, all_assessment_tests, first_try):
		self.ability = ability
		self.variance = variance
		self.cur_state = prior_plevel
		self.cur_observation = prior_plevel
		self.budget = budget
		self.exam_time = budget - int((sum(self.goal) - sum(self.prior_plevel))*first_try)
		self.goal = goal
		self.leanring_history = [(PriorAssessmentTest, self.cur_state)]
		self.observation = [(PriorAssessmentTest, self.cur_state)]
		
		self.unreach = list(range(len(self.cur_state)))
		self.updateUnreach()
		self.all_assessment_tests = all_assessment_tests
		self.all_actions = all_actions

	def transition(self, learning_module):
		assert(isinstance(learning_module, LearningModule))
		if budget == 0 or self.unreach == []:
			return False
		else:
			c = self.ability + self.variance * np.random.normal()
			self.cur_state , self.cur_observation = learning_module.Transition(self.cur_state, c)
			self.observation.append((learning_module, self.cur_observation))
			self.leanring_history.append((learning_module, self.cur_state))
			self.budget -= 1
			self.updateUnreach()
			return True

	def updateUnreach(self):
		remove = []
		for learning_module in self.unreach:
			if self.cur_observation[1][i] >= self.goal[i]:
				remove.append(i)
		for learning_module in remove:
			self.unreach.remove(i)

	def list2str(self,state):
		s = ""
		for i in range(len(state)):
			s += str(state[i])
		return s 

	def chooseLearningMod(self):
		if self.budget <= self.exam_time:
			key = list2str(unreach)
			while key not in self.all_assessment_tests:
				delete = random.randrange(len(key))
				key = key[:delete] + key[delte+1:]
			return self.all_assessment_tests[random.randrange(len(self.all_assessment_tests[key]))]
		key = list2str(unreach)
		while key not in self.all_actions:
			delete = random.randrange(len(key))
			key = key[:delete] + key[delete+1:]
		return self.all_actions[random.randrange(len(self.all_actions[key]))]

	def ShowHistory(self):
		learning_modules, observations = list(zip(*self.observations))
		learning_modules.append(FinalExam())
		observations.append(cur_state)
		actions = []
		for learning_module in learning_modules:
			actions.append(learning_module.Name())

		return actions, observations

	def Gen_learning_episode(self):
		while True:
			action = self.chooseLearningMod()
			end_learning = self.transition(action)
			if end_learning:
				return self.ShowHistory()

	def Restart(self, ability = self.ability, variance = self.variance, 
		prior_plevel, budget = self.budget, goal = self.goal, first_try = self.first_try):
		self.ability = ability
		self.variance = variance
		self.cur_state = prior_plevel
		self.cur_observation = prior_plevel
		self.budget = budget
		self.exam_time = budget - int((sum(self.goal) - sum(self.prior_plevel))*first_try)
		self.goal = goal
		self.leanring_history = [(PriorAssessmentTest, self.cur_state)]
		self.observation = [(PriorAssessmentTest, self.cur_state)]
		self.unreach = list(range(len(self.cur_state)))
		self.updateUnreach()
	

