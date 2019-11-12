from learningModule import LearningModule
import numpy as np, random
import copy
class Student(object):
	def __init__(self, Id, ability, variance, prior_plevel, budget, goal, all_actions, all_assessment_tests, first_try):
		self.id = Id
		self.ability = ability
		self.variance = variance
		self.cur_state = prior_plevel
		self.budget = budget

		self.goal = goal		
		self.exam_time = budget - int((sum(self.goal) - sum(self.cur_state))*first_try)

		self.priorAssessmentTest = LearningModule(0,"Prior Assessment Test", [], [] , {},[])
		self.finalExam = LearningModule(0,"Final Exam", [], [] , {},[])
		self.leanring_history = [(self.priorAssessmentTest, copy.deepcopy(self.cur_state))]
		self.observations = [(self.priorAssessmentTest, copy.deepcopy(self.cur_state))]
		
		self.unreach = list(range(len(self.cur_state)))
		self.updateUnreach(copy.deepcopy(self.cur_state))
		self.all_assessment_tests = all_assessment_tests
		self.all_actions = all_actions

		self.first_try = first_try
		self.maxBudget = budget

	def transition(self, learning_module):
		assert(isinstance(learning_module, LearningModule))
		if self.budget == 0 or self.unreach == []:
			return True
		else:
			c = self.ability + self.variance * np.random.normal()
			self.cur_state , observation = learning_module.Transition(self.cur_state, c)
			#print("dfghj",self.observations)
			self.observations.append((learning_module, copy.deepcopy(observation)))
			self.leanring_history.append((learning_module,  copy.deepcopy(self.cur_state)))
			self.budget -= 1
			self.updateUnreach(observation)

			return False

	def updateUnreach(self, observation):
		remove = []
		for i in self.unreach:
			if i in observation and observation[i] >= self.goal[i]:
				remove.append(i)
		for i in remove:
			self.unreach.remove(i)

	def list2str(self,state):
		s = ""
		for i in range(len(state)):
			s += str(state[i])
		return s 

	def chooseLearningMod(self):
		if self.budget <= self.exam_time:
			key = self.list2str(self.unreach)
			while key not in self.all_assessment_tests:
				delete = random.randrange(len(key))
				key = key[:delete] + key[delete+1:]
			return self.all_assessment_tests[key][random.randrange(len(self.all_assessment_tests[key]))]
		key = self.list2str(self.unreach)
		#print(self.all_actions)
		while key not in self.all_actions:
			delete = random.randrange(len(key))
			key = key[:delete] + key[delete+1:]
		return self.all_actions[key][random.randrange(len(self.all_actions[key]))]

	def ShowHistory(self):
		learning_modules, observations = list(zip(*self.observations))
		learning_modules, observations = list(learning_modules), list(observations)

		learning_modules.append(self.finalExam)
		observations.append(copy.deepcopy(self.cur_state))
		actions = []
		for learning_module in learning_modules:
			actions.append(learning_module.Name())

		return actions, observations

	def Gen_learning_episode(self):
		while True:
			#print(self.observations)
			action = self.chooseLearningMod()
			#print(self.observations)
			end_learning = self.transition(action)
			#print(self.observations)
			if end_learning:
				#print(self.ShowHistory())
				return self.ShowHistory()

	def Restart(self, prior_plevel):
		self.cur_state = prior_plevel
		self.cur_observation = prior_plevel
		self.budget = self.maxBudget
		self.exam_time = budget - int((sum(self.goal) - sum(self.prior_plevel))*self.first_try)
		self.leanring_history = [(PriorAssessmentTest, self.cur_state)]
		self.unreach = list(range(len(self.cur_state)))
		self.updateUnreach(self.cur_state)
	
	def GetId(self):
		return self.id

