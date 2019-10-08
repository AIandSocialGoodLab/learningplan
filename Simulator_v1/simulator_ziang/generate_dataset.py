from student import Student
from learningModule import LearningModule

class Generator(object):
	def __init__(self, num_kc, num_plevels, num_students, goal, actions):
		self.num_kc = num_kc
		self.num_plevels = num_plevels
		self.num_students = num_students
		self.goal = goal
		self.action_list = []
		self.all_actions = dict()
		self.all_assessment_tests = dict()
	

	def Add_action(self,learning_module):
		self.action_list.append(learning_module)
		key = learning_module.Key()
		if key in self.all_actions:
			self.all_actions[key].append(learning_module)
		else:
			self.all_actions[key] = [learning_module]

		isAssess = learning_module.IsAssessmentTest()
		if isAssess:
			if key in self.all_assessment_tests:
				self.all_assessment_tests[key].append(learning_module)
			else:
				self.all_assessment_tests[key] = [learning_module]

	def CreateStudent(ability, stability, prior_plevel, budget, goal, first_try):
		self.student = Student(ability, stability, prior_plevel, budget, goal, all_actions, all_assessment_tests, first_try)
