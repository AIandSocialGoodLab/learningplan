from student import Student
from learningModule import LearningModule
import numpy as np, pandas as pd , random

class Generator(object):
	def __init__(self, num_kc, num_plevels, num_students, prior_plevel_std, student_learning_params):
		self.num_kc = num_kc
		self.num_plevels = num_plevels
		self.num_students = num_students
		self.action_list = []
		self.all_actions = dict()
		self.all_assessment_tests = dict()
		self.prior_plevel_std = prior_plevel_std
		self.amean,self.astd,self.stdmean,self.stdstd = student_learning_params
		self.outputActions = {"action":[], "related_kc":[]}

	def addAction(self,learning_module):
		learning_module.setNumPlevels(self.num_plevels)
		self.action_list.append(learning_module)
		keys = learning_module.Key()
		self.outputActions["action"].append(learning_module.Name())
		self.outputActions["related_kc"].append(learning_module.related_kc)
		for key in keys:
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

	def formLearningModule(self, name_code, assessment = False):
		related_kc = []
		for i in range(self.num_kc):
			if random.random() > 0.7:
				related_kc.append(i)
		if related_kc == []:
			related_kc = [random.randrange(self.num_kc)] 
		kc_distribution_params = dict()
		kc_minimum_requirement = dict()
		if assessment:
			base = 0.4
		else:
			base = 0.6
		for kc in related_kc:
			kc_distribution_params[kc] = (base + 0.5/len(related_kc), random.random())
			kc_minimum_requirement[kc] = min(self.num_plevels - 1, int(abs(np.random.normal() * 0.3)))
		if assessment:
			reveal_kc = related_kc
		else:
			reveal_kc = []
		if assessment:
			name_code = "AT"+name_code
		return LearningModule(self.num_kc, name_code, related_kc, reveal_kc, kc_distribution_params, kc_minimum_requirement)

	def createStudent(self,id, ability, std, first_try):
		prior_plevel = [min(2, int(np.abs(np.random.normal())*self.prior_plevel_std)) for _ in range(self.num_kc)]
		self.student = Student(id, ability, std, prior_plevel, self.budget, self.goal, self.all_actions, self.all_assessment_tests, first_try)

	def genEpisodeStudent(self):
		action, observation = self.student.Gen_learning_episode()
		return [self.student.GetId()]*len(action), action, observation

	def genAbilityStd(self):
		ability = np.abs(self.amean + self.astd * np.random.normal())
		std = np.abs(self.stdmean + self.stdstd*np.random.normal())
		return ability, std 

	def GenDataSet(self, budget, goal, output_path = "../dataset/MDPdatasheet.csv", action_path = "../dataset/action.csv"):
		self.budget = budget
		self.goal = goal
		actions, observations, ids = [], [], []
		for student_id in range(self.num_students):
			print student_id
			ability, std = self.genAbilityStd()
			first_try = np.random.uniform(0.7, 0.9)
			self.createStudent(student_id, ability, std, first_try)
			cur_id, action, observation = self.genEpisodeStudent()
			actions += action
			ids += cur_id
			observations += observation
		dataset = pd.DataFrame({'Student_ID':ids,
								'Action_Types': actions,
								'Cur_Proficiency': observations})
		dataset.to_csv(path_or_buf = output_path, columns=["Student_ID", "Action_Types","Cur_Proficiency"], index = False)
		actionset = pd.DataFrame(self.outputActions)
		actionset.to_csv(path_or_buf = action_path, index = False)


	def GenLearningModules(self,num_models, num_assess_tests):
		for i in range(num_models):
			self.addAction(self.formLearningModule(str(i + 1)))
		for j in range(num_assess_tests):
			self.addAction(self.formLearningModule(str(num_models + j + 1), assessment = True))



generator = Generator(num_kc = 5, num_plevels = 3, num_students = 5000, prior_plevel_std = 0.8, student_learning_params = [0.2, 0.6, 0.8, 0.2])
generator.GenLearningModules(10, 10)
print generator.all_actions
generator.GenDataSet(8, [2,2,2,2,2])
