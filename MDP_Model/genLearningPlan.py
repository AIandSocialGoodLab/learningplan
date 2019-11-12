import copy, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
from parseDatasheet import encode_state, decode_state, str2list, state_match
sys.path.append("..")
sys.dont_write_bytecode = True
from Simulator2.gmm_mdp_simulator import GMMMDPSimulator
from  naivePolicy import genNaivePolicy

transition_matrix_path = "../policy/P.npy"
action_path = "../dataset/action.csv"
policy_path = "../policy/"

GOAL = [2,2,2,2,2]
actionset = pd.read_csv(action_path)
revealKC = dict()
for row in range(actionset.shape[0]):
	revealKC[actionset["action"][row]] = str2list(actionset["related_kc"][row])
ACTION_LIST = []
for action in revealKC:
	ACTION_LIST.append(action)

ACTION_LIST = sorted(ACTION_LIST)
print "All actions:", ACTION_LIST


try:
	P = np.load(transition_matrix_path)
	print("successfully load p!")
except:
	print "no transition probability has been made!"
	exit(1)

file_name = policy_path = "../policy/"+str(encode_state(GOAL))+".npy"
try:
	policy = np.load(file_name)
	print("successfully load policy!")
except:
	print "no policy for this state is available"
	exit(1)


class LeanringPlanGenerator(object):
	def __init__(self, policy, transition_p, maximum_step, goal, num_kc, num_plevels, actions, alpha = 0.75, simulator = None, budget = 10, nondeterministic_policy = None):
		self.policy = policy
		self.nondeterministic_policy = nondeterministic_policy
		self.transition_p = transition_p
		self.maximum_step = maximum_step
		self.goal = goal
		self.num_kc = num_kc
		self.num_plevels = num_plevels
		self.actions = actions
		self.simulator = simulator
		self.alpha = alpha	
		self.budget = budget

	def SetStep(self, maximum_step):
		self.maximum_step = maximum_step

	def genRandomInitialState(self):
		p_level = [0]*(self.num_kc)
		for i in range(self.num_kc):
			p_level[i] = int(3* random.random()**2)
		self.state = np.zeros(self.num_plevels**self.num_kc)
		self.state[encode_state(p_level)] = 1.0
		self.simulator.reset_prof(p_level)
		self.initial_p  = p_level
		self.additive_finish = 0
		self.discount = 1.0


	def reachGoal(self, p_levels):
		for i in range(self.num_kc):
			if p_levels[i] < self.goal[i]:
				return False
		return True


	def transition(self, nondeterministic = False):
		actions = np.zeros(len(self.actions) + 1)
		p = 0 
		for state in range(len(self.state)):
			if self.reachGoal(decode_state(state)):
				p += self.state[state]
				self.state[state] = 0
		if np.sum(self.state) == 0:
			return False, self.simulator.make_action('Final Exam'), 'Final Exam'
		self.state = self.state/(np.sum(self.state))
		self.additive_finish += p * self.discount
		if self.additive_finish > self.alpha:
			return False, self.simulator.make_action('Final Exam'), 'Final Exam'


		self.discount *= (1 - p)
		for state in range(len(self.state)):
			if self.state[state]!= 0:
				if nondeterministic:
					cur_action = self.nondeterministic_policy[state]

					actions += self.state[state] * cur_action

				else:
					cur_action = self.policy[state]
					actions[cur_action] += self.state[state]
		l = np.argwhere(actions == np.amax(actions)).flatten()

		action = len(self.actions)
		while action == len(self.actions):
			action = l[random.randrange(len(l))]
		
		action_name = self.actions[action]
		cur_state = np.zeros(self.num_plevels**self.num_kc)
		for state in range(len(self.state)):
			if self.state[state]!=0:
				new_state = self.transition_p[action][state][:-1]
				cur_state += (new_state * self.state[state])
		self.state = cur_state 

		observation = self.simulator.make_action(action_name)


		observation_state = [-1]*self.num_kc
		for i in observation:
			observation_state[i] = observation[i]

		for state in range(len(self.state)):
			if self.state[state] > 0 and not state_match(observation_state, decode_state(state)):
				self.state[state] = 0
		if np.sum(self.state) == 0:
			return False, self.simulator.make_action('Final Exam'), 'Final Exam'
		self.state = self.state/(np.sum(self.state))

		return True, observation_state, action_name


	def GenerateOneEpisode(self, num_episodes, enable_print = False, nondeterministic = False):
		result = []
		for i in range(num_episodes):
			self.genRandomInitialState()
			cont = True
			observations = []
			actions = []
			step = 0
			while cont and step < self.budget:
				step += 1
				cont, observation, action = self.transition(nondeterministic)
				observations.append(observation)
				actions.append(action)
			if enable_print:
				print self.initial_p, self.reachGoal(observations[-1]), actions
			result.append((self.initial_p, self.reachGoal(observations[-1]), actions))
		return result

	def Point_summary(self, num_episodes, nondeterministic = False):
		res = self.GenerateOneEpisode(num_episodes, nondeterministic = nondeterministic)
		count_pass = 0
		num_steps = 0
		for _, x, process in res:
			count_pass += x 
			num_steps += len(process)
		pass_rate = count_pass/float(num_episodes)
		avg_step = num_steps/float(num_episodes)
		return pass_rate, avg_step


action_index = dict()
for i in range(len(ACTION_LIST)):
	action_index[ACTION_LIST[i]] = i




num_episodes = 5000
alphas = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

simulator = GMMMDPSimulator(5, 3, "../Simulator2/gmm_pickled.txt", transition_matrix_path, action_index, revealKC)
random_policy = genNaivePolicy(GOAL, 5, 3, revealKC, ACTION_LIST)
from datetime import date
today = date.today()


for budget in range(6, 14):
	x, y = [], []
	xr, yr = [], []
	for alpha in alphas:
		generator = LeanringPlanGenerator(policy, P, 10, GOAL, 5, 3, ACTION_LIST, alpha = alpha, simulator = simulator, budget = budget, nondeterministic_policy = random_policy)
		random_generator = LeanringPlanGenerator(policy, P, 10, GOAL, 5, 3, ACTION_LIST, alpha = alpha, simulator = simulator, budget = budget, nondeterministic_policy = random_policy)
		p_rate, avg_step = generator.Point_summary(num_episodes)
		p_rate_r, avg_step_r = generator.Point_summary(num_episodes,True)
		x.append(avg_step)
		y.append(p_rate)
		xr.append(avg_step_r)
		yr.append(p_rate_r)

		print("POLICY - passrate: %.2f, average step: %.2f, budget: %d, alpha: %.2f" % (p_rate, avg_step, budget, alpha))
		print("RANDOMPOLICY - passrate: %.2f, average step: %.2f, budget: %d, alpha: %.2f" % (p_rate_r, avg_step_r, budget, alpha))
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(x, y , marker = '*', color = 'g', markersize = 10)
	plt.plot(xr, yr , marker = '*', color = 'c', markersize = 10)
	plt.xlabel("average steps")
	plt.ylabel("pass rate")
	plt.title("Graph for Budget = %d" % budget)
	for i in range(len(alphas)):
	    ax.annotate('%.2f ' % alphas[i], xy=(x[i],y[i]), xytext=(-2,2), fontsize = "7", textcoords='offset points')
	try:
		plt.savefig('../plots/' + str(today) + '/policy_graph_budget_%d.png'%budget)
	except:
		import os
		os.mkdir('../plots/'+str(today))
		plt.savefig('../plots/' + str(today) + '/policy_graph_budget_%d.png'%budget)








