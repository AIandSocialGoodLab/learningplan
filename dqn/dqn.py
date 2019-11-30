import sys, copy, argparse, tensorflow as tf, numpy as np, collections, random
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
sys.path.append("..")
sys.dont_write_bytecode = True
#from Simulator2.gmm_mdp_simulator import GMMMDPSimulator
from Simulator1.simulator import Simulator1

class Replay_Memory():

  def __init__(self, memory_size=10000):

	  # The memory essentially stores transitions recorder from the agent
	  # taking actions in the environment.

	  # Burn in episodes define the number of episodes that are written into the memory from the 
	  # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
	  # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 

	  # Hint: you might find this useful:
	  # 		collections.deque(maxlen=memory_size)
	  self.memory = collections.deque(maxlen=memory_size)

	  #for transition in transitions:
	  #  self.memory.append(transition)

  def sample(self, batch_size=32):
	  # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
	  # You will feed this to your model to train.
	  if len(self.memory) < batch_size:
		  return None
	  return random.sample(self.memory, batch_size)

  def append(self, transition):
	  self.memory.append(transition)

class Qnetwork():
	def __init__(self, num_kc, num_plevels, actions, budget, learning_rate, goal, simulator, exploration_rate = 0.25, exploration_decay = 1e-4, gamma = 0.9): 
		self.budget = budget
		self.num_kc = num_kc
		self.num_plevels = num_plevels
		self.goal = goal
		self.state_dim = num_kc
		self.actions = actions
		self.action_dim = len(actions)
		
		self.epsilon = exploration_rate
		self.epsilon_decay = exploration_decay
		self.gamma = gamma
		self.simulator = simulator
		self.simulator.create_test_pool(budget, goal, 10000)
		self.lr = learning_rate
		self.epsilon_min = 0.01
		self.model = self.create_model()
		self.memory =Replay_Memory()

	def create_model(self):
		state1 = Input(shape = [self.state_dim])
		state2 = Input(shape = [self.state_dim])
		state3 = Input(shape = [self.state_dim])
		action1 = Input(shape = [self.action_dim])
		action2 = Input(shape = [self.action_dim])
		layer1 = Dense(200, activation = 'relu')(state1)
		layer2 = Dense(200, activation = 'relu')(state2)
		layer3 = Dense(200, activation = 'relu')(state3)
		h1 = Concatenate()([layer1, action1])
		h2 = Concatenate()([layer2, action2])
		bel1 = Dense(200, activation = 'relu')(h1)
		bel2 = Dense(200, activation = 'relu')(h2)
		h3 = Concatenate()([bel1, bel2, layer3])
		final_layer = Dense(600, activation = 'relu')(h3)
		output_layer= Dense(self.action_dim, activation='linear')(final_layer)
		model = Model(inputs=[state1, state2, state3, action1, action2], outputs=output_layer)
		model.compile(loss="mse", optimizer=Adam(lr=self.lr))
		return model

	def reach_goal(self, state):
		for i in range(self.state_dim):
			if state[i] < self.goal[i]:
				return False
		return True

	def replay(self):
		# sample from the memory and fit the model
		transitions = self.memory.sample()
		states1, states2, states3, actions1, actions2, targets_f = [], [], [], [], [], []
		if transitions == None:
			return
		for state1, state2, state3, action1, action2, action3, reward, next_state, done in transitions:

			if done:
				target = reward
			else:
				target = (reward + self.gamma * 
						  np.amax(self.model.predict([state2, state3, next_state, action2, action3])[0]))
  
			target_f = self.model.predict([state1, state2, state3, action1, action2])

			ind = list(action3[0]).index(1)
			#print ind 
			target_f[0][ind] = target
		  
			states1.append(state1[0])
			states2.append(state2[0])
			states3.append(state3[0])
			actions1.append(action1[0])
			actions2.append(action2[0])
			targets_f.append(target_f[0])

		self.model.fit([np.array(states1), np.array(states2), np.array(states3), np.array(actions1), np.array(actions2)], 
			np.array(targets_f), epochs=1, verbose=0)


	def move(self, state1, state2, state3, action1, action2, epsilon=True):
		if epsilon and np.random.rand() <= self.epsilon:
			action = random.randrange(self.action_dim)
		else:
			action = np.argmax(self.model.predict([state1, state2, state3, action1, action2])[0])
	  
		return action

	def test(self, num_tests):
		sucess = 0
		steps = 0
		self.simulator.reset_test_pool(self.budget)
		for test in range(1, num_tests + 1):
			if test % 1000 == 0:
				print "test:", test           
			t = 0

			action1, action2 = np.zeros(self.action_dim), np.zeros(self.action_dim)
			state1 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state2 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state3 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state1 = np.reshape(state1, (1, self.state_dim))
			state2 = np.reshape(state2, (1, self.state_dim))
			state3 = np.reshape(state3, (1, self.state_dim))
			action1 = np.reshape(action1, (1, self.action_dim))
			action2 = np.reshape(action2, (1, self.action_dim))
			done = False
			while not done:
				#print t
				if t == self.budget:
					next_action = -1
				else:
					next_action = self.move(state1, state2, state3, action1, action2, epsilon=False)
				action_name = self.actions[next_action]
				observation = self.simulator.test_make_action(action_name, test - 1)
				next_state = copy.deepcopy(state3)
				if action_name == 'Final Exam':
					next_state[0] = observation
				else:
					for i in observation:
						next_state[0][int(i)] = observation[int(i)]
				
				if action_name == 'Final Exam' and self.reach_goal(next_state[0]):
					sucess += 1
					steps += t
					done = True
				elif action_name == 'Final Exam' and not self.reach_goal(next_state[0]):
					steps += t
					done = True
				t += 1

				action1 = copy.deepcopy(action2)
				a =  np.zeros(self.action_dim)
				a[next_action] = 1
				next_action = np.reshape(a, (1, self.action_dim))			
				action2 = copy.deepcopy(next_action)
				state1 = copy.deepcopy(state2)
				state2 = copy.deepcopy(state3)
				state3 = copy.deepcopy(next_state)
		return float(sucess)/num_tests, float(steps)/num_tests
	def test_finalpolicy(self):
		sucess = 0
		steps = []
		self.simulator.reset_test_pool(100)
		for test in range(1, 10000 + 1):
			if test % 1000 == 0:
				print "test:", test           
			t = 0

			action1, action2 = np.zeros(self.action_dim), np.zeros(self.action_dim)
			state1 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state2 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state3 = copy.deepcopy(self.simulator.students[test - 1].cur_state)			
			state1 = np.reshape(state1, (1, self.state_dim))
			state2 = np.reshape(state2, (1, self.state_dim))
			state3 = np.reshape(state3, (1, self.state_dim))
			action1 = np.reshape(action1, (1, self.action_dim))
			action2 = np.reshape(action2, (1, self.action_dim))
			done = False
			while not done:
				#print t
				t += 1
				if t >= 100:
					next_action = -1
				else:
					next_action = self.move(state1, state2, state3, action1, action2, epsilon=False)
				action_name = self.actions[next_action]
				observation = self.simulator.test_make_action(action_name, test - 1)
				next_state = copy.deepcopy(state3)
				if action_name == 'Final Exam':
					next_state[0] = observation
				else:
					for i in observation:
						next_state[0][int(i)] = observation[int(i)]
				
				if action_name == 'Final Exam' and self.reach_goal(next_state[0]):
					sucess += 1
					steps.append(t)
					done = True
				elif action_name == 'Final Exam' and not self.reach_goal(next_state[0]):
					steps.append(t + 1)
					done = True
				

				action1 = copy.deepcopy(action2)
				a =  np.zeros(self.action_dim)
				a[next_action] = 1
				next_action = np.reshape(a, (1, self.action_dim))			
				action2 = copy.deepcopy(next_action)
				state1 = copy.deepcopy(state2)
				state2 = copy.deepcopy(state3)
				state3 = copy.deepcopy(next_state)

		#print steps
		res = dict()
		for val in set(steps):
			res[val] = steps.count(val)
		return res

	def train(self, num_episodes):
		reward_sum = 0
		file = open("record.txt", "wr")
		for e in range(1, num_episodes + 1):
			self.simulator.create_student(self.budget, self.goal)
			action1, action2 = np.zeros(self.action_dim), np.zeros(self.action_dim)
			state1 = copy.deepcopy(self.simulator.student.cur_state)			
			state2 = copy.deepcopy(self.simulator.student.cur_state)			
			state3 = copy.deepcopy(self.simulator.student.cur_state)			
			state1 = np.reshape(state1, (1, self.state_dim))
			state2 = np.reshape(state2, (1, self.state_dim))
			state3 = np.reshape(state3, (1, self.state_dim))
			action1 = np.reshape(action1, (1, self.action_dim))
			action2 = np.reshape(action2, (1, self.action_dim))
			total_reward = 0
			td_errors = []
			step = 0
			done = False
			t = 0
			while not done:
				#print t
				if t == self.budget:
					next_action = -1
				else:
					next_action = self.move(state1, state2, state3, action1, action2)
				action_name = self.actions[next_action]
				observation = self.simulator.make_action(action_name)
				next_state = copy.deepcopy(state3)
				if action_name == 'Final Exam':
					next_state[0] = observation
				else:
					for i in observation:
						next_state[0][int(i)] = observation[int(i)]
				
				if action_name == 'Final Exam' and self.reach_goal(next_state[0]):
					reward = 100
					done = True
				elif action_name != 'Final Exam':
					reward = -1
				else:
					reward = -100
					done = True

				total_reward += reward

				V_state = np.amax(self.model.predict([state1, state2, state3, action1, action2])[0])
				
				a =  np.zeros(self.action_dim)
				a[next_action] = 1
				next_action = np.reshape(a, (1, self.action_dim))

				V_next_state = np.amax(self.model.predict([state2, state3, next_state, action2, next_action])[0])
				td_error = abs(reward + self.gamma * V_next_state - V_state)
				td_errors.append(td_error)
							  
				self.memory.append((state1, state2, state3, action1, action2, next_action, reward, next_state, done))
				self.replay()

				action1 = copy.deepcopy(action2)
				
				action2 = copy.deepcopy(next_action)
				state1 = copy.deepcopy(state2)
				state2 = copy.deepcopy(state3)
				state3 = copy.deepcopy(next_state)
				
				t += 1

			avg_td_errors = np.mean(td_errors)
			reward_sum += total_reward
			if self.epsilon > self.epsilon_min:
				self.epsilon -= self.epsilon_decay
			if e % 1000 == 0:
				print "start test..."

				success_rate, avg_steps = self.test(1000)
				print("TESTING: episode: {}/{}, reward: {}, avg_reward: {}, succss_rate: {}, avg_steps: {}".format(e, num_episodes, total_reward, reward_sum/100.0, success_rate, avg_steps))
				file.write("TESTING: episode: " +str(e) + str(num_episodes)+ ", avg_reward: "+str((reward_sum/100.0))+ ", success_rate: "+str(success_rate) +", avg_steps: "+str(avg_steps))
				file.write("\n")
				reward_sum = 0
				print("=" * 20)

def main():
	goal = [2,2,2,2,2]
	actions = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9', 'AT11', 'AT12', 'AT13', 'AT14', 'AT15', 'AT16', 'AT17', 'AT18', 'AT19', 'AT20', 'Final Exam']
	simulator = Simulator1()
	qnetwork = Qnetwork(num_kc = 5, num_plevels = 3, actions = actions, budget = 12, learning_rate = 2e-5, goal = goal, simulator = simulator)
	qnetwork.train(12000)
	print(qnetwork.test_finalpolicy())

main()





