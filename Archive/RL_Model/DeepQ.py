from enums import *
import random, csv, copy
import tensorflow as tf
import numpy as np
from simulate_action import transit
NUM_KC = 5
NUM_PLEVEL = 5
NUM_ACTIONS = 40
ACTIONS = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20',
 '21', '22', '23', '24', '3', '4', '5', '6', '7', '8', '9', 'AT25', 'AT26', 'AT27', 'AT28', 'AT29',
  'AT30', 'AT31', 'AT32', 'AT33', 'AT34', 'AT35', 'AT36', 'AT37', 'AT38', 'AT39']




class DeepLearner:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.learning_rate = learning_rate
        self.discount = discount 
        self.exploration_rate = 1.0 
        self.exploration_delta = 1.0 / iterations 
        self.input_count = NUM_KC**NUM_PLEVEL
        self.output_count = NUM_ACTIONS
        self.session = tf.Session()
        self.define_model()
        self.session.run(self.initializer)

    # Define tensorflow model graph
    def define_model(self):
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_count])
        fc1 = tf.layers.dense(self.model_input, 16000, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((self.input_count, 16000))))
        fc2 = tf.layers.dense(fc1, 16000, activation=tf.sigmoid, kernel_initializer=tf.constant_initializer(np.zeros((16000, self.output_count))))
        self.model_output = tf.layers.dense(fc2, self.output_count)
        self.target_output = tf.placeholder(shape=[None, self.output_count], dtype=tf.float32)
        loss = tf.losses.mean_squared_error(self.target_output, self.model_output)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        self.initializer = tf.global_variables_initializer()

    def get_Q(self, state):
        return self.session.run(self.model_output, feed_dict={self.model_input: self.to_one_hot(state)})[0]

    def to_one_hot(self, state):
        one_hot = np.zeros((1, NUM_KC**NUM_PLEVEL))
        one_hot[0, [state]] = 1
        return one_hot
"""
    def get_next_action(self, state):
        if random.random() > self.exploration_rate: 
            return self.greedy_action(state)
        else:
            return self.random_action()

    
    def greedy_action(self, state):
        return np.argmax(self.get_Q(state))

    def random_action(self):
        return
"""
    def train(self, old_state, action, reward, new_state):
        old_state_Q_values = self.get_Q(old_state)
        new_state_Q_values = self.get_Q(new_state)
        old_state_Q_values[action] = reward + self.discount * np.amax(new_state_Q_values)
        training_input = self.to_one_hot(old_state)
        target_output = [old_state_Q_values]
        training_data = {self.model_input: training_input, self.target_output: target_output}
        self.session.run(self.optimizer, feed_dict=training_data)

    def update(self, old_state, new_state, action, reward):
        self.train(old_state, action, reward, new_state)
        if self.exploration_rate > 0:
            self.exploration_rate -= self.exploration_delta







