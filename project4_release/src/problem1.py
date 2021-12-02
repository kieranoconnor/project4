# -------------------------------------------------------------------------
'''
    Problem 1: Implement Q-learning using a table.
'''
import gym
import random
import pandas as pd
import numpy as np

class Agent_QTable(object):
    '''' Agent that learns via tabular Q-learning. '''
    # --------------------------
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.9):
        """
        :param env: Environment defined by Gym
        :param alpha: learning rate for updating the Q-table
        :param epsilon: the Epsilon in epsilon-greedy policy
        :param gamma: discounting factor
        """
        np.random.seed(21)
        random.seed(21)

        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # the Q-table: key=state, value=[q_of_s_action_0, q_of_s_action_1]
        self.Q_table = dict()

        # Following variables for statistics
        self.training_trials = 0
        self.testing_trials = 0

        # The performance of q-leaning is very sensitive to
        # the following ranges when discretizing continuous state variables.
        self.cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        self.pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        self.cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        self.angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]

    #--------------------------
    def encode_state(self, observed_state):
        '''
        First, discretize each of the four dimensions of observed_state to an integer bin number.
        For example, if cart_position = observed_state[0] = 0.1, then the first integer in the return value should be 5,
        since cart_position_bins = [-1.92 -1.44 -0.96 -0.48  0.    0.48  0.96  1.44  1.92] and 0.1 is in the 6-th bin:
        0th bin = (-inf, -1.92)
        1st bin = [-1.92,-1.44)
        ...
        10th bin = [1.92,inf)
        Second, concatenate the four integers into a string. E.g if [1,2,3,4] are the bin indices, then create a string '1234'.
        This string will be a key in the Q-table (a dictionary).

        :param observed_state: [horizontal_cart_position, cart_velocity, angle_of_pole, angular_velocity],
                                each dimension is a floating number
        :return an integer (e.g., 1234) representing the observed state.
         [cart_position, pole_angle, cart_velocity, angle_rate]
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #########################################
        output = ""
        for count, i in enumerate(self.cart_position_bins):
            if observed_state[0]<i:
                output+=str(count)
                break
        for count, i in enumerate(self.pole_angle_bins):
            if observed_state[1]<i:
                output+=str(count)
                break
        for count, i in enumerate(self.cart_velocity_bins):
            if observed_state[2]<i:
                output+=str(count)
                break
        for count, i in enumerate(self.angle_rate_bins):
            if observed_state[3]<i:
                output+=str(count)
                break        
            

        return int(output)

    #--------------------------
    def epsilon_greedy(self, state):
        '''
        Given a state (represented by a state string), choose an action (0 or 1) using epsilon-greedy policy
                with the current Q function.
            If state is not in the Q-table, register the state with the Q-table and randomly pick an action.
        :param state: state string
        :return action that agent will take. Should be either 0 or 1
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        p = random.random()
        if state not in self.Q_table.keys():
            self.Q_table[state] = [0,0]
        if p< self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])
        #########################################
        action = 0
        r = random.random()
        if state not in self.Q_table.keys() or r < self.epsilon:
            action = self.env.action_space.sample()
            
            if state not in self.Q_table.keys():
                self.Q_table[state] = np.zeros((2))
            return action
        else:
            return np.argmax(self.Q_table[state])


    #--------------------------
    def learn(self, prev_state, prev_action, prev_reward, next_state):
        '''
        Update Q[prev_state, prev_action] using the Q-learning formula.

        :param prev_state: previous state
        :param prev_action: action taken at the previous state.
        :param prev_reward: Reward at previous time step.
        :param next_state: next state.
        :return return the updated q entry
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #########################################
        prev_Q = self.Q_table[prev_state]
        if next_state in self.Q_table.keys():
            max_Q = np.argmax(self.Q_table[next_state])
        else:
            max_Q = self.env.action_space.sample()
        max_Q = np.argmax(prev_Q)
        up_Q = prev_Q[prev_action] + self.alpha * ((prev_reward + self.gamma * max_Q) - prev_Q[prev_action])
        self.Q_table[prev_state][prev_action] = up_Q
        print(up_Q)
        return up_Q