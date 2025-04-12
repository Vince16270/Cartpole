import random
import pandas as pd
import numpy as np


class AgentBasic(object):
    ''' Simple agent class. '''
    def __init__(self):
        pass

    def act(self, obs):
        ''' Based on angle, return action. '''
        angle = obs[2]
        return 0 if angle < 0 else 1


class AgentRandom(object):
    ''' Random agent class. '''
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        ''' Agent randomly chooses an action. '''
        return self.action_space.sample()


class QLearningAgent(object):
    ''' Agent that can learn via Q-learning. '''
    def __init__(self, env, alpha=0.1, epsilon=1.0, gamma=0.7):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q_table = dict()
        self._set_seed()

    def _set_seed(self):
        ''' Set random seeds for reproducibility. '''
        np.random.seed(21)
        random.seed(21)

    def build_state(self, features):
        ''' Build state by concatenating features into int. '''
        return int("".join(map(lambda feature: str(int(feature)), features)))

    def create_state(self, obs):
        ''' Create a discrete state from observation. '''
        cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
        pole_angle_bins = pd.cut([-2, 2], bins=10, retbins=True)[1][1:-1]
        cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
        angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
        state = self.build_state([
            np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
            np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
            np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
            np.digitize(x=[obs[3]], bins=angle_rate_bins)[0]
        ])
        return state

    def choose_action(self, state):
        ''' Choose an action using epsilon-greedy. '''
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            max_Q = self.get_maxQ(state)
            actions = [a for a, q in self.Q_table[state].items() if q == max_Q]
            return random.choice(actions) if actions else self.env.action_space.sample()

    def create_Q(self, state, valid_actions):
        ''' Initialize Q values if state not in Q table. '''
        if state not in self.Q_table:
            self.Q_table[state] = {action: 0.0 for action in valid_actions}

    def get_maxQ(self, state):
        ''' Return the max Q value for a state. '''
        return max(self.Q_table[state].values())

    def learn(self, state, action, prev_reward, prev_state, prev_action):
        ''' Q-learning update step. '''
        self.Q_table[prev_state][prev_action] = (
            (1 - self.alpha) * self.Q_table[prev_state][prev_action] +
            self.alpha * (prev_reward + self.gamma * self.get_maxQ(state))
        )

    def decay_epsilon(self, decay=0.995, min_epsilon=0.01):
        ''' Decay epsilon for exploration/exploitation tradeoff. '''
        self.epsilon = max(self.epsilon * decay, min_epsilon)
