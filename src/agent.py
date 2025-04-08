import numpy as np

class QLearningAgent:
    def __init__(self, env, num_bins=6, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialiseer de Q-learning agent.
        
        Parameters:
          - env: de Gymnasium-omgeving
          - num_bins: aantal bins per dimensie voor de discretisatie van de continue state space
          - alpha: learning rate
          - gamma: discount factor
          - epsilon: initiële waarde voor de epsilon-greedy strategie
          - epsilon_min: minimumwaarde voor epsilon
          - epsilon_decay: factor waarmee epsilon per episode afneemt
        """
        self.env = env
        self.num_bins = num_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Creëer de bins om de continue observatieruimte te discretiseren
        self.bins = self._create_bins()

        # Initialiseer de Q-table met de juiste dimensies:
        # (num_bins per state-dimensie) + (aantal acties)
        self.q_table = np.zeros(self._get_q_table_shape())

    def _create_bins(self):
        """
        Creëer discretisatie-bins voor elke dimensie van de observatie.
        Voor CartPole kunnen sommige grenzen oneindig zijn; vervang deze met redelijke limieten.
        """
        bins = []
        low = self.env.observation_space.low.copy()
        high = self.env.observation_space.high.copy()
        # Voor CartPole: pas grenzen aan voor cart velocity en pole velocity
        low[1] = -3.0
        high[1] = 3.0
        low[3] = -3.0
        high[3] = 3.0

        for i in range(len(low)):
            # Creëer bins met numpy.linspace en sluit de uiterste grenzen uit
            bins.append(np.linspace(low[i], high[i], self.num_bins + 1)[1:-1])
        return bins

    def _get_q_table_shape(self):
        """
        Bepaal de shape van de Q-table op basis van het aantal bins per state-dimensie
        en het aantal acties in de omgeving.
        """
        dims = [self.num_bins] * self.env.observation_space.shape[0] + [self.env.action_space.n]
        return tuple(dims)
    
    def discretize(self, observation):
        """
        Converteer een continue observatie naar een discrete staat.
        
        Parameters:
          - observation: de continue observatie (numpy array)
        
        Returns:
          - Een tuple met discrete indices voor elke dimensie
        """
        discretized = []
        for i, val in enumerate(observation):
            discretized.append(np.digitize(val, self.bins[i]))
        return tuple(discretized)

    def choose_action(self, state):
        """
        Kies een actie volgens de epsilon-greedy strategie.
        
        Parameters:
          - state: de huidige discrete staat
        
        Returns:
          - Een actie (int) op basis van exploratie of exploitatie
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Exploratie: kies een willekeurige actie
        else:
            return np.argmax(self.q_table[state])  # Exploitatie: kies de beste bekende actie
    
    def update(self, state, action, reward, next_state, done):
        """
        Voer de Q-learning update uit voor een gegeven (state, action)-paar.
        
        Parameters:
          - state: de huidige discrete staat
          - action: de gekozen actie (int)
          - reward: de ontvangen beloning
          - next_state: de volgende discrete staat
          - done: boolean die aangeeft of de episode is beëindigd
        """
        if done:
            td_target = reward
        else:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state + (best_next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        """Verlaag epsilon zodat de agent in latere episodes meer gaat exploiteren."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay