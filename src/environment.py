import gymnasium as gym

class CartPoleEnvironment:
    def __init__(self, env_name='CartPole-v1'):
        """
        Initialiseert de CartPole-omgeving met render_mode zodat video-opnames mogelijk zijn.
        
        Parameters:
          - env_name: Naam van de omgeving (standaard 'CartPole-v1')
        """
        self.env = gym.make(env_name, render_mode='rgb_array')
        self._adjust_environment()

    def _adjust_environment(self):
        """
        Past de grenzen van de continue observatie-ruimte aan.
        In de CartPole-omgeving zijn sommige grenzen oneindig; 
        deze worden vervangen door redelijke limieten.
        """
        low = self.env.observation_space.low.copy()
        high = self.env.observation_space.high.copy()

        # Voor CartPole: vervang oneindige grenzen voor cart velocity en pole velocity
        low[1] = -3.0
        high[1] = 3.0
        low[3] = -3.0
        high[3] = 3.0

        # Update de grenzen in de environment
        self.env.observation_space.low = low
        self.env.observation_space.high = high

    def reset(self):
        """
        Reset de omgeving naar de beginstaat.
        
        Returns:
          - state: de initiële observatie
          - info: bijkomende informatie (dictionary)
        """
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        """
        Voert een actie uit in de omgeving.
        
        Parameters:
          - action: de uit te voeren actie
        
        Returns:
          - next_state: observatie na het uitvoeren van de actie
          - reward: beloning die is ontvangen
          - done: boolean die aangeeft of de episode is beëindigd
          - truncated: boolean die aangeeft of de episode voortijdig is afgebroken
          - info: bijkomende informatie (dictionary)
        """
        return self.env.step(action)

    def render(self):
        """Toont de huidige toestand van de omgeving."""
        self.env.render()

    def close(self):
        """Sluit de omgeving netjes af."""
        self.env.close()