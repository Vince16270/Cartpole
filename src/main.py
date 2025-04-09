import gymnasium as gym
from agent import QLearningAgent

def main():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Initialiseer de Q-learning agent met standaard hyperparameters
    agent = QLearningAgent(env)
    
    num_episodes = 1000
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_disc = agent.discretize(state)
        total_reward = 0
        done = False
        
        while not done:
            # Kies een actie via de epsilon-greedy strategie
            action = agent.choose_action(state_disc)
            # Voer de gekozen actie uit in de omgeving
            next_state, reward, done, truncated, info = env.step(action)
            next_state_disc = agent.discretize(next_state)
            
            # Werk de Q-table bij met de ontvangen feedback
            agent.update(state_disc, action, reward, next_state_disc, done)
            
            state_disc = next_state_disc
            total_reward += reward
            
            # Indien de episode afgelopen is, breek de loop
            if done:
                break
                
        # Pas epsilon decay toe
        agent.decay_epsilon()
        
        print(f"Episode {episode+1}: Totale reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    main()