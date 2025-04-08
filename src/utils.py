import numpy as np
import matplotlib.pyplot as plt

def create_bins(low, high, num_bins):
    """
    Creëert discretisatie-bins voor elke dimensie van de observatieruimte.

    Parameters:
      - low: array met lage waarden per dimensie
      - high: array met hoge waarden per dimensie
      - num_bins: aantal bins per dimensie

    Returns:
      - Een lijst met numpy-arrays die de snijpunten (bin grenzen) bevatten voor iedere dimensie.
    """
    bins = []
    for l, h in zip(low, high):
        bins.append(np.linspace(l, h, num_bins + 1)[1:-1])
    return bins

def discretize_state(observation, bins):
    """
    Converteert een continue observatie naar een discrete staat met behulp van de opgegeven bins.

    Parameters:
      - observation: de continue observatie (numpy array of lijst)
      - bins: lijst van numpy-arrays met discretisatiegrenzen per dimensie

    Returns:
      - Een tuple met de discrete indices voor elke dimensie.
    """
    discretized = []
    for i, val in enumerate(observation):
        discretized.append(np.digitize(val, bins[i]))
    return tuple(discretized)

def plot_rewards(reward_list, title='Episode Rewards', filename='rewards.png'):
    """
    Plot de totale rewards per episode en slaat de grafiek op als een afbeelding.

    Parameters:
      - reward_list: lijst met totale rewards per episode
      - title: titel van de plot
      - filename: bestandsnaam waarin de plot wordt opgeslagen
    """
    plt.figure(figsize=(10, 6))
    plt.plot(reward_list, marker='o', linestyle='-', markersize=3)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()