import gymnasium as gym
import numpy as np
import car_racing
import copy
import pickle

MAX_STEPS = 1000
TRACK_SEED = 123
GAMMA = 1 
NUM_ACTIONS = 5

class Node:
    def __init__(self, parent, terminal):
        self.parent = parent 
        self.terminal = terminal

        self.children = {action:None for action in range(NUM_ACTIONS)} 
        self.visit_count = 0 
        self.total_reward = 0

    def get_value(self):
        return self.total_reward/self.visit_count
    
    def backpropagate(self, reward):
        here = self 
        while here != None:
            here.visit_count+=1
            here.total_reward += reward
            reward = reward * GAMMA
            here = here.parent
    def height(self):
        height = 1 
        for child in self.children.values():
            if child is not None:
                height = max(height, 1+child.height())

        return height


# Goes down the tree until we reach an unexpanded node.
# Expand it and backpropagate the reward.
def expand_new(root):
    if root.terminal:
        return 
    
    # Try to find an unexpanded node at this level.
    for action in range(NUM_ACTIONS):
        if root.children[action] == None:
            _, reward, terminated, truncated, _ = env.step(action)
            expanded = Node(root, terminated or truncated)
            root.children[action] = expanded
            expanded.backpropagate(reward)
            return

    # No unexpanded node, move to the next level.
    children = list(root.children.values())
    if np.random.rand() < 0.2:
        random_action = np.random.randint(0, NUM_ACTIONS)
        env.step(random_action)
        expand_new(children[random_action])
        return

    best_index, best_child = max(enumerate(children), key=lambda child: child[1].get_value())
    env.step(best_index)
    expand_new(best_child)

# Replays the history
def reset_env(history):
    env.reset()
    for action in history:
        env.step(action)

def monte_carlo(history):
    global root, env

    for x in range(100):
        expand_new(root)
        reset_env(history)


    child_values = np.array([child.get_value() for child in root.children.values()])
    best_action = np.argmax(child_values) 
    print(root.height())
    return best_action


root = Node(None, False)
history = []
np.random.seed(TRACK_SEED)
env = car_racing.CarRacing(continuous=False)
observation, info = env.reset()

while True:
    action = monte_carlo(history)
    observation, reward, terminated, truncated, info = env.step(action)
    history.append(action)
    root = root.children[action]
    print(history, reward)

    if terminated or truncated:
        break

env.close()
