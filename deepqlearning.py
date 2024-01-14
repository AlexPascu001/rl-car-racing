from collections import namedtuple, deque
import random
import torch
from torch import nn, cuda 
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import feature_extraction
from feature_extraction import raycast

BATCH_SIZE = 128
GAMMA = 0.99

EPSILON_START = 1 
EPSILON_END = 0.01
EPSILON_DECAY = 0.99

TAU = 0.1
LEARNING_RATE = 0.1

# Speed, Angle to COM, Ray forward, Ray right, Ray left.
NUM_OBSERVATIONS = 5
NUM_ACTIONS = 5 

def get_epsilon(n_episode):
    epsilon = max(EPSILON_START * (EPSILON_DECAY**n_episode), EPSILON_END)
    return epsilon


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(NUM_OBSERVATIONS, NUM_ACTIONS)
target_net = DQN(NUM_OBSERVATIONS, NUM_ACTIONS)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(10000)

# Array of states -> array of actions
def select_action(state, epsilon):
    r = np.random.rand()

    if r < epsilon:
        return torch.tensor([np.random.randint(0, NUM_ACTIONS)]).unsqueeze(0)

    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)

    start_states = [transition.state for transition in transitions]
    actions = [transition.action for transition in transitions]
    end_states = [transition.next_state for transition in transitions]
    rewards = [transition.reward for transition in transitions]

    start_batch = torch.cat(start_states)
    action_batch = torch.cat(actions)
    end_batch = torch.cat(end_states)
    reward_batch = torch.cat(rewards)

    #   left                      q
    #    v                        v
    # Q(st, a) = r + GAMMA * max_a(Q(se, a))

    # Compute Q(st, a) for the start batch.
    left = policy_net(start_batch).gather(1, action_batch)

    with torch.no_grad():
        # Compute max_a(Q(se, a)) for the end batch.
        q = target_net(end_batch).max(1).values 

    # Compute r + GAMMA * max_a(Q(se, a))
    right = reward_batch + GAMMA * q 

    criterion = nn.SmoothL1Loss()
    loss = criterion(left, right.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def extract_feature_tensor(observation):
    indicator_bar = feature_extraction.extract_indicators(observation)
    gameplay = feature_extraction.extract_gameplay(observation)

    extracted_speed = feature_extraction.extract_true_speed(indicator_bar)/100
    extracted_gyroscope = feature_extraction.extract_gyroscope(indicator_bar)
    extracted_steering = feature_extraction.extract_steering(indicator_bar)
    extracted_com = feature_extraction.extract_angle_to_street_com(gameplay)
    forward, right, left = [
        raycast(gameplay, 0)/5329, 
        raycast(gameplay, np.pi/2)/500, 
        raycast(gameplay, -np.pi/2)/500
        ]
    return torch.tensor([extracted_speed, forward, right, left, extracted_com], dtype=torch.float32).unsqueeze(0)


def soft_update():
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)


def run_episode(max_steps, epsilon, render=False):
    if render:
        env = gym.make("CarRacing-v2", render_mode="human", max_episode_steps=max_steps, continuous=False)
    else:
        env = gym.make("CarRacing-v2", max_episode_steps=max_steps, continuous=False)

    observation, info = env.reset()
    old_state = extract_feature_tensor(observation)
    cumulative_reward = 0

    while True:
        action = select_action(old_state, epsilon)

        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward])
        new_state = extract_feature_tensor(observation)

        if terminated or truncated:
            break

        memory.push(Transition(old_state, action, new_state, reward))
        old_state = new_state

        optimize_model()
        soft_update() 
        cumulative_reward+=reward

    return cumulative_reward.item()

if __name__ == "__main__":
    for x in range(1000):
        epsilon = get_epsilon(x)
        reward = run_episode(800, epsilon)
        print(f"Episode: {x}, Epsilon: {epsilon}, Reward: {reward}")