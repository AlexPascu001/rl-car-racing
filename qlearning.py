import gymnasium as gym
import numpy as np
import feature_extraction
from feature_extraction import raycast
import pickle
import sys

SPEED_DIM  = 10
GYRO_DIM = 10
STEERING_DIM = 10
RAYCAST_FORWARD_DIM = 10
RAYCAST_LR_DIM = 10
NUM_ACTIONS = 5

SPEED_SPACE = np.linspace(0, 100, SPEED_DIM)
GYROSCOPE_SPACE = np.linspace(-15, 15, GYRO_DIM) 
STEERING_SPACE = np.linspace(-0.42, 0.42, STEERING_DIM)
RAYCAST_FORWARD = np.linspace(0, 5400, RAYCAST_FORWARD_DIM)
RAYCAST_LR = np.linspace(0, 500, RAYCAST_LR_DIM)

GAMMA = 0.95
LEARNING_RATE = 0.1
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.99


def digitize_speed(speed):
    return np.digitize(speed, SPEED_SPACE)

def digitize_gyroscope(gyroscope):
    return np.digitize(gyroscope, GYROSCOPE_SPACE)

def digitize_steering(gyroscope):
    return np.digitize(gyroscope, STEERING_SPACE)

def digitize_raycast_forward(raycast):
    return np.digitize(raycast, RAYCAST_FORWARD)

def digitize_raycast_lr(raycast):
    return np.digitize(raycast, RAYCAST_LR)


def extract_feature_vector(observation):
    indicator_bar = feature_extraction.extract_indicators(observation)
    gameplay = feature_extraction.extract_gameplay(observation)

    extracted_speed = feature_extraction.extract_true_speed(indicator_bar)
    extracted_gyroscope = feature_extraction.extract_gyroscope(indicator_bar)
    extracted_steering = feature_extraction.extract_steering(indicator_bar)
    forward, right, left = [
        raycast(gameplay, 0), 
        raycast(gameplay, np.pi/2), 
        raycast(gameplay, -np.pi/2)
        ]
    
    return (digitize_speed(extracted_speed), 
            digitize_gyroscope(extracted_gyroscope), 
            digitize_steering(extracted_steering),
            digitize_raycast_forward(forward),
            digitize_raycast_lr(right),
            digitize_raycast_lr(left)
            )

def get_epsilon(n_episode):
    epsilon = max(EPSILON_START * (EPSILON_DECAY**n_episode), EPSILON_END)
    return epsilon

def sample_q_table(q_table, feature_vector, epsilon):
    r = np.random.rand()

    if r < epsilon:
        return np.random.randint(0, NUM_ACTIONS)
    
    return np.argmax(q_table[feature_vector])

def update_q_table(q_table, last_feature_vector, new_feature_vector, action, reward):
    tau = reward + (GAMMA* np.max(q_table[new_feature_vector])) - q_table[last_feature_vector+(action,)]
    q_table[last_feature_vector+(action,)] += LEARNING_RATE*tau 



def run_episode(max_steps, q_table, epsilon, render=False):
    if render:
        env = gym.make("CarRacing-v2", render_mode="human", max_episode_steps=max_steps, continuous=False)
    else:
        env = gym.make("CarRacing-v2", max_episode_steps=max_steps, continuous=False)

    observation, info = env.reset()
    old_state = extract_feature_vector(observation)
    cumulative_reward = 0

    while True:
        action = sample_q_table(q_table, old_state, epsilon)
        observation, reward, terminated, truncated, info = env.step(action)

        new_state = extract_feature_vector(observation)

        update_q_table(q_table, old_state, new_state, action, reward)

        cumulative_reward+=reward
        old_state = new_state

        if terminated or truncated:
            break
    return cumulative_reward

def save_qtable(q_table):
    with open("qtable.pkl", "wb") as f:
        pickle.dump(q_table, f)

def load_qtable():
    with open('qtable.pkl','rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    if "--train" in sys.argv:
        q_table = np.zeros(shape=(SPEED_DIM+1, GYRO_DIM+1, STEERING_DIM+1, RAYCAST_FORWARD_DIM+1, RAYCAST_LR_DIM+1, RAYCAST_LR_DIM+1, NUM_ACTIONS))
        num_episodes = 400 
        for x in range(num_episodes):
            epsilon = get_epsilon(x)
            reward = run_episode(800, q_table, epsilon)
            print(f"Episode number: {x}, Epsilon: {epsilon}, Reward: {reward}")
            save_qtable(q_table)
    else:
        q_table = load_qtable()
        run_episode(10000, q_table, epsilon=EPSILON_END, render=True)    
