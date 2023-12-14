import gymnasium as gym
import pygame
import numpy as np

MAX_STEPS = 1000
SCREEN_WIDTH = 400 
SCREEN_HEIGHT = 400 

# Create the pygame screen, used for debugging.
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Create the environment.
env = gym.make("CarRacing-v2", max_episode_steps=MAX_STEPS)
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the observation using pygame.
    # Rotate + flip the observation, see: 
    # https://stackoverflow.com/questions/66241275/pygame-rotates-camera-stream-from-opencv-camera.
    rotated = np.rot90(observation, k=1, axes=(0,1))
    flipped = np.flip(rotated, axis=0)
    surface = pygame.surfarray.make_surface(flipped)

    # Scale the surface to the screen size.
    surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

    screen.blit(surface, (0,0))
    pygame.display.update()

    if terminated or truncated:
        break

env.close()