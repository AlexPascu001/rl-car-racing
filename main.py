import gymnasium as gym
import pygame
import numpy as np
import feature_extraction
import cv2 as cv
import util
import time

MAX_STEPS = 100000000
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

# User-controlled input.
a = np.array([0.0, 0.0, 0.0])

MAX_GYROSCOPE_VAL = 22  # found by trial and error

def register_input():
    global a
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                a[0] = -1.0
            if event.key == pygame.K_RIGHT:
                a[0] = +1.0
            if event.key == pygame.K_UP:
                a[1] = +1.0
            if event.key == pygame.K_DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            # to close the game
            if event.key == pygame.K_ESCAPE:
                env.close()
                exit()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                a[0] = 0
            if event.key == pygame.K_RIGHT:
                a[0] = 0
            if event.key == pygame.K_UP:
                a[1] = 0
            if event.key == pygame.K_DOWN:
                a[2] = 0


# Create the pygame screen, used for debugging.
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Create the environment.
env = gym.make("CarRacing-v2", max_episode_steps=MAX_STEPS)
obs, info = env.reset()


while True:
    register_input()
    observation, reward, terminated, truncated, info = env.step(a)

    indicator_bar = feature_extraction.extract_indicators(observation)
    true_speed = feature_extraction.extract_true_speed(indicator_bar)
    print(f"True speed: {true_speed}")

    # Rendering
    surface = pygame.surfarray.make_surface(util.flip_and_rotate(observation))
    surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(surface, (0, 0))
    pygame.display.update()

    if terminated or truncated:
        break

    time.sleep(0.01)

env.close()

