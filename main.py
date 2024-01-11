import gymnasium as gym
import pygame
import numpy as np
import feature_extraction
import util

MAX_STEPS = 200000
SCREEN_WIDTH = 400 
SCREEN_HEIGHT = 400 

# User-controlled input.
a = np.array([0.0, 0.0, 0.0])

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
    extracted_speed = feature_extraction.extract_true_speed(indicator_bar)
    extracted_abs = feature_extraction.extract_abs(indicator_bar)
    extracted_gyroscope = feature_extraction.extract_gyroscope(indicator_bar)
    extracted_steering = feature_extraction.extract_steering(observation)

    true_speed = np.sqrt(
        np.square(env.car.hull.linearVelocity[0])
        + np.square(env.car.hull.linearVelocity[1])
    )
    true_abs = tuple(env.car.wheels[i].omega for i in range(4))
    true_gyroscope = env.car.hull.angularVelocity
    true_steering = env.car.wheels[0].joint.angle

    print(f"True speed: {true_speed}, extracted speed: {extracted_speed}")
    print(f"True abs:{true_abs}, extracted abs: {extracted_abs}")
    print(f"True gyroscope:{true_gyroscope}, extracted gyroscope: {extracted_gyroscope}")

    # Rendering
    surface = pygame.surfarray.make_surface(util.flip_and_rotate(observation))
    screen.blit(surface, (0,0))
    pygame.display.update()

    if terminated or truncated:
        break

env.close()