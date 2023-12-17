import gymnasium as gym
import pygame
import numpy as np
import feature_extraction

MAX_STEPS = 100000000
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

    # Render the observation using pygame.
    # Rotate + flip the observation, see: 
    # https://stackoverflow.com/questions/66241275/pygame-rotates-camera-stream-from-opencv-camera.
    rotated = np.rot90(observation, k=1, axes=(0,1))
    flipped = np.flip(rotated, axis=0)
    extracted_speed = feature_extraction.extract_true_speed(observation)
    extracted_abs = feature_extraction.extract_abs(observation)

    true_speed = np.sqrt(
            np.square(env.car.hull.linearVelocity[0])
            + np.square(env.car.hull.linearVelocity[1])
        )
    true_abs = tuple(env.car.wheels[i].omega for i in range(4))
    print(f"Extracted speed: {extracted_speed}, Actual: {true_speed}")
    print(f"Extracted abs: {extracted_abs}, Actual: {true_abs}")

    surface = pygame.surfarray.make_surface(flipped)

    # Scale the surface to the screen size.
    surface = pygame.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

    screen.blit(surface, (0,0))
    pygame.display.update()

    if terminated or truncated:
        break

env.close()