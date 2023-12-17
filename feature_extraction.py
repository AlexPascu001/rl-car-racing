import math
import numpy as np
IMAGE_WIDTH = 96
IMAGE_HEIGHT= 96


# Extract the rectangle that holds all the indicators.
def extract_indicators(image):
    h = IMAGE_HEIGHT / 40.0
    indicators = image[0:IMAGE_WIDTH, math.ceil(IMAGE_HEIGHT- 5 * h):IMAGE_HEIGHT]
    return indicators


# Extracts the true speed indicator from the image.
def extract_true_speed(image):
    place = 5

    # These ratios are taken from the car_racing.py env.
    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    left_x = math.floor(place*s)
    right_x = math.ceil((place+1)*s)

    down_y = math.floor(IMAGE_HEIGHT- 5 * h)
    up_y = math.ceil(IMAGE_HEIGHT -h)

    # Extract the true speed bar.
    speed_bar = image[down_y:up_y, left_x:right_x]
    
    # Sum for each line in the speed bar.
    speed_bar = np.flip(np.sum(speed_bar, axis=(1,2)))

    # Hand-crafted magic numbers chosen to match the actual true speed.
    true_speed = (speed_bar[0]*(12/1362)+ np.sum(speed_bar[1:5]) * (20/1893) + speed_bar[5]*(8/606))

    return true_speed
