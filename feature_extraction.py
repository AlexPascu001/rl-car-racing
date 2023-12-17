import math
import numpy as np
import cv2 as cv

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

def extract_steering(image):
    place = 20

    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    left_x = math.floor((place - 4.2) * s)
    right_x = math.floor((place + 4.2) * s)
    center_x = math.floor(place * s)
    line_y = math.floor((2 * IMAGE_HEIGHT - 5 * h) / 2) - 1

    # image = cv.line(image, (left_x, -1000), (left_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (right_x, -1000), (right_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (center_x, -1000), (center_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (-1000, line_y), (1000, line_y), (0, 0, 255), 1)
    # cv.imwrite(f"auxiliaryImages/{extract_steering.cnt}.jpg", image)

    green_channel = image[:, :, 1]
    green_channel[green_channel > 0] = 255

    extract_steering.cnt += 1

    steering_val_left = np.sum(image[line_y, left_x - 1:center_x + 1, 1])
    steering_val_right = np.sum(image[line_y, center_x - 1:right_x + 1, 1])

    steering_val_maximum = max(center_x - left_x + 1, right_x - center_x + 1) * 255

    steering_val_left_normalized = steering_val_left * 0.42 / steering_val_maximum
    steering_val_right_normalized = steering_val_right * 0.42 / steering_val_maximum

    return -steering_val_right_normalized if steering_val_right > steering_val_left else steering_val_left_normalized

extract_steering.cnt = 1