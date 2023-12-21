import math
import numpy as np
import cv2

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96


# Extract the rectangle that holds all the indicators.
def extract_indicators(image):
    rotated = np.rot90(image, k=1, axes=(0, 1))
    flipped = np.flip(rotated, axis=0)
    image = flipped
    h = IMAGE_HEIGHT / 40.0
    indicators = image[0:IMAGE_WIDTH, math.ceil(IMAGE_HEIGHT - 5 * h):IMAGE_HEIGHT]
    return indicators


# Extracts the true speed indicator from the image.
def extract_true_speed(image):
    place = 5

    # These ratios are taken from the car_racing.py env.
    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    left_x = math.floor(place * s)
    right_x = math.ceil((place + 1) * s)

    down_y = math.floor(IMAGE_HEIGHT - 5 * h)
    up_y = math.ceil(IMAGE_HEIGHT - h)

    # Extract the true speed bar.
    speed_bar = image[down_y:up_y, left_x:right_x]

    # Sum for each line in the speed bar.
    speed_bar = np.flip(np.sum(speed_bar, axis=(1, 2)))

    # Hand-crafted magic numbers chosen to match the actual true speed.
    true_speed = (speed_bar[0] * (12 / 1362) + np.sum(speed_bar[1:5]) * (20 / 1893) + speed_bar[5] * (8 / 606))

    return true_speed


# Extract the value of the abs sensors from the image.
def extract_abs(image):
    place_left = 7
    place_right = 11

    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    # Compute the whole abs bounding box.
    left_x = math.floor(place_left * s)
    right_x = math.ceil(place_right * s)
    down_y = math.floor(IMAGE_HEIGHT - 5 * h)
    up_y = math.ceil(IMAGE_HEIGHT - h)

    # Extract the whole abs sensor bar.
    abs_bar = image[down_y:up_y, left_x:right_x, 2]


    # Extract individual bars.
    bar1 = abs_bar[:, 1]
    bar2 = abs_bar[:, 3]

    bar3 = abs_bar[:, 7]
    bar4 = abs_bar[:, 9]

    # These two values were computed through trial-and-error
    # The maximum possible value on a bar.
    bar_max = 2283
    # The maximum possible angular velocity of a wheel
    wheel_max = 305

    sensor1 = np.sum(bar1) * (wheel_max / bar_max)
    sensor2 = np.sum(bar2) * (wheel_max / bar_max)
    sensor3 = np.sum(bar3) * (wheel_max / bar_max)
    sensor4 = np.sum(bar4) * (wheel_max / bar_max)

    return (sensor1, sensor2, sensor3, sensor4)


def extract_gyroscope(image):
    """
    Extracts the gyroscope value (length of the red indicator) from a given image.
    """
    image_rgb = image
    # Define the range for the red color (gyroscope indicator)
    lower_red = np.array([200, 0, 0])
    upper_red = np.array([255, 80, 80])

    # Create a mask that isolates the red area
    mask = cv2.inRange(image_rgb, lower_red, upper_red)

    # Apply the mask to get the isolated red area
    isolated_red = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Find the contours of the isolated red area
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return 0 as the length (no gyroscope indicator visible)
    if not contours:
        return 0

    # Assuming the largest contour is the gyroscope indicator, we find its bounding box
    gyroscope_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(gyroscope_contour)
    # x and y are the coordinates of the top-left corner of the bounding box
    # w and h are the width and height of the bounding box

    start_height = 72
    if y < start_height:  # the gyroscope is turning left
        return -h
    else:  # the gyroscope is turning right
        return h
