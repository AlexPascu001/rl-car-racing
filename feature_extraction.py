import math
import numpy as np
import cv2

IMAGE_WIDTH = 96
IMAGE_HEIGHT= 96

# Extract the rectangle that holds all the indicators, except the score.
# The score is considered noise.
def extract_indicators(image):
    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    low_y = math.floor(IMAGE_HEIGHT - 5 * h)
    left_x = math.floor(5*s) 

    indicators = image[low_y:IMAGE_HEIGHT, left_x:IMAGE_WIDTH]
    return indicators

# Extracts the true speed indicator value from the image.
def extract_true_speed(image):
    # Define the range for the white color (true speed indicator)
    lower_white = np.array([60, 60, 60])
    upper_white= np.array([255, 255, 255])

    # Get the image mask.
    mask = cv2.inRange(image, lower_white, upper_white)
    isolated_white = cv2.bitwise_and(image, image, mask=mask)

    # The sum of the pixels when the speed is 100 is 9420.
    return np.sum(isolated_white) * 100 / 9420

def extract_steering(image):
    place = 20

    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    left_x = math.floor((place - 4.5) * s)
    right_x = math.floor((place + 4.5) * s)
    center_x = math.floor(place * s)
    line_y = math.floor((2 * IMAGE_HEIGHT - 5 * h) / 2) - 1

    # image = cv.line(image, (left_x, -1000), (left_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (right_x, -1000), (right_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (center_x, -1000), (center_x, 1000), (0, 0, 255), 1)
    # image = cv.line(image, (-1000, line_y), (1000, line_y), (0, 0, 255), 1)
    # cv.imwrite(f"auxiliaryImages/{extract_steering.cnt}.jpg", image)

    steering_val_left = np.sum(image[line_y, left_x - 1:center_x, 1])
    steering_val_right = np.sum(image[line_y, center_x:right_x + 1, 1])

    steering_val_maximum = max(center_x - left_x, right_x - center_x) * 255

    steering_val_left_normalized = steering_val_left * 0.42 / steering_val_maximum
    steering_val_right_normalized = steering_val_right * 0.42 / steering_val_maximum

    return -steering_val_right_normalized if steering_val_right > steering_val_left else steering_val_left_normalized

# Extract the value of the abs sensors from the image.
def extract_abs(image):
    place_left = 2 
    place_right = 6 

    s = IMAGE_WIDTH / 40.0

    # Compute the whole abs bounding box.
    left_x = int(place_left * s)
    right_x = int(place_right * s)

    # Extract the whole abs sensor bar.
    abs_bar = image[:,  left_x:right_x, 2]

    # Extract individual bars.
    bar1 = abs_bar[:, 1]
    bar2 = abs_bar[:, 3]

    bar3 = abs_bar[:, 7]
    bar4 = abs_bar[:, 9]

    # These two values were computed through trial-and-error
    # The maximum possible value on a bar.
    bar_max = 1913 
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

    # found by trial and error
    MAX_GYROSCOPE_WIDTH = 22 
    MAX_GYROSCOPE_VAL = 15 

    # Assuming the largest contour is the gyroscope indicator, we find its bounding box
    gyroscope_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(gyroscope_contour)
    # x and y are the coordinates of the top-left corner of the bounding box
    # w and h are the width and height of the bounding box

    middle_x = 60 
    if x < middle_x :  # the gyroscope is turning left
        return w * MAX_GYROSCOPE_VAL /MAX_GYROSCOPE_WIDTH
    else:  # the gyroscope is turning right
        return -w *MAX_GYROSCOPE_VAL /MAX_GYROSCOPE_WIDTH
