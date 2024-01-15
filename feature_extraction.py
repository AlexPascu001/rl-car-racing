import math
import numpy as np
import cv2

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96

# Extract the rectangle that holds all the indicators, except the score.
# The score is considered noise.
def extract_indicators(image):
    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    low_y = math.floor(IMAGE_HEIGHT - 5 * h)
    left_x = math.floor(5*s) 

    indicators = image[low_y:IMAGE_HEIGHT, left_x:IMAGE_WIDTH]
    return indicators

# Extracts the rectangle that holds only the gameplay, ignoring the
# indicators
def extract_gameplay(image):
    s = IMAGE_WIDTH / 40.0
    h = IMAGE_HEIGHT / 40.0

    low_y = math.floor(IMAGE_HEIGHT - 5 * h)

    indicators = image[:low_y,: ]
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
    image_rgb = image
    # Define the range for the green color (steering indicator)
    lower_green= np.array([0, 100, 0])
    upper_green = np.array([0, 255, 80])

    # Create a mask that isolates the green area
    mask = cv2.inRange(image_rgb, lower_green, upper_green)

    # Find the contours of the isolated green area
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return 0 as the length (no gyroscope indicator visible)
    if not contours:
        return 0 

    # found by trial and error
    MAX_STEERING_WIDTH = 10 
    MAX_STEERING_VAL = 0.42 

    # Assuming the largest contour is the gyroscope indicator, we find its bounding box
    gyroscope_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(gyroscope_contour)
    # x and y are the coordinates of the top-left corner of the bounding box
    # w and h are the width and height of the bounding box

    middle_x = 34
    if x < middle_x :  # the gyroscope is turning left
        return w *MAX_STEERING_VAL  / MAX_STEERING_WIDTH 
    else:
        return -w *MAX_STEERING_VAL /MAX_STEERING_WIDTH 

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

def raycast(image, angle):
    # grass_color = rgb(102, 230, 102)
    height , width, _ = image.shape

    lower_green = np.array([80, 200, 90])
    upper_green = np.array([110, 250, 120])

    mask = cv2.inRange(image, lower_green, upper_green)

    car_center_y = int(3/4*IMAGE_HEIGHT)
    car_center_x = int(1/2*IMAGE_HEIGHT)

    def inside(x, y):
        return x>=0 and x < width and y>=0 and y<height

    dir_x = np.sin(angle)
    dir_y = -np.cos(angle)

    current_x = car_center_x
    current_y = car_center_y

    while True:
        posx = round(current_x)
        posy = round(current_y)

        if not inside(posx, posy):
            break
        if mask[posy, posx] != 0:
            break
        image[posy][posx] = [255, 0, 0]
        current_y+=dir_y
        current_x+=dir_x

    return (current_x-car_center_x) ** 2 + (current_y - car_center_y)**2

# Utility function used to draw lines at a given angle from the car.
def draw_ray(image, angle):
    height , width, _ = image.shape

    car_center_y = int(3/4*IMAGE_HEIGHT)
    car_center_x = int(1/2*IMAGE_HEIGHT)

    dir_x = np.sin(angle)
    dir_y = -np.cos(angle)

    current_x = car_center_x
    current_y = car_center_y

    def inside(x, y):
        return x>=0 and x < width and y>=0 and y<height

    while True:
        posx = round(current_x)
        posy = round(current_y)

        if not inside(posx, posy):
            break

        image[posy][posx] = [255, 0, 0]
        current_y+=dir_y
        current_x+=dir_x



# Extracts the center of mass of the street.
def extract_street_com(image):
    height , width, _ = image.shape
    lower_gray = np.array([100, 100, 100])
    upper_gray = np.array([110, 110, 110])

    mask = cv2.inRange(image, lower_gray, upper_gray)
    xpos = []
    ypos = []
    for y in range(height):
        for x in range(width):
            if mask[y,x]:
                xpos.append(x)
                ypos.append(y)

    num_pos = len(xpos)
    if num_pos > 0:
        avg_x = int(sum(xpos) / num_pos)
        avg_y = int(sum(ypos) / num_pos)

        # Draw COM for debugging.
        image[avg_y-1:avg_y+2, avg_x-1:avg_x+2] = [0,0,0]
        return (avg_x, avg_y) 
    return None 

# Extracts the angle in radians to the center of mass of the street.
def extract_angle_to_street_com(image):
    street_com = extract_street_com(image)
    if not street_com:
        return 0
    street_com_x, street_com_y = street_com

    car_center_y = int(3/4*IMAGE_HEIGHT)
    car_center_x = int(1/2*IMAGE_HEIGHT)

    diff_x = street_com_x - car_center_x
    diff_y = car_center_y - street_com_y 

    vector = diff_y + 1j*diff_x

    angle = np.angle(vector)

    # Draw ray to COM for debugging.
    draw_ray(image, angle)
    return angle 

# Extracts closest point to a collectible.
def extract_closest_point_to_collectible(image):
    height , width, _ = image.shape
    lower_gray = np.array([100, 100, 100])
    upper_gray = np.array([101, 101, 101])

    mask = cv2.inRange(image, lower_gray, upper_gray)

    car_center_y = int(3/4*IMAGE_HEIGHT)
    car_center_x = int(1/2*IMAGE_HEIGHT)

    min_dist = 100000
    min_x = 0
    min_y = 0
    # the collectible should be "above" the car
    for y in range(car_center_y):
        for x in range(width):
            if mask[y,x]:
                dist = (x-car_center_x)**2 + (y-car_center_y)**2
                if dist < min_dist:
                    min_dist = dist
                    min_x = x
                    min_y = y

    # Draw closest point for debugging.
    image[min_y-1:min_y+2, min_x-1:min_x+2] = [0,0,0]
    return (min_x, min_y)

# Extracts the angle in radians to the closest point to a collectible.
def extract_angle_to_closest_point(image):
    closest_point = extract_closest_point_to_collectible(image)
    closest_point_x, closest_point_y = closest_point

    car_center_y = int(3/4*IMAGE_HEIGHT)
    car_center_x = int(1/2*IMAGE_HEIGHT)

    diff_x = closest_point_x - car_center_x
    diff_y = car_center_y - closest_point_y

    vector = diff_y + 1j*diff_x

    angle = np.angle(vector)

    # Draw ray to closest point for debugging.
    draw_ray(image, angle)
    return angle