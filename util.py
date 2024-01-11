import numpy as np
def flip_and_rotate(image):
    # Render the observation using pygame.
    # Rotate + flip the observation, see: 
    # https://stackoverflow.com/questions/66241275/pygame-rotates-camera-stream-from-opencv-camera.
    image = np.rot90(image, k=1, axes=(0, 1))
    image = np.flip(image, axis=0)
    return image
