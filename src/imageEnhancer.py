import cv2
import numpy as np

# Load the image
image = cv2.imread('sets/Actinic keratoses and intraepithelial carcinomae/1.jpg', cv2.IMREAD_COLOR)

# Convert to grayscale (if not already)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase brightness by adding a constant value
brightness_increase = 50  # Adjust this value to control brightness
brightened_image = cv2.add(gray_image, brightness_increase)

# Clip pixel values to ensure they stay within the valid range (0-255)
brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)

# Define a lower and upper threshold for skin color (adjust these values)
lower_skin_color = np.array([90, 80, 70], dtype=np.uint8)
upper_skin_color = np.array([120, 120, 120], dtype=np.uint8)

# Create a mask for skin color pixels
skin_color_mask = cv2.inRange(image, lower_skin_color, upper_skin_color)

# Create a mask for non-skin color pixels
non_skin_color_mask = cv2.bitwise_not(skin_color_mask)

# Increase brightness of non-skin color pixels
brightened_non_skin = cv2.add(image, brightness_increase, mask=non_skin_color_mask)

# Combine the brightened non-skin color pixels with the skin color pixels
brightened_image = cv2.addWeighted(brightened_non_skin, 1, image, 1, 0)

# Save the enhanced image
cv2.imwrite('brightened_skin_lesion.jpg', brightened_image)
