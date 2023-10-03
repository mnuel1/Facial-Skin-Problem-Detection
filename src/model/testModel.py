import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import numpy as np

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# CONFIG
BATCH_SIZE = 16


TEST_PATH = 'archive_4/'


# Load the pretrained model (replace 'your_model_path.h5' with the actual path)
model = keras.models.load_model('my_model.h5')

class_labels = { 4: ('nv', ' melanocytic nevi'),
                    6: ('mel', 'melanoma'),
                    2 :('bkl', 'benign keratosis-like lesions'), 
                    1:('bcc' , ' basal cell carcinoma'),
                    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
                    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
                    3: ('df', 'dermatofibroma')}

# Define the root directory where subfolders are located
root_dir = 'sets/'

# List of subfolder names
subfolders = ['melanocytic nevi', 'melanoma', 'benign keratosis-like lesions', 'basal cell carcinoma',
              'pyogenic granulomas and hemorrhage', 'Actinic keratoses and intraepithelial carcinomae', 'dermatofibroma']

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Load and preprocess the test image
test_image_path = 'test.jpg'  # Replace with the path to your test image
img = image.load_img(test_image_path, target_size=(28, 28))  # Resize the image to match model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0).reshape(-1, 28, 28, 3)  # Add batch dimension
# img_array = img_array / 255.0  # Normalize pixel values (if needed)

# Make predictions on the test image
predictions = model.predict(img_array)

# Get the class label with the highest probability
predicted_class_index = np.argmax(predictions)

# Print the predicted class and confidence score
predicted_class = class_labels[predicted_class_index]
confidence_score = predictions[0][predicted_class_index]

print(f'Predicted Class: {predicted_class}')
print(f'Confidence Score: {confidence_score:.4f}')
