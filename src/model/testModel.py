
import numpy as np
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.models import load_model

image_path = "TEST.jpg"  # Replace with the path to your image
img = image.load_img(image_path, target_size=(224, 224))  # Resize to the input size
img = image.img_to_array(img)  # Convert to NumPy array
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Apply preprocessing (similar to ImageDataGenerator)
img = preprocess_input(img)

# Make predictions
predictions = load_model('my_model.h5').predict(img)

# Get the class with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
print("Predicted class:", predicted_class)