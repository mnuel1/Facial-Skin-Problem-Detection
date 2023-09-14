from keras.models import load_model
from PIL import Image
import numpy as np

def image(image_path):
    # Load and preprocess the input image
    input_image_path = image_path  # Replace with the path to your input image
    input_image = Image.open(input_image_path)
    input_image = input_image.resize((720, 420))  # Resize to match the model's input shape
    input_image = np.array(input_image) / 255.0  # Normalize pixel values
    input_image = np.expand_dims(input_image, axis=0)  # Add a batch dimension
    
    return input_image


# Load the saved model
loaded_model = load_model("my_model.h5")  # Replace with your model filename


ndArray = image('OIP.jpg')

# Use the loaded model to predict the class label
predictions = loaded_model.predict(ndArray)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Define class names for your dataset
class_names = ['acne-closed-comedo', 'acne-cystic', 'acne-excoriated', 'acne-open-comedo',
                'acne-pustular','acne-scar']

# Get the predicted class name
predicted_class_name = class_names[predicted_class_index]

print(f"The predicted class is: {predicted_class_name}")
