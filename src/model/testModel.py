import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# CONFIG
BATCH_SIZE = 16


TEST_PATH = 'archive_4/'


# Load the pretrained model (replace 'your_model_path.h5' with the actual path)
model = keras.models.load_model('my_model.h5')

# Load and preprocess the test image
test_image_path = 'TEST.jpg'  # Replace with the path to your test image
img = image.load_img(test_image_path, target_size=(28, 28))  # Resize the image to match model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0).reshape(-1, 28, 28, 3)  # Add batch dimension
# img_array = img_array / 255.0  # Normalize pixel values (if needed)

# Make predictions on the test image
predictions = model.predict(img_array)

# Get the class label with the highest probability
predicted_class_index = np.argmax(predictions)

# Assuming 'self.data' contains class labels mapping (you may need to adjust this)
class_labels = { 4: ('nv', ' melanocytic nevi'),
                    6: ('mel', 'melanoma'),
                    2 :('bkl', 'benign keratosis-like lesions'), 
                    1:('bcc' , ' basal cell carcinoma'),
                    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
                    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
                    3: ('df', 'dermatofibroma')}

# Print the predicted class and confidence score
predicted_class = class_labels[predicted_class_index]
confidence_score = predictions[0][predicted_class_index]

print(f'Predicted Class: {predicted_class}')
print(f'Confidence Score: {confidence_score:.4f}')
# test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
#     'base_dir/val_dir',
#     target_size=(28, 28,3),  # Resize images to match your model's input shape
#     batch_size=16,
#     shuffle=False  # Important: Ensure the order of predictions matches the order of the images
# )




# model = load_model('my_model.h5')

# # Make predictions
# model_preds = model.predict(test_data, steps=len(test_data), verbose=1)
# model_pred_classes = np.argmax(model_preds , axis=1)

# # Print the predicted class
# # print("Predicted class:", model_pred_classes) 

# # Predict and display the images
# for i in range(len(test_data)):
#     batch_images, batch_labels = test_data[i]
#     batch_predictions = model.predict(batch_images)
#     batch_pred_classes = np.argmax(batch_predictions, axis=1)

#     for j in range(len(batch_pred_classes)):
#         image_array = batch_images[j]  # Get the image from the batch
#         true_class = np.argmax(batch_labels[j])  # True class label
#         predicted_class = batch_pred_classes[j]  # Predicted class label

#         # Load the image using matplotlib (un-preprocess it)
#         image_display = image.array_to_img(image_array, scale=False)

#         # Display the image with true and predicted class labels
#         plt.imshow(image_display)
#         plt.title(f"True Class: {true_class}, Predicted Class: {predicted_class}")
#         plt.axis('off')
#         plt.show()




