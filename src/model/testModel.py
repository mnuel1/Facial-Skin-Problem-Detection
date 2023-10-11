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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import pandas as pd




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

folder_path = 'Datasets/test/akiec'

total_predictions = 0
correct_predictions = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)

        # test_data = ImageDataGenerator()
        # test_data.flow_from_directory(
        #     directory=folder_path,
        #     target_size=(28, 28),
        #     batch_size=BATCH_SIZE,
        #     shuffle=False,
        #     seed=42
        # )
        # Load and preprocess the test image
        test_image_path = 'test.jpg'  # Replace with the path to your test image
        img = image.load_img(image_path, target_size=(28, 28))  # Resize the image to match model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).reshape(-1, 28, 28, 3)  # Add batch dimension
        # img_array = img_array / 255.0  # Normalize pixel values (if needed)

        # Make predictions on the test image
        predictions = model.predict(img_array)

        # Get the class label with the highest probability
        predicted_class_index = np.argmax(predictions)

        
        # self.plot_training_history(model.history)
        # Evaluate the model
        # evaluate_model(tsdata, predicted_class_index)

        # Print the predicted class and confidence score
        predicted_class = class_labels[predicted_class_index]
        confidence_score = predictions[0][predicted_class_index]

        # Check if the prediction is correct
        is_correct = predicted_class_index == 3
        
        # Update counts
        total_predictions += 1
        correct_predictions += is_correct
       
        print(correct_predictions)
        print(f'Predicted Class: {predicted_class}')
        print(f'Confidence Score: {confidence_score:.4f}')

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100

print(f"Accuracy: {accuracy:.2f}%")

        # true_classes = test_data.classes

        # # Calculate accuracy, precision, recall, and F1-score
        # acc = accuracy_score(true_classes, predicted_class_index)
        # precision = precision_score(true_classes, predicted_class_index, average='macro')
        # recall = recall_score(true_classes, predicted_class_index, average='macro')
        # f1 = f1_score(true_classes, predicted_class_index, average='macro')

        # # Print evaluation metrics
        # print("DenseNet121-based Model Accuracy: {:.2f}%".format(acc * 100))
        # print('Precision: %.3f' % precision)
        # print('Recall: %.3f' % recall)
        # print('F1 Score: %.3f' % f1)

        # # Generate a classification report
        # print('Classification Report')
        # target_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']
        # print(classification_report(true_classes, predicted_class_index))

        # # Generate a confusion matrix and plot it
        # # x = confusion_matrix(true_classes, predicted_class_index)
        # # plot_confusion_matrix(x)

        # # Plot heatmap
        # class_names = test_data.class_indices.keys()
        # fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
    
        # fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
        # fig.tight_layout()
        
        # plt.show()

        
        

        # cm = confusion_matrix(true_classes, predicted_class_index)
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
        #         xticklabels=class_names, yticklabels=class_names)
        # ax1.set_title("Model", fontsize=16)
        # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        # ax1.set_ylabel('True Label', fontsize=12)
        # ax1.set_xlabel('Predicted Label', fontsize=12)