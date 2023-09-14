import tensorflow as np
import PIL as p
import cv2 as cv2
import tflite as tf
import scipy as s

print(np.__version__)
print(p.__version__)
print(cv2.__version__)
print(tf.__version__)
print(s.__version__)

# import numpy as np
# import tensorflow as tf
# from keras import models,layers
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os
# from PIL import Image


# def load_custom_dataset():

#     image_paths = []
#     labels = []

#     # Create a dictionary to map class names to integer labels
#     class_to_label = {}
#     label_counter = 0

#     # Iterate through the subfolders in the image folder, assuming each subfolder represents a class
#     for class_folder in os.listdir('datasets/sample'):
#         class_path = os.path.join('datasets/sample', class_folder)
#         if os.path.isdir(class_path):
#             class_to_label[class_folder] = label_counter
#             label_counter += 1

#             for image_file in os.listdir(class_path):
#                 if image_file.endswith(".jpg"):  # You can adjust the file format
#                     image_path = os.path.join(class_path, image_file)
#                     image_paths.append(image_path)
#                     labels.append(class_folder)
                    
#     # Convert labels to integers using the class_to_label dictionary
#     y = np.array([class_to_label[label] for label in labels])

#     # Split the dataset into training and test sets (80% training, 20% test)
#     X_train, X_test, y_train, y_test = train_test_split(image_paths, y, test_size=0.2, random_state=42)

#     return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))

# # Load and preprocess images from file paths
# def load_and_preprocess_images(image_paths, target_size=(720, 420)):
#     images = []
#     for image_path in image_paths:
#         image = Image.open(image_path)
        
#         # Resize the image to the target size    
#         image = image.resize(target_size)

#         # Convert to NumPy array and normalize pixel values
#         image = np.array(image) / 255.0
       
#         images.append(image)
#     return np.array(images)

# # Define class names for your custom dataset
# class_names = ['acne-closed-comedo', 'acne-cystic', 'acne-excoriated', 'acne-open-comedo',
#                 'acne-pustular','ance-scar']

# # Function to display images with integer labels
# def display_images_with_labels(image_paths, labels, class_names, num_images=25):
#     plt.figure(figsize=(10, 10))
#     for i in range(min(num_images, len(image_paths))):
#         plt.subplot(5, 5, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)

#         # Load and display the image
#         image = Image.open(image_paths[i])
#         plt.imshow(image)
        
#         # Display the integer label
#         label = labels[i]
#         if label >= 0 and label < len(class_names):  # Check if the label is within the valid range
#             class_name = class_names[label]
#             plt.xlabel(f"({class_name})")
#         else:
#             plt.xlabel(f"(Invalid)")

#     plt.show()

# # Load your custom dataset with integer labels
# (train_images, train_labels), (test_images, test_labels) = load_custom_dataset()

# # Resize and preprocess train images to match the model's input shape
# X_train = load_and_preprocess_images(train_images, target_size=(720, 420))

# # Resize and preprocess test images to match the model's input shape
# X_test = load_and_preprocess_images(test_images, target_size=(720, 420))

# # Display some images from your custom dataset
# # display_images_with_labels(train_images, train_labels, class_names)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(420, 720, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
# # model.summary()

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(X_train, train_labels, epochs=10, validation_data=(X_test, test_labels))

# plt.figure(figsize=(10, 10))
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
# plt.show()
# print(test_acc)

# # Save the entire model to a file
# model.save("my_model.h5")
