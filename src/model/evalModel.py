
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
# CONFIG
BATCH_SIZE = 16

TRAIN_PATH = 'datasets/archive_3/Augmented Images/Augmented Images/FOLDS_AUG/fold5_AUG/Train'
TEST_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold4/Test'
VALID_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold5/Valid'


# Preproccess data
trdata = ImageDataGenerator()
train_data = trdata.flow_from_directory(directory=TRAIN_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True, seed=42)

vdata = ImageDataGenerator()
val_data = vdata.flow_from_directory(directory=VALID_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True,seed=42)

tsdata = ImageDataGenerator()
test_data = tsdata.flow_from_directory(directory=TEST_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=False, seed = 42)


model = load_model('my_model.h5')

model_preds = model.predict(test_data,test_data.samples//test_data.batch_size+1)
model_pred_classes = np.argmax(model_preds , axis=1)

# Print the predicted class
# print("Predicted class:", model_pred_classes) 

# Predict and display the images
for i in range(len(test_data)):
    batch_images, batch_labels = test_data[i]
    batch_predictions = model.predict(batch_images)
    batch_pred_classes = np.argmax(batch_predictions, axis=1)

    for j in range(len(batch_pred_classes)):
        image_array = batch_images[j]  # Get the image from the batch
        true_class = np.argmax(batch_labels[j])  # True class label
        predicted_class = batch_pred_classes[j]  # Predicted class label

        # Load the image using matplotlib (un-preprocess it)
        image_display = image.array_to_img(image_array, scale=False)

        # Display the image with true and predicted class labels
        plt.imshow(image_display)
        plt.title(f"True Class: {true_class}, Predicted Class: {predicted_class}")
        plt.axis('off')
        plt.show()







# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# true_classes = test_data.classes
# acc = accuracy_score(true_classes, model_pred_classes)
# print("DenseNet121-based Model Accuracy: {:.2f}%".format(acc * 100))

# print('Precision: %.3f' % precision_score(true_classes, model_pred_classes,average='macro'))
# print('Recall: %.3f' % recall_score(true_classes, model_pred_classes,average='macro'))
# print('F1 Score: %.3f' % f1_score(true_classes, model_pred_classes,average='macro'))


# from mlxtend.plotting import plot_confusion_matrix
# from sklearn.metrics import classification_report, confusion_matrix
# x = confusion_matrix(test_data.classes, model_pred_classes)
# plot_confusion_matrix(x)


# print('Classification Report')
# target_names = ['Chickenpox','Cowpox','HFMD','Healthy','Measles','Monkeypox']
# print(classification_report(test_data.classes, model_pred_classes))



# # GRAPH REPORT
# # Get the names of the ten classes
# class_names = test_data.class_indices.keys()

# def plot_heatmap(y_true, y_pred, class_names, ax, title):
#     cm = confusion_matrix(y_true, y_pred)
#     sns.heatmap(
#         cm, 
#         annot=True, 
#         square=True, 
#         xticklabels=class_names, 
#         yticklabels=class_names,
#         fmt='d', 
#         cmap=plt.cm.Greens, #Blues, YlGnBu, YlOrRd
#         cbar=False,
#         ax=ax
#     )
#     ax.set_title(title, fontsize=16)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.set_ylabel('True Label', fontsize=12)
#     ax.set_xlabel('Predicted Label', fontsize=12)


# #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
# fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
# plot_heatmap(true_classes, model_pred_classes, class_names, ax1, title="ResNet50")    
# # plot_heatmap(true_classes, vgg_pred_classes, class_names, ax2, title="Transfer Learning (VGG16) No Fine-Tuning")    
# # plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax3, title="Transfer Learning (VGG16) with Fine-Tuning")    

# fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
# fig.tight_layout()
# fig.subplots_adjust(top=1.25)
# plt.show()