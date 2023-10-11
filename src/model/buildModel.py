import numpy as np
import random
import tensorflow as tf
import keras
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from livelossplot import PlotLossesKeras

from sklearn.model_selection import train_test_split
import pandas as pd 
from imblearn.over_sampling import RandomOverSampler
from keras.utils import to_categorical

from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix



class SkinDiseaseClassifier:
    def __init__(self):

        # Set random seeds for reproducibility,consistency
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # Configuration
        data_dir = 'archive_4/hmnist_28_28_RGB.csv'
        self.data = pd.read_csv(data_dir)
        self.data.head()

    def create_model(self):
        
     
        model = keras.models.Sequential()

        # Create Model Structure
        model.add(keras.layers.Input(shape=[28, 28, 3]))
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPooling2D())

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.L1L2()))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Dense(units=7, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))

        model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

        # model.summary()

        return model

    def train_model(self):

        """ 
            Data Preproccesing
        """

        Label = self.data["label"]
        Data = self.data.drop(columns=["label"])   

         
        """
            Handling imbalanced datasets
        """
        oversample = RandomOverSampler(random_state=42)
        Data, Label  = oversample.fit_resample(Data, Label)
        Data = np.array(Data).reshape(-1, 28, 28, 3)
        print('Shape of Data :', Data.shape)  

        Label = np.array(Label)
# 2 :('bkl', 'benign keratosis-like lesions')
# 3: ('df', 'dermatofibroma')}
        """ Classes of Skin Cancer """
        classes = { 4: ('nv', ' melanocytic nevi'),
                    6: ('mel', 'melanoma'),                    
                    1:('bcc' , ' basal cell carcinoma'),
                    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
                    0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
                }
        
        X_train , X_test , y_train , y_test = train_test_split(Data , Label , test_size = 0.25 , random_state = 42)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
       
        #  Create and compile the model
        model = self.create_model()

        
        """
            Implements the checkpoint
                - for keeping the model to be the best model based on its val_accuracy
            Implement the early stop
                - for stopping the training if there is no improvements in val_accuracy
            Implement the learning rate reduction
                - to stabilize and avoid overfitting
        """
        checkpoint = ModelCheckpoint("./my_model1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)
       
        history = model.fit(X_train ,
                    y_train ,
                    epochs=20 ,
                    batch_size=128,
                    validation_data=(X_test , y_test) ,
                    callbacks=[learning_rate_reduction,PlotLossesKeras(),early_stop,checkpoint])
       

        train_score = model.evaluate(X_train, y_train, verbose= 1)
        test_score = model.evaluate(X_test, y_test, verbose= 1)

        print("Train Loss: ", train_score[0])
        print("Train Accuracy: ", train_score[1])
        print('-' * 20)
        print("Test Loss: ", test_score[0])
        print("Test Accuracy: ", test_score[1])

        y_true = np.array(y_test)
        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred , axis=1)
        y_true = np.argmax(y_true , axis=1)


        classes_labels = []
        for key in classes.keys():
            classes_labels.append(key)

        print(classes_labels)
        

        self.evaluate_model(y_test, y_pred, classes)

    def evaluate_model(self, y_true, y_pred, classes):
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Print evaluation metrics
        print("Model Accuracy: {:.2f}%".format(acc * 100))
        print('Precision: {:.3f}'.format(precision))
        print('Recall: {:.3f}'.format(recall))
        print('F1 Score: {:.3f}'.format(f1))

        # Generate a classification report
        target_names = classes
        print(classification_report(y_true, y_pred, target_names=target_names))

        # Plot heatmap
        class_names = classes
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
        self.plot_heatmap(y_true, y_pred, class_names, ax1, title="DenseNet121")    
        fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
        fig.tight_layout()
        
        plt.show()

    def plot_heatmap(self, y_true, y_pred, class_names, ax, title):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=class_names, yticklabels=class_names)
        ax.set_title(title, fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

if __name__ == "__main__":
    classifier = SkinDiseaseClassifier()
    classifier.train_model()






