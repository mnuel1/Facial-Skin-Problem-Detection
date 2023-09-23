import numpy as np
import random
import numpy as np
import random
import tensorflow as tf
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKeras
from collections import Counter



class SkinDiseaseClassifier:
    def __init__(self):
        # Set random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # Configuration
        self.BATCH_SIZE = 16
        self.TRAIN_PATH = 'datasets/archive_3/Augmented Images/Augmented Images/FOLDS_AUG/fold5_AUG/Train'        
        self.VALID_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold5/Valid'
        
        self.input_shape = (224, 224, 3)
        self.opt = Adam(learning_rate=0.00001)
        self.n_classes = 6
        self.ft = 0

    def create_model(self, fine_tune):
        
        conv_base = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_shape)

        # Customize the top layers
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.5)(top_model)
        top_model = Dense(256, activation='relu')(top_model)
        top_model = Dropout(0.45)(top_model)
        output_layer = Dense(self.n_classes, activation='softmax')(top_model)

        # Combine base and top layers into the final model
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compile the model
        model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.summary()

        return model

    def train_model(self):

        # Data preprocessing
        trdata = ImageDataGenerator()
        train_data = trdata.flow_from_directory(directory=self.TRAIN_PATH, target_size=(224, 224), batch_size=self.BATCH_SIZE, shuffle=True, seed=42)

        vdata = ImageDataGenerator()
        val_data = vdata.flow_from_directory(directory=self.VALID_PATH, target_size=(224, 224), batch_size=self.BATCH_SIZE, shuffle=True, seed=42)

        # Create and compile the model
        model = self.create_model(self.ft)

        # Calculate class weights
        counter = Counter(train_data.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}

        # Checkpoint and early stopping
        checkpoint = ModelCheckpoint("./my_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)

        # Train the model
        history = model.fit(
            train_data,
            epochs=20,
            steps_per_epoch=len(train_data),
            class_weight=class_weights,
            validation_data=val_data,
            validation_steps=len(val_data),
            callbacks=[checkpoint, early_stop, PlotLossesKeras()]
        )


if __name__ == "__main__":
    classifier = SkinDiseaseClassifier()
    classifier.train_model()






