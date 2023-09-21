from keras.applications import DenseNet121
import tensorflow as tf

from keras.models import Model

from keras.layers import Dense,Flatten,Dropout
from livelossplot import PlotLossesKeras
from keras.preprocessing.image import ImageDataGenerator


from keras.optimizers import * 
from keras.callbacks import *
from keras.activations import *

# CONFIG
BATCH_SIZE = 16

TRAIN_PATH = 'datasets/archive_3/Augmented Images/Augmented Images/FOLDS_AUG/fold5_AUG/Train'
TEST_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold5/Test'
VALID_PATH = 'datasets/archive_3/Original Images/Original Images/FOLDS/fold5/Valid'



def create_model(input_shape, n_classes , optimizer, fine_tune):
    """
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = DenseNet121(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Initialize the model with weights from HAM10000 keeping by name True
    #conv_base.load_weights("/kaggle/input/weightofham/DenseNet121.h5",by_name=True)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
#     if fine_tune > 0:
#         for layer in conv_base.layers[:-fine_tune]:
#             layer.trainable = False
#     else:
#         for layer in conv_base.layers:
#             layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)    
    top_model = Dense(256, activation='relu')(top_model)
    top_model = Dropout(0.15)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    
    # Compiles the model for training.
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
    # model.summary()
    
    return model

# Preproccess data
trdata = ImageDataGenerator()
train_data = trdata.flow_from_directory(directory=TRAIN_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True, seed=42)

vdata = ImageDataGenerator()
val_data = vdata.flow_from_directory(directory=VALID_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=True,seed=42)

tsdata = ImageDataGenerator()
test_data = tsdata.flow_from_directory(directory=TEST_PATH, target_size=(224,224),batch_size=BATCH_SIZE, shuffle=False, seed = 42)


input_shape = (224, 224, 3)
opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
n_classes = 6
ft = 0

# First we'll train the model without Fine-tuning
model = create_model(input_shape, n_classes, opt, ft)

STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
STEP_SIZE_VALID = val_data.n//test_data.batch_size

from collections import Counter
counter = Counter(train_data.classes)                       
max_val = float(max(counter.values()))   
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
class_weights


checkpoint = ModelCheckpoint("./my_model.h5", monitor='val_accuracy', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto')

early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)

history = model.fit(train_data,
                    epochs =100,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    class_weight = class_weights,
                    validation_data = val_data,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[checkpoint, early_stop, PlotLossesKeras()]
                    )



