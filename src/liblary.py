import tensorflow as tf


import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model

from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers import UpSampling2D, Dropout, Add, Multiply, Subtract, AveragePooling2D
from keras.layers import Activation, SpatialDropout2D
from keras.layers import Dense, Lambda
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, Flatten

from keras.utils import plot_model

from keras.optimizers import * 
from keras.callbacks import *
from keras.activations import *

from sklearn.metrics import classification_report, confusion_matrix

import os
import numpy as np
import pandas as pd
import random
import glob
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.image as mimg
# %matplotlib inline
from PIL import Image
from scipy import misc