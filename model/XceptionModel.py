imageSize=[299,299]
trainPath=r"/content/preprocessed dataset/preprocessed dataset/training"
testPath=r"/content/preprocessed dataset/preprocessed dataset/testing"

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('/content/preprocessed dataset/preprocessed dataset/training', target_size=(299, 299), batch_size=32, class_mode='categorical')
test_set=test_datagen.flow_from_directory('/content/preprocessed dataset/preprocessed dataset/testing', target_size=(299, 299), batch_size=32, class_mode='categorical')
