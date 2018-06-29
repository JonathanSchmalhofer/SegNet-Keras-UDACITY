import csv
import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt
import time

import keras
from keras import optimizers
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import __version__ as version_from_keras

import tensorflow as tf
import h5py
import sklearn
from sklearn.model_selection import train_test_split


################################################################################
## H E L P E R S

def CheckTensorflowGPU():
    if tf.test.gpu_device_name():
        print('++++ GPU-Version of TF seems to be used')
        print('     Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print('---- CPU-Version of TF seems to be used')

def CheckKerasVersion(model_filename):
    # check that model Keras version is same as local Keras version
    f = h5py.File(model_filename, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(version_from_keras).encode('utf8')
    print('You are using Keras version ', keras_version)
    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

def BackupExistingModelFile(filename):
    if os.path.isfile(filename):
        date_time = time.strftime("%Y%m%d_%H%M%S")
        file_name_stem, file_extension = os.path.splitext(filename)
        new_filename = file_name_stem + '_' + date_time + file_extension
        temp_model = None
        temp_model = load_model(filename)
        temp_model.save(new_filename)
        del temp_model
        print('Backuped  {}  to  {}'.format(filename,new_filename))

def Normalize(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

def OneHotCategorial(labels, sizes = (360,480), nb_classes=32):
    rows = sizes[0]
    cols = sizes[1]
    x = np.zeros([rows,cols,nb_classes])
    for i in range(0,rows):
        for j in range(0,cols):
            x[i,j,labels[i][j]] = 1
    x_flattened = x.reshape(rows*cols,nb_classes)
    return x_flattened

def PreprocessInput(X):
    return imagenet_utils.preprocess_input(X)

def QualityCheckLabels(labels,nb_classes = 32):
    for label in labels:
        label_content = cv2.imread(label)[:,:,0]
        if label_content.max() >= nb_classes:
            print(label)

def Generator(data, labels, batch_size = 32, nb_classes = 32):
    assert(len(data) == len(labels))
    num_samples = len(data)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_data   = data[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            images_data   = []
            images_labels = []
            assert(len(batch_data) == len(batch_labels))
            current_batch_size = len(batch_data)
            for idx,_ in enumerate(batch_data):
                batch_sample_data  = batch_data[idx]
                batch_sample_label = batch_labels[idx]
                image_in = Normalize(cv2.imread(batch_sample_data))                    # shape (360,480,3)
                image_out = OneHotCategorial(cv2.imread(batch_sample_label)[:,:,0], sizes = (image_in.shape[0],image_in.shape[1],))
                images_data.append(image_in)
                images_labels.append(image_out)
                image_in_flipped  = np.fliplr(image_in)  # Flipped
                image_out_flipped = np.fliplr(image_out) # Flipped
                images_data.append(image_in_flipped)
                images_labels.append(image_out_flipped)
            X_train = np.array(images_data)
            y_train = np.array(images_labels)
            X_train = PreprocessInput(X_train)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

# Note: Please use forward slashes only!
def AppendDataAndLabels(data, labels, segnet_repo_path, list_path):
    with open(list_path) as csvfile:
        # Ignore Header line in CSV
        # see: https://stackoverflow.com/questions/11349333
        has_header = csv.Sniffer().has_header(csvfile.read(4096))
        csvfile.seek(0) # Rewind
        reader = csv.reader(csvfile, delimiter='\t')
        if has_header:
            next(reader) # Skip header row
        for line in reader:
            data_path  = segnet_repo_path + '/' + line[0]
            label_path = segnet_repo_path + '/' + line[1]
            data.append(data_path)
            labels.append(label_path)
        data, labels = sklearn.utils.shuffle(data,labels)
    return data, labels

def GetSegNetArchitecture(input_shape=(360, 480, 3), classes = 32, batch_size = 32):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    
    model = Sequential()
    # Encoder
    model.add(Convolution2D(64, 3, 3, border_mode="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(512, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    # Decoder
    model.add(Convolution2D(512, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    
    model.add(Convolution2D(classes, 1, 1, border_mode="valid"))
    model.add(Reshape((input_shape[0]*input_shape[1], classes)))
    model.add(Activation("softmax"))
    return model


