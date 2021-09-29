from types import new_class
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate 
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.transform import resize



from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import  Reshape
from tensorflow.keras.layers.experimental.preprocessing import  Resizing




import os
import random
import numpy as np
import re
"""
from tensorflow.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers.merge import concatenate
"""


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs],)
    return model


def get_multiUnet(input_img, nclasses, n_filters=16, dropout=0.1, batchnorm=True):

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(nclasses, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs],)
    return model


class DataLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, num_bands, input_img_paths, target_img_paths, nclasses=2, augment=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.num_bands = num_bands
        self.nclasses = nclasses

        self.augment = augment
        if self.augment:
            self.augmentation = keras.Sequential([
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.2),
            ])

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size +
                     (self.num_bands,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            with rasterio.open(path) as src:
                rast = src.read()

            img = reshape_as_image(rast)
            
            x[j] = img

        if self.nclasses > 2:
            y = np.zeros((self.batch_size,) + self.img_size +
                         (self.nclasses,), dtype="uint8")
        else:
            y = np.zeros((self.batch_size,) +
                         self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):
            with rasterio.open(path) as src:
                rast = src.read()
                
            img = reshape_as_image(rast)
            
            if self.nclasses > 2:
                img = keras.utils.to_categorical(img,num_classes=self.nclasses)
                y[j] = img
    
            else:
                y[j] = img

        if self.augment:
            nimg = np.concatenate([x, y], axis=3)
            nimg = self.augmentation(nimg)

            if self.nclasses > 2:
                 x, y = nimg[:, :, :, :-self.nclasses], nimg[:, :, :, -self.nclasses:]
            else:
                x, y = nimg[:, :, :, :-1], nimg[:, :, :, -1:]

        return x, y


def train_folders(input_dir, target_dir):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".tif")
        ])

    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".tif") and not fname.startswith(".")
        ])

    return input_img_paths, target_img_paths


def split_train_val(input_dir, target_dir, num_val_samples, batch_size, img_size, nbands, nclasses=2, augment=False):
    # Split our img paths into a training and a validation set

    input_img_paths, target_img_paths = train_folders(input_dir, target_dir)
    val_samples = num_val_samples
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = DataLoader(
        batch_size, img_size, nbands, train_input_img_paths, train_target_img_paths,nclasses=nclasses, augment=augment
    )
    val_gen = DataLoader(batch_size, img_size, nbands,
                         val_input_img_paths, val_target_img_paths, nclasses=nclasses,augment=augment)

    return train_gen, val_gen


def mk_input_img(in_shape):
    input_img = Input(in_shape)
    return input_img


def predict(model):
    def predictionFunc(raster):
        img = rasterio.plot.reshape_as_image(raster)

        img = np.expand_dims(img, 0)

        y = model.predict(img)[0]
        y = reshape_as_raster(y)
        return y

    return predictionFunc





