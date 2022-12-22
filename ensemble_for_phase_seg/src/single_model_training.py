## This code is for the case when the model encoder type is obtained from the user

import sys
if len(sys.argv) != 2:
    print('Required input is python single_model_training <backbone>')
    sys.exit()
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
import skimage, json, random, os, cv2, time
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models import get_preprocessing
from tensorflow.python.framework import tensor_util
from global_defs import *
from aux_funcs import *

image_mask_paths = [(training_dir + f"stack_{i}.png", training_dir + f"stack_{i}_label.png") for i in range(start, end+1)] # store the image file paths for the training data

images, masks = get_aug_dataset(image_mask_paths, False)
images_shuffled, masks_shuffled = get_shuffled_datasets(images, masks)
strategy = tf.distribute.MirroredStrategy()
backbone = sys.argv[1]  # The other option is to import the backbone name from global_defs
print(backbone)
          
if not os.path.exists(weights_dir + f'{backbone}'):
    os.mkdir(weights_dir + f'{backbone}')
    
with strategy.scope():
    model = model_init(arch, backbone)
    lr = lr_per_replica*strategy.num_replicas_in_sync
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'],  run_eagerly=True)
train_images, train_masks = images_shuffled[:], masks_shuffled[:]
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=200, verbose=1, mode="min", restore_best_weights=True)
global_batch_size = batch_size_per_replica*strategy.num_replicas_in_sync
history = model.fit(train_images, train_masks, batch_size=global_batch_size, epochs=epochs, verbose = verbose, validation_split = val_split, callbacks = [callback])  
print(f"Accuracy with {batch_size_per_replica} and {lr} is: {history.history['accuracy'][-1]}")
model.save_weights(weights_dir + f'{backbone}/trained_weights_{backbone}_{arch}_{batch_size_per_replica}.h5')  
