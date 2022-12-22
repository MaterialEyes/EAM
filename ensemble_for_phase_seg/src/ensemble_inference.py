import sys
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
import skimage, json, random, os, cv2, time
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models import get_preprocessing
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_util
from acat_global import *
from acat_aux_funcs import *

strategy = tf.distribute.MirroredStrategy()

models = get_ensemble_models(backbones = backbones_for_inference, train_iter=1)
test_files = os.listdir(testing_dir)
for image_file_path in test_files:
	filename = image_file_path.split('/')[-1].split('.')[0]
	img = cv2.imread(image_file_path)
	img = preprocess_img(img)
	img = np.reshape(img, (1, h, w, 1))
	pred_mean, pred_ensemble, sdev = ensemble_prediction(models, img)
	pred_mean_write_status = cv2.imwrite(inference_dir + f"{filename}_mean_pred.png", pred_mean*255.)
	pred_ensemble_write_status = cv2.imwrite(inference_dir + f"{filename}_std_dev.png", sdev*255.)
