import sys
if len(sys.argv) != 2:
    print('Required input is python acat_ice <backbone>')
    sys.exit()
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
backbone = sys.argv[1]  # The other option is to import the backbone name from global_defs

models = get_ensemble_models(backbones = [backbone], train_iter=1)
test_img_path = .. # complete this path
img = cv2.imread(test_img_path)
img = preprocess_img(img)
img = np.reshape(img, (1, h, w, 1))
pred_mean, pred_ensemble, sdev = ensemble_prediction(models, img)
pred_mean_write_status = cv2.imwrite(inference_dir + "test_pred_mean.png", pred_mean*255.)
pred_ensemble_write_status = cv2.imwrite(inference_dir + "test_pred_ensemble.png", pred_ensemble*255.)
