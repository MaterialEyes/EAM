from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread
import sys, skimage, json, random, os, cv2, time
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.python.framework import tensor_util
from tensorflow.keras.utils import plot_model
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models import get_preprocessing
from skimage import exposure
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from global_defs import *
from sklearn.metrics import brier_score_loss

def conv(filters, kernel_size = 3, stride = 2):
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2())
    return x

def preprocess_img(img):
    #img = exposure.equalize_adapthist(img)
    img = cv2.resize(img, (s1,s2), interpolation=cv2.INTER_CUBIC)
    thi, bw = cv2.threshold(img , np.median(img), 255, cv2.THRESH_BINARY) 
    w, h  = img.shape
    bw = (255.0 - bw)/255.
    for i in range(w):
        for j in range(h):
            img[i][j] = bw[i][j]*img[i][j]
    img = np.reshape(img, (s1,s2,1))
    img = img/np.max(img)
    return img

def preprocess_mask(mask):
    mask = cv2.resize(mask, (s1,s2), interpolation=cv2.INTER_CUBIC)
    mask = mask/255.
    mask = np.reshape(mask, (s1,s2,1))
    return mask

def preprocess_test_mask(img_gt):
    img_gt = cv2.resize(img_gt, (s1,s2), interpolation=cv2.INTER_CUBIC)
    thi, img_gt = cv2.threshold(img_gt , np.median(img_gt), 255, cv2.THRESH_BINARY) 
    img_gt = img_gt/255
    return img_gt

def smooth(img):
    return 0.5*img + 0.5*(
        np.roll(img, +1, axis=0) + np.roll(img, -1, axis=0) +
        np.roll(img, +1, axis=1) + np.roll(img, -1, axis=1) )

def read_data(img_mask_path, pp=False):
    img_in = imread(img_mask_path[0])
    #img_in = exposure.equalize_adapthist(img_in)
    if pp==True:
        img_in = preprocess_img(img_in)
    else:
        img_in = cv2.resize(img_in, (s1,s2), interpolation=cv2.INTER_CUBIC)
        img_in = np.reshape(img_in, (s1,s2,1))
        img_in = img_in/np.max(img_in)
    mask_in = imread(img_mask_path[1]) #get_mask_from_coco(img_mask_path[1]) 
    mask_in = preprocess_mask(mask_in)
    return (img_in, mask_in)

def get_aug_dataset(data_paths, pp = False):
    num_total = len(data_paths)
    out_imgs = np.zeros((aug*num_total,)+(s1,s2)+(1,)) # define the input images for the model
    out_masks = np.zeros((aug*num_total,)+(s1,s2)+(2,)) # define the input masks for the model
    for i, img_mask_path in enumerate(data_paths):
        img, mask = read_data(img_mask_path, pp) # import images, save as array, crop and resize the image to (256,256,1)
        #mask_flipped = np.flip(mask, 0)
        mask = tf.keras.utils.to_categorical(mask, num_classes = 2, dtype ="uint8")
        #mask_flipped = tf.keras.utils.to_categorical(mask_flipped, num_classes = 2, dtype ="uint8")
        out_imgs[aug*i,...] = img # create single array of images
        out_imgs[aug*i + 1, ...] = np.flip(img, 0)
        out_imgs[aug*i + 2, ...] = np.flip(img, 1)
        out_imgs[aug*i + 3, ...] = np.flip(np.flip(img,0), 1)
        #out_imgs[aug*i + 4, ...] = skimage.util.random_noise(img)
        out_masks[aug*i,...] = mask # create single array of masks
        out_masks[aug*i + 1,...] = np.flip(mask, 0) #mask_flipped
        out_masks[aug*i + 2,...] = np.flip(mask, 1) #mask_flipped
        out_masks[aug*i + 3,...] = np.flip(np.flip(mask, 0), 1) 
        #out_masks[aug*i + 4,...] = mask
    return out_imgs, out_masks

def get_test_dataset():
    out_imgs = np.zeros((1,)+(s1,s2)+(1,))
    out_masks = np.zeros((1,)+(s1,s2)+(2,))
    for i, img_mask_path in enumerate(test_paths):
        img, mask = read_data(img_mask_path)
        out_imgs[i,...] = img
        out_masks[i,...] = mask
    return out_imgs, out_masks

def get_shuffled_datasets(images, masks):
    indices = np.arange(1, len(images)+1)
    random.shuffle(indices)

    images_shuffled = np.zeros((len(images),)+(s1,s2)+(1,))
    masks_shuffled = np.zeros((len(images),)+(s1,s2)+(2,))

    for n in range(len(indices)):
        images_shuffled[n,...] = images[indices[n]-1,...]
        masks_shuffled[n,...] = masks[indices[n]-1,...]
    
    return images_shuffled, masks_shuffled

def model_init(arch, backbone):
    #with strategy.scope():
    preprocess_input = get_preprocessing(backbone)
    N = 1
    if arch == "unet":
        base_model = Unet(backbone, classes = 2, encoder_weights='imagenet')
    if arch == "fpn":
        base_model = FPN(backbone, classes = 2, encoder_weights='imagenet')
    if arch == "linknet":
        base_model = Linknet(backbone, classes = 2, encoder_weights='imagenet')
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)
    model_1 = Model(inp, out, name=base_model.name)
    return model_1

def trunc_model_init(arch, backbone):
    preprocess_input = get_preprocessing(backbone)
    N = 1
    if arch == "unet":
        base_model = Unet(backbone, classes = 2, encoder_weights='imagenet')
    if arch == "fpn":
        base_model = FPN(backbone, classes = 2, encoder_weights='imagenet')
    if arch == "linknet":
        base_model = Linknet(backbone, classes = 2, encoder_weights='imagenet')
    new_base_model = Model(base_model.input, base_model.layers[-2].output)
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = new_base_model(l1)
    model_1 = Model(inp, out)
    return model_1

#"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    
def get_ensemble_models(backbones = ["densenet169", "vgg19", "inceptionresnetv2", "inceptionv3", "seresnet50", "vgg16"], weight_dir = '/global/cscratch1/sd/ab1992/acat_trained_weights', train_iter = 1):
    # Currently, only using two backbones because of issues with large file storage on git
    arch = 'unet'
    models = []
    for backbone in backbones:
        model = trunc_model_init(arch, backbone)
        if train_iter==1:
            model.load_weights(weight_dir + f'/{backbone}/trained_weights_{backbone}_{arch}_8.h5') 
        elif train_iter==2:
            model.load_weights(weight_dir + f'/{backbone}/retrained_weights_recrys_{backbone}_{arch}_8.h5') 
        elif train_iter==3:
            model.load_weights(weight_dir + f'/{backbone}/retrained_weights_recrys_c3_{backbone}_{arch}_8.h5') 
        elif train_iter==4:
            model.load_weights(weight_dir + f'/{backbone}/retrained_weights_recrys_c3_c8_{backbone}_{arch}_8.h5') 
        models.append(model)
    return models

def get_ensemble2_models(backbone = 'vgg19'):
    weight_dir = f'/global/cscratch1/sd/ab1992/acat_trained_weights/{backbone}_ensemble2'
    arch = 'unet'
    models = []
    for i in range(1,6):
        model = trunc_model_init(arch, backbone)
        model.load_weights(weight_dir + f'/trained_weights_{backbone}_{arch}_8_{i}.h5') 
        models.append(model)
    return models

def ensemble_prediction(models, image):
    preds = []
    for model in models:
        preds.append(model.predict_on_batch(image))
    pred_probs = np.zeros((len(preds), s1,s2))
    for m in range(len(models)):
        for x in range(s1):
            for y in range(s2):
                pred_probs[m][x][y] = np.exp(-preds[m][0][x][y][0])/(1 + np.exp(-preds[m][0][x][y][0]))
    pred_mean = np.zeros((s1,s2))
    for pred_prob in pred_probs:
        pred_mean += pred_prob[:,:]/(len(preds))
    pred_ensemble = np.where(pred_mean > 0.5, 1.0, 0.0)
    sdev = np.zeros((s1,s2))
    for i in range(s1):
        for j in range(s2):
            for m in range(len(preds)):
                sdev[i][j] += (pred_probs[m][i][j] - pred_mean[i][j])**2
            sdev[i][j] = sdev[i][j]/len(preds)
    return pred_mean, pred_ensemble, sdev

def get_ppca_sdev_dice(models, out_imgs, out_masks):
    stats = []
    for i, img in enumerate(out_imgs):
        img = np.reshape(img, (1,s1,s2,1))
        pred_mean, pred_ensemble, sdev = ensemble_prediction(models, img)
        ppa = 0
        for x in range(s1):
            for y in range(s2):
                if pred_ensemble[x][y] == out_masks[i][x][y][1]:
                    ppa += 1
        dice = np.sum(out_masks[i][pred_ensemble==1])*2.0 / (np.sum(out_masks[i]) + np.sum(pred_ensemble))
        #brier = brier_score_loss(np.ravel(out_masks[i][...,1]), np.ravel(pred_mean))
        stats.append([ppa/(s1*s2), np.sum(sdev)/(s1*s2), dice])
    return stats

def get_roc_stats(models, test_dir = '/global/cscratch1/sd/ab1992/ice_kdd/D1/testing/', test_size = 3):
    y_true = []
    y_pred = []
    for i in range(1,test_size+1):
        img = imread(test_dir + f"stack_{i}.png")
        img = np.reshape(preprocess_img(img), (1,s1,s2,1))
        pred_means, pred_ensemble, sdev = ensemble_prediction(models, img)  # get prediction for each pixel in the image
        true_seg = imread(test_dir + f"stack_{i}_label.png")
        true_seg = np.reshape(true_seg, (s1,s2,1))//255
        y_true.append(true_seg.flatten())  # flatten all targets
        y_pred.append(pred_ensemble.flatten())  # flatten all predictions
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr,tpr)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return fpr, tpr, roc_thresholds, roc_auc, ix, roc_thresholds[ix]

def get_pr_stats(models):
    y_true = []
    y_pred = []
    for i in range(1,test_size+1):
        img = imread(test_dir + f"stack_{i}.png")
        img = np.reshape(preprocess_img(img), (1,s1,s2,1))
        pred_means, pred_ensemble, sdev = ensemble_prediction(models, img)  # get prediction for each pixel in the image
        true_seg = imread(test_dir + f"stack_{i}_label.png")
        true_seg = np.reshape(true_seg, (s1,s2,1))//255
        y_true.append(true_seg.flatten())  # flatten all targets
        y_pred.append(pred_ensemble.flatten())  # flatten all predictions
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    p, r, pr_thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(r,p)
    fscore = (2 * p * r) / (p + r)
    ix = np.argmax(fscore)
    return p, r, pr_thresholds, pr_auc, ix, pr_thresholds[ix]

def get_center_of_mass(img):
    xs = []
    ys = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 1.0:  
                xs.append(j)   # axes switched here because of numpy reading format
                ys.append(i)
    origin = (np.mean(xs), np.mean(ys))
    return origin

def get_lines(img, origin, num_lines):
    angles = np.arange(0, 180, 180//num_lines)
    slopes = [np.tan((np.pi/180)*angle) for angle in angles]
    lines_x = []
    lines_y = []
    for s in range(len(slopes)):
        xs = []
        ys = []
        if slopes[s] == np.tan((np.pi/180)*90):
            xs = [origin[0] for _ in range(100)]
            ys = np.arange(0, np.shape(img)[0], np.shape(img)[0]/100)
            lines_x.append(xs)
            lines_y.append(ys)    
            continue
        for x in range(0, img.shape[1] + 1, 1):
            if ((origin[1] + slopes[s]*(x - origin[0])) > 0) & ((origin[1] + slopes[s]*(x - origin[0])) < np.shape(img)[0]):
                xs.append(x)
                ys.append(origin[1] + slopes[s]*(x - origin[0]))
        lines_x.append(xs)
        lines_y.append(ys)      
    return slopes, lines_x, lines_y

def get_intersection(img, lines_x, lines_y):
    intersections = [] #10 x 2 x 2 (num lines x num of intersection with boundary x num of coordinates)
    for n in range(len(lines_x)):
        #print(lines_y[n][-1], lines_y[n][-1] > 0.5*np.shape(img)[1])
        intersection = []
        if lines_y[n][-1] > 0.5*np.shape(img)[1]:
            for i in range(len(lines_x[n])):
                if img[int(lines_y[n][i])][int(lines_x[n][i])] >= 0.5:
                    intersection.append([int(lines_x[n][i]), int(lines_y[n][i])])
                    break
            for i in range(len(lines_x[n])-2, -1, -1):
                if img[int(lines_y[n][i])][int(lines_x[n][i])] >= 0.5:
                    intersection.append([int(lines_x[n][i]), int(lines_y[n][i])])
                    break
        else:
            for i in range(len(lines_x[n])-2, -1, -1):
                if img[int(lines_y[n][i])][int(lines_x[n][i])] >= 0.5:
                    intersection.append([int(lines_x[n][i]), int(lines_y[n][i])])
                    break
            for i in range(len(lines_x[n])):
                if img[int(lines_y[n][i])][int(lines_x[n][i])] >= 0.5:
                    intersection.append([int(lines_x[n][i]), int(lines_y[n][i])])
                    break
        intersections.append(intersection)
    return intersections

def get_curvature(img):
    for i in range(0, s1, 2):   #Getting the lowest x line that intersects with the mask
        if np.sum(img[i])!=0.0:
            low_x = i
            break
    for i in range(low_x, s1, 2):  # Getting the highest x line that intersects with the mask
        if np.sum(img[i])==0.0:
            high_x = i-1
            break
    x = []
    y = []
    for i in range(low_x, high_x+1, 2):  # Getting the intersection closest to left axis
        for j in range(s1):
            if img[i][j] >= 0.8:
                x.append(i)
                y.append(j)
                break
    for i in range(high_x, low_x-1, -2):  # Getting the intersection closest to right axis
        for j in range(s1-1, 0, -1):
            if img[i][j] >= 0.8:
                x.append(i)
                y.append(j)
                break
    fprimes = []
    fdprimes = []
    kappas = []
    for i in range(len(x)-1):
        if((x[i+1] - x[i-1]) == 0.0):
            grad = 0.0
        else:
            grad = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        if ((x[i+1] - x[i])**2) == 0.0:
            grad2 = 0.0
        else:
            grad2 = (y[i+1] + y[i-1] - 2*y[i])/((x[i+1] - x[i])**2)
        fprimes.append(grad)
        fdprimes.append(grad2)
        kappas.append(np.abs(grad2)/pow((1 + grad**2),1.5))
    return x, y, kappas
