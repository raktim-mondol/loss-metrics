import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K

from keras import backend as K

smooth = 1.
dropout_rate = 0.5

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

    
def jaccard_coef_v1(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)

def jacard_coef_v2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou_np(y_true, y_pred, smooth=1.):
    intersection = y_true * y_pred
    union = y_true + y_pred
    return (np.sum(intersection) + smooth) / (np.sum(union - intersection) + smooth)


def iou_thresholded_np(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos
    union = y_true + y_pred_pos
    return (np.sum(intersection) + smooth) / (np.sum(union - intersection) + smooth)


def iou_thresholded_np_imgwise(y_true, y_pred, threshold=0.5, smooth=1.):
    y_true = y_true.reshape((y_true.shape[0], y_true.shape[1]**2))
    y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1]**2))
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos   
    union = y_true + y_pred_pos
    return (np.sum(intersection, axis=1) + smooth) / (np.sum(union - intersection, axis=1) + smooth)



