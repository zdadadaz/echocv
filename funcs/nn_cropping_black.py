from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import timeit
import random

from skimage.draw import circle
from collections import deque
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
############################
# Neural Network Functions #
############################

# Convolution Layer
def conv(x, filter_size, num_filters, stride, weight_decay, name, padding='SAME', groups=1, trainable=True, relu=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=trainable,
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', shape=[num_filters], trainable=trainable, initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases

def deconv(x, filter_size, num_filters, stride, weight_decay, name, padding='SAME', relu=True):
    activation = None
    if relu:
        activation = tf.nn.relu
    return tf.layers.conv2d_transpose(x, num_filters, filter_size, stride, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=activation, name=name)
    
    
# Fully Connected Layer
def fc(x, num_out, weight_decay,  name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        weights = tf.get_variable('W',
                                  shape=[num_in, num_out], 
                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                  trainable=trainable, 
                                  regularizer=regularizer,
                                  collections=['variables'])
        biases = tf.get_variable('b', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
        x = tf.matmul(x, weights) + biases
        if relu:
            x = tf.nn.relu(x) 
    return x

# Local Response Normalization
def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


#################################
# Training/Validation Functions #
#################################

def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output

def validate(sess, model, x_test, y_test):
    '''
    Calculates accuracy of validation set
    
    @params sess: Tensorflow Session
    @params model: Model defined from a neural network class
    @params x_test: Numpy array of validation images
    @params y_test: Numpy array of validation labels
    @params batch_size: Integer defining mini-batch size
    '''
    scores = [0] * int(y_test.shape[3]-1)
    for i in range(int(x_test.shape[0])):
        for j in range(int(y_test.shape[3]-1)):
            gt = np.argmax(y_test[i,:,:,:], 2)
            gt = create_seg(gt, j+1)
            pred = np.argmax(model.predict(sess, x_test[i:i+1])[0,:,:,:], 2)
            pred = create_seg(pred,j+1)           
            overlap = np.minimum(gt, pred)
            dice = 2*np.sum(overlap)/(np.sum(gt) + np.sum(pred))
            scores[j] = scores[j] + dice 
            
    return [score/float(x_test.shape[0]) for score in scores]


def validate_bagging(sess, model, x_test, y_test, num_sets):
    '''
    Calculates accuracy of validation set by randomly sampling (with replacement)
    the validation set. Provides more accurate estimation of model accuracy.
    
    @params many same as validate()
    @params num_sets: Integer defining number of validation sets to test
    '''
    val_accs = []
    for i in range(num_sets):
        indicies = (np.random.sample((x_test.shape[0],))*x_test.shape[0]).astype(int)
        val_accs.append(validate(sess,model,x_test[indicies],y_test[indicies]))
    return [val_acc for val_acc in np.mean(val_accs, 0)], [std for std in np.std(val_accs, 0)]

def train_print(i, j, loss, batch, batch_total, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.4} |".format(loss),
          "Data: {}/{} |".format(batch, batch_total), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")
    
def val_print(i, j, loss, acc, time):
    '''
    Formats print statements to update on same print line.
    
    @params are integers or floats
    '''
    print("Epoch {:1} |".format(i), 
          "Iter {:1} |".format(j), 
          "Loss: {:.2} |".format(loss),
          "Acc: {} |".format(np.round(acc,3)), 
          "Time {:1.2} ".format(time), 
          "   ", end="\r")

def crop_data(img, segmentation, crop_max):
    ret_img = img.copy()
    ret_segmentation = segmentation.copy()
    
    if crop_max:
        x_min = random.randint(0,crop_max)
        x_max = 388 - random.randint(0,crop_max)
        y_min = random.randint(0,crop_max)
        y_max = 388 - random.randint(0,crop_max)
        crop = ret_img[92:480,92:480,0]
        crop = imresize(crop[x_min:x_max, y_min:y_max],(388,388))
        # for i in range(9):
        #     crop[:,:,i] = imresize(crop[x_min:x_max, y_min:y_max,i],(388,388))
        ret_img[92:480,92:480,0] = crop
        
        # ret_segmentation[:,:,0] = imresize(ret_segmentation[x_min:x_max, y_min:y_max], (388,388))
        for i in range(segmentation.shape[2]):
            ret_segmentation[:,:,i] = imresize(ret_segmentation[:,:,i][x_min:x_max, y_min:y_max], (388,388))
    return ret_img, ret_segmentation


def blackout_data(img, segmentation, blackout_max):
    img = img.copy()
    if blackout_max:
        segmentation = segmentation.copy()
        edges = gaussian_filter(sobel(segmentation.astype('uint8')),0.3)
        #edges = gaussian_filter(cv2.Canny(segmentation.astype('uint8'),0, 1),0.3)
        border = np.where(edges > 0)
        point = random.randint(0,len(border[0])-1)
        rr, cc = circle(border[0][point]+92, border[1][point]+92,random.randint(1,blackout_max))
        img[rr,cc] = random.randint(0,20)
    return img

def data_augmentation(x_train, y_train, blackout_max, crop_max):
    x_train_copy = x_train.copy()
    y_train_copy = y_train.copy()
    seg_num = y_train.shape[3]
    for i in range(x_train.shape[0]):
        x_train_copy[i] = blackout_data(x_train_copy[i], y_train_copy[i,:,:,random.randint(1,seg_num-1)], blackout_max)
        x_train_copy[i], y_train_copy[i] = crop_data(x_train_copy[i], y_train_copy[i], crop_max)
    return x_train_copy, y_train_copy
    
def train(sess, model, x_train, y_train, x_test, y_test, epochs, batch_size, summary_writer = 0, train_validation = 5, blackout_max = 0, crop_max = 0, start_step = 0):
    '''
    Main function for training neural network model. 
    
    @params many identical to those in validate()
    @params summary_writer: Tf.summary.FileWriter used for Tensorboard variables
    @params batch_size: Integer defining mini-batch size
    @params train_validation: Integer defining how many train steps before running accuracy on training mini-batch
    '''
    losses = deque([])
    train_accs = deque([])
    step = start_step
    for i in range(epochs):
        # Shuffle indicies
        indicies = range(x_train.shape[0])
        np.random.shuffle(indicies)
        # Start timer
        start = timeit.default_timer()

        for j in range(int(x_train.shape[0]/batch_size)):
            # Shuffle Data
            temp_indicies = indicies[j*batch_size:(j+1)*batch_size]
            x_train_temp, y_train_temp = data_augmentation(x_train[temp_indicies], y_train[temp_indicies], blackout_max, crop_max)
            loss, loss_summary = model.fit_batch(sess,x_train_temp, y_train_temp)
            if summary_writer:
                summary_writer.add_summary(loss_summary, step)
            if len(losses) == 20:
                losses.popleft()
            losses.append(loss)
            # How often to test accuracy on training batch
            stop = timeit.default_timer()
            
            train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
            step = step + 1

        # Tail case 
        if x_train.shape[0] % batch_size != 0:
            temp_indicies = indicies[(j+1)*batch_size:]
            x_train_temp, y_train_temp = data_augmentation(x_train[temp_indicies], y_train[temp_indicies], blackout_max, crop_max)
            loss, loss_summary = model.fit_batch(sess,x_train_temp, y_train_temp)
            if summary_writer:
                summary_writer.add_summary(loss_summary, step)
            if len(losses) == 20:
                losses.popleft()
            losses.append(loss)
            stop = timeit.default_timer()
            train_print(i, j, np.mean(losses), j*batch_size, x_train.shape[0], stop - start)
            step = step + 1
        stop = timeit.default_timer()
        acc = validate(sess, model, x_test, y_test)
        summary = tf.Summary()
        for k in range(len(acc)):
            summary.value.add(tag="validation_acc_" + str(k), simple_value=acc[k])
        if summary_writer:    
            summary_writer.add_summary(summary, step)
        val_print(i, j, np.mean(losses), acc, stop - start)
        print()
        
