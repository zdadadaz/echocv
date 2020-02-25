# coding: utf-8
import sys
import nn_cropping_black as nn
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import dicom
from echoanalysis_tools import output_imgdict, create_imgdict_from_dicom
from scipy.misc import imread, imsave, imresize
import subprocess
import dicom
from subprocess import Popen, PIPE


class Unet(object):
    def __init__(self, mean, weight_decay, learning_rate, label_dim = 8, maxout = False):
        self.x_train = tf.placeholder(tf.float32, [None, 572, 572, 1])
        self.y_train = tf.placeholder(tf.float32, [None, 388,388, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 572, 572, 1])
        self.y_test = tf.placeholder(tf.float32, [None, 388,388, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob = 1.0, reuse = True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
    
    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob = 0.8, reuse = None, maxout = False):
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: nn.max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding = 'VALID', relu = True, filter_size = 3: nn.conv(x, filter_size, output_depth, 1, self.weight_decay,
                                                                                                           name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: nn.deconv(x, 2, output_depth, 2, self.weight_decay, name=name)
            fc_ = lambda x, features, name, relu = True: nn.fc(x, features, self.weight_decay, name, relu)
            
            conv_1_1 = conv_(input, 64, 'conv1_1')
            conv_1_2 = conv_(conv_1_1, 64, 'conv1_2')
            
            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, 128, 'conv2_1')
            conv_2_2 = conv_(conv_2_1, 128, 'conv2_2')
            
            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, 256, 'conv3_1')
            conv_3_2 = conv_(conv_3_1, 256, 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, 512, 'conv4_1')
            conv_4_2 = conv_(conv_4_1, 512, 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = tf.nn.dropout(conv_(pool_4, 1024, 'conv5_1'), keep_prob)
            conv_5_2 = tf.nn.dropout(conv_(conv_5_1, 1024, 'conv5_2'), keep_prob)

            up_6 = tf.concat([deconv_(conv_5_2, 512, 'up6'), conv_4_2[:,4:60,4:60,:]], 3)
            
            conv_6_1 = conv_(up_6, 512, 'conv6_1')
            conv_6_2 = conv_(conv_6_1, 512, 'conv6_2')
           
            up_7 = tf.concat([deconv_(conv_6_2, 256, 'up7'), conv_3_2[:,16:120,16:120,:]], 3)
            
            conv_7_1 = conv_(up_7, 256, 'conv7_1')
            conv_7_2 = conv_(conv_7_1, 256, 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, 128, 'up8'), conv_2_2[:,40:240,40:240,:]], 3)
            
            conv_8_1 = conv_(up_8, 128, 'conv8_1')
            conv_8_2 = conv_(conv_8_1, 128, 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, 64, 'up9'), conv_1_2[:,88:480,88:480,:]], 3)
            
            conv_9_1 = conv_(up_9, 64, 'conv9_1')
            conv_9_2 = conv_(conv_9_1, 64, 'conv9_2')
            
            conv_10 = conv_(conv_9_2, self.label_dim, 'conv10_2', filter_size = 1, relu = False)
            return conv_10

def extract_images(framedict):
    images = []
    orig_images = []
    for key in framedict.keys():
        image = np.zeros((572,572))
        image[92:480,92:480] = imresize(framedict[key], (388,388,1))
        images.append(image)
        orig_images.append(framedict[key])
    images = np.array(images).reshape((len(images), 572,572,1))
    return images, orig_images

def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output

def extract_segs_a2c_a4c(dicomdir, videofile, model, sess, lv_label, la_label, lvo_label):
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    images, orig_images = extract_images(framedict)
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = range(1, 8)
    label_good = [lv_label, la_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
    lv_segs = []
    lvo_segs = []
    la_segs = []
    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, orig_images, lvo_segs, preds


def extract_segs_psax(dicomdir, videofile, model, sess, lvo_label, lvi_label):
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    images, orig_images = extract_images(framedict)
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = range(1, 3)
    label_good = [lvi_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
    lvo_segs = []
    lvi_segs = []
    for seg in segs:
        lvo_seg = create_seg(seg, lvo_label)
        lvi_seg = create_seg(seg, lvi_label)
        lvo_segs.append(lvo_seg)
        lvi_segs.append(lvi_seg)
    return lvo_segs, lvi_segs, orig_images, preds

def extract_segs_plax(dicomdir, videofile, model, sess, lv_label, la_label):
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    images, orig_images = extract_images(framedict)
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = range(1, 6)
    label_good = [lv_label, la_label]
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
    lv_segs = []
    la_segs = []
    for seg in segs:
        lv_seg = create_seg(seg, lv_label)
        la_seg = create_seg(seg, la_label)
        lv_segs.append(lv_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, orig_images, preds

def segmenta2c(video, dicomdir, model, sess):
    thumbdir = dicomdir + "/thumbnails/"
    if not os.path.exists(thumbdir):
        os.makedirs(thumbdir)
    outpath = dicomdir + "/unet/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    a2c_lv_segs, a2c_la_segs, a2c_images, a2c_lvo_segs, preds = extract_segs_a2c_a4c(dicomdir, video, model, sess, 3, 4, 2)
    np.save(outpath + '/' + video + '_lvo', np.array(a2c_lvo_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_lv', np.array(a2c_lv_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_la', np.array(a2c_la_segs).astype('uint8'))
    plt.figure(figsize = (2.5,2.5))
    plt.axis('off')
    plt.imshow(preds)
    plt.savefig(thumbdir + '/' + video + '_segmentation.jpg')
    plt.close()
    return 1

def segmenta4c(video, dicomdir, model, sess):
    thumbdir = dicomdir + "/thumbnails/"
    if not os.path.exists(thumbdir):
        os.makedirs(thumbdir)
    outpath = dicomdir + "/unet/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    a4c_lv_segs, a4c_la_segs, a4c_images, a4c_lvo_segs, preds = extract_segs_a2c_a4c(dicomdir, video, model, sess, 3, 5, 2)
    np.save(outpath + '/' + video + '_lvo', np.array(a4c_lvo_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_lv', np.array(a4c_lv_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_la', np.array(a4c_la_segs).astype('uint8'))
    plt.figure(figsize = (2.5, 2.5))
    plt.axis('off')
    plt.imshow(preds)
    plt.savefig(thumbdir + '/' + video + '_segmentation.jpg')
    plt.close()
    return 1

def segmentpsax(video, dicomdir, model, sess):
    thumbdir = dicomdir + "/thumbnails/"
    if not os.path.exists(thumbdir):
        os.makedirs(thumbdir)
    outpath = dicomdir + "/unet/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    psax_outer_segs, psax_inner_segs, psax_images, preds = \
    extract_segs_psax(dicomdir, video, model, sess, 1, 2)
    np.save(outpath + '/' + video + '_psaxo', np.array(psax_outer_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_psaxi', np.array(psax_inner_segs).astype('uint8'))
    plt.figure(figsize = (2.5,2.5))
    plt.axis('off')
    plt.imshow(preds)
    plt.savefig(thumbdir + '/' + video + '_segmentation.jpg')
    plt.close()
    return 1

def segmentplax(video, dicomdir, model, sess):
    thumbdir = dicomdir + "/thumbnails/"
    if not os.path.exists(thumbdir):
        os.makedirs(thumbdir)
    outpath = dicomdir + "/unet/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    plax_lv_segs, plax_la_segs, plax_images, preds = extract_segs_plax(dicomdir, video, model, sess, 1, 5)
    np.save(outpath + '/' + video + '_lv', np.array(plax_lv_segs).astype('uint8'))
    np.save(outpath + '/' + video + '_la', np.array(plax_la_segs).astype('uint8'))
    plt.figure(figsize = (2.5,2.5))
    plt.axis('off')
    plt.imshow(preds)
    plt.savefig(thumbdir + '/' + video + '_plax_segmentation.jpg')
    plt.close()
    return 1


