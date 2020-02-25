from __future__ import division, print_function, absolute_import
import sys
import time
import dicom
import os
sys.path.append('./funcs/')
sys.path.append('./nets/')
import numpy as np
import scipy as sp
import itertools
from scipy.misc import imread, imsave, imresize
import shutil
import random
import tensorflow as tf
from echoanalysis_tools import *
import vgg as network
from unet_a4c_a2c_psax_plax import segmenta2c, segmenta4c, Unet, segmentpsax, \
segmentplax

def segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir):
    modeldir = "./models/"
    mean = 24
    weight_decay = 1e-6
    learning_rate = 1e-4
    maxout = False
    g_1 = tf.Graph()
    g_2 = tf.Graph()
    g_3 = tf.Graph()
    g_4 = tf.Graph()
    with g_1.as_default():
        label_dim = 7
        sess1 = tf.Session()
        model1 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
        sess1.run(tf.local_variables_initializer())
    with g_2.as_default():
        label_dim = 5
        sess2 = tf.Session()
        model2 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
        sess2.run(tf.global_variables_initializer())
    with g_3.as_default():
        label_dim = 4
        sess3 = tf.Session()
        model3 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
        sess3.run(tf.global_variables_initializer())
    with g_4.as_default():
        label_dim = 7
        sess4 = tf.Session()
        model4 = Unet(mean, weight_decay, learning_rate, label_dim , maxout = maxout)
        sess4.run(tf.global_variables_initializer())
    with g_1.as_default():
        saver = tf.train.Saver()
        saver.restore(sess1, modeldir + "a4c_v1")
    with g_2.as_default():
        saver = tf.train.Saver()
        saver.restore(sess2, modeldir + "a2c_v1")
    with g_3.as_default():
        saver = tf.train.Saver()
        saver.restore(sess3, modeldir + "psax_v1")
    with g_4.as_default():
        saver = tf.train.Saver()
        saver.restore(sess4, modeldir + "plax_v1")
    for video in viewlist_a4c:
        segmenta4c(video, dicomdir,  model1, sess1)
    for video in viewlist_a2c:
        segmenta2c(video, dicomdir,  model2, sess2)
    for video in viewlist_psax:
        segmentpsax(video, dicomdir, model3, sess3)
    for video in viewlist_plax:
        segmentplax(video, dicomdir, model4, sess4)
    return 1


def main():
    viewfile = "view_22_e7_class_vgg_dicomsample_probabilities.txt"
    dicomdir = "dicomsample"
    viewlist_a2c = []
    viewlist_a4c = []
    viewlist_plax = []
    viewlist_psax = []
    
    infile = open("viewclasses_view_22_e7_class_vgg.txt")
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]

    viewdict = {}

    for i in range(len(infile)):
        viewdict[infile[i]] = i + 2
     
    probthresh = 0.5 #arbitrary choice of "probability" threshold for view classification

    infile = open(viewfile)
    infile = infile.readlines()
    infile = [i.rstrip() for i in infile]
    infile = [i.split('\t') for i in infile]

    for i in infile[1:]:
        dicomdir = i[0]
        filename = i[1]
        if eval(i[viewdict['psax_pap']]) > probthresh:
            viewlist_psax.append(filename)
        elif eval(i[viewdict['a4c']]) > probthresh:
            viewlist_a4c.append(filename)
        elif eval(i[viewdict['a2c']]) > probthresh:
            viewlist_a2c.append(filename)
        elif eval(i[viewdict['plax_plax']]) > probthresh:
            viewlist_plax.append(filename)
    print(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax)
    segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax, dicomdir)
    tempdir = os.path.join(dicomdir, "image")
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

if __name__ == '__main__':
    main()
