from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import sys
import itertools
sys.path.append('./funcs/')
sys.path.append('./nets/')
import nn_cropping_black as nn
from echoanalysis_tools import *
import cv2
import pickle
import shutil
from optparse import OptionParser
from scipy.misc import imread, imresize
import pandas as pd
import scipy.fftpack as fft
from scipy import signal
import statsmodels.formula.api as sm
from pandas import rolling_median
import disease_vgg as network
import tensorflow as tf

def classifydisease(dicomdir, videofile, index1, index2, feature_dim, label_dim, model_name):
    """
    outputs a probability for a specific disease
    """
    targetnetwork = network
    imagepair = []
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    frame_length = len(framedict.keys())
    mask = create_mask([framedict[i] for i in range(int(frame_length))])
    imagepair.append(imresize(mask*framedict[index1],(224,224)).astype('uint8'))
    imagepair.append(imresize(mask*framedict[index2],(224,224)).astype('uint8'))
    imagepair = np.array(imagepair)
    if len(imagepair.shape) == 4: 
            imagepair = imagepair[:,:,:,1]
    imagepair = imagepair.transpose((1,2,0))
    imagearray = np.array([imagepair])
    tf.reset_default_graph()
    sess = tf.Session()
    model = targetnetwork.VGG(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)
    probout = model.probabilities(sess, imagearray)
    diseaseprob = np.around(probout[0][1], decimals=3)
    return diseaseprob

def phase_areas(lv_seg):
    '''
    determines approximate end-systole and end-diastole frame indices
    '''
    lv_segs = remove_periphery(lv_seg)
    lv_areas = extract_areas(lv_segs)
    lv_areas = rolling_median(pd.DataFrame(lv_areas)[0], window=3, center=True).fillna(method='bfill').fillna(method='ffill').tolist() 
    x, y = smooth_fft(lv_areas, 2500)
    frame10 = np.argsort(y)[np.int(0.10*len(y))]
    frame90 = np.argsort(y)[np.int(0.90*len(y))]
    return frame10, frame90

def extract_areas(segs):
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas

def smooth_fft(displist, cutoff):
    x = np.arange(len(displist))
    N = len(displist)
    y = np.array(displist)

    w = fft.rfft(y)
    f = fft.rfftfreq(N, x[1] - x[0])
    spectrum = w ** 2

    cutoff_idx = spectrum < (spectrum.max() / cutoff)
    w2 = w.copy()
    w2[cutoff_idx] = 0

    y2 = fft.irfft(w2)
    return x, y2

def main():
    viewfile = "view_22_e7_class_vgg_dicomsample_probabilities.txt"
    dicomdir = "dicomsample"
    viewlist_a4c = []
    viewlist_plax = []
    model_dir = "/media/deoraid03/disease_models/"
    
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
        if eval(i[viewdict['a4c']]) > probthresh:
            viewlist_a4c.append(filename)
        elif eval(i[viewdict['plax_plax']]) > probthresh:
            viewlist_plax.append(filename)

    diseasedict = {}
    diseaselist = ['hcm', 'amyloid']
    viewlist = ['a4c', 'plax']
    for disease in diseaselist:
       diseasedict[disease] = {}
       for view in viewlist:
           diseasedict[disease][view] = []
    npydir = dicomdir + "/unet/"
    for video in viewlist_plax: 
       lv_seg = np.load(npydir + "/" + video + "_lv.npy")
       index1, index2 = phase_areas(lv_seg)
       for disease in diseaselist:
           model_list = ['2a', '2b', '2c', '2d']
           feature_dim = 2
           label_dim = 2
           for model in model_list:
               model_name = model_dir + disease + "/plax/" + model
               prob = classifydisease(dicomdir, video, index1, index2, feature_dim, label_dim, model_name)
               diseasedict[disease]['plax'].append(prob)
               print(disease, prob)
    for video in viewlist_a4c: 
       lv_seg = np.load(npydir + "/" + video + "_lv.npy")
       index1, index2 = phase_areas(lv_seg)
       for disease in diseaselist:
           model_list = ['1a', '1b', '1c', '1d']
           feature_dim = 2
           label_dim = 2
           for model in model_list:
               model_name = model_dir + disease + "/a4c/" + model
               prob = classifydisease(dicomdir, video, index1, index2, feature_dim, label_dim, model_name)
               diseasedict[disease]['a4c'].append(prob)
               print(disease, prob)
    out = open("disease_probabilities_" + dicomdir + ".txt", 'w')
    pickle.dump(diseasedict, out)
    out.close()
    tempdir = os.path.join(dicomdir, "image")
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)


if __name__ == '__main__':
    main()

