#! /home/rdeo/anaconda/bin/python2.7
from __future__ import division, print_function, absolute_import
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import cv2
import subprocess
from subprocess import Popen, PIPE
import time
import dicom
import os
import numpy as np
import scipy as sp
import trackpy as tp
import pims
import itertools
import pandas as pd
from pandas import rolling_median
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import convex_hull_image
import scipy.fftpack as fft
from ggplot import *
from scipy import signal
import statsmodels.formula.api as sm
import shutil
import random
from echoanalysis_tools import *
from process_gls_fcn import process_gls
from strain_strainrate_unet import outputstrain
from output_cropped_forstrain import outputcropped
import gc

def main():
    viewfile = "view_22_e7_class_vgg_dicomsample_probabilities.txt"
    dicomdir = "dicomsample"
    viewlist_a2c = []
    viewlist_a4c = []
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
    print(viewlist_a2c, viewlist_a4c, viewlist_psax)
    measuredict = {}
    lvlengthlist = []
    for videofile in viewlist_a4c + viewlist_a2c:
        measuredict[videofile] = {}
        bsa, ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(dicomdir, videofile)
        window =  int(((60 / hr) / (ft / 1000))) #approximate number of frames per cardiac cycle
        outputcropped(dicomdir, videofile, x_scale, y_scale, nrow, ncol) 
        time.sleep(20)
        outputstrain(dicomdir, videofile, 0.85, ft, nrow, ncol, window, x_scale, y_scale)
        gls, L, N, badframepct_L, badframepct_R = process_gls(dicomdir, videofile, window)
        measuredict[videofile]['gls'] = gls
        measuredict[videofile]['particles'] = L
        measuredict[videofile]['measurements'] = N
        measuredict[videofile]['badframepct_L'] = badframepct_L
        measuredict[videofile]['badframepct_R'] = badframepct_R
    out = open(dicomdir + "_strain_dict.txt", 'w')
    pickle.dump(measuredict, out)
    out.close()

if __name__ == '__main__':
    main()

