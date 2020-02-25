from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
from pandas import rolling_median
from scipy.misc import imread, imsave, imresize
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import convex_hull_image
import scipy.fftpack as fft
from ggplot import *
from scipy import signal
import statsmodels.formula.api as sm

def smooth3(y):
    box_pts = 3
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotstrain(fl, fr, window):
    mpl.rc('figure', figsize=(12, 8))
    mpl.rc('image', cmap='gray')
    mpl.use('pdf')
    sr_r = fr['strainrate']
    sr_l = fl['strainrate']
    sr = np.hstack((sr_r, sr_l))
    smoothlist = []
    x = sr[:,0]
    if window > len(x):
        window = len(x)-1
    for start in range(0, len(x) - window, int(window/2)):
        end = np.min((start + window, len(x) - 1))
        if (end - start) > 0.8*window:
            a = sr[start:end, :]
            b = np.apply_along_axis(smooth3, 0, a)
            c = np.apply_along_axis(np.cumsum, 0, b)
            d = c - np.apply_along_axis(np.max, 0, c)
            d[(d < -0.30) | (d > -0.05)] = np.nan
            e = np.apply_along_axis(np.nanmean, 1, d)
            f = -np.nanmin(smooth3(e)) 
            smoothlist.append(f)
    gls = np.nanpercentile(smoothlist, 50)
    return gls, len(smoothlist)

def process_gls(dicomdir, videofile, window):
    weightlist = []
    glslist = []
    weightcount = 0
    weightlist = []
    glslist_L_R = []
    glslist_L_R_unique = []
    pkldir = dicomdir + "/"
    allfiles = os.listdir(pkldir)
    counter = 0
    for pklfile in allfiles:
        if videofile in pklfile:
            counter = 1
            if "strain" in pklfile:
                fileprefix = pklfile.split("_left")[0].split("_right")[0]
                leftfile = pkldir + "/" + fileprefix + "_left.pkl"
                ritefile = pkldir + "/" + fileprefix + "_right.pkl"
                if (os.path.exists(leftfile) and os.path.exists(ritefile)):
                    fl = open(leftfile)
                    fl = pickle.load(fl)
                    fr = open(ritefile)
                    fr = pickle.load(fr)
                    nonelist = ['nan', 'None']
                    Lweight = np.mean((fl['L_size'], fr['L_size']))
                    badframepct_L = fl['badframepct']
                    badframepct_R = fr['badframepct']
                    if fl.has_key('strainrate') and fr.has_key('strainrate'):
                        #print(fl['strainrate'].shape, fr['strainrate'].shape)
                        if fl['strainrate'].shape[0] == \
                        fr['strainrate'].shape[0]:
                            gls, N = plotstrain(fl, fr, window)
                            print("gls ", gls)
                            if not gls in nonelist:
                                return gls, Lweight, N, badframepct_L, badframepct_R
                            else:
                                return "NA", "NA", "NA", "NA","NA"
                        else:
                            return "NA", "NA", "NA", "NA","NA"
                    else:
                        return "NA", "NA", "NA", "NA","NA"
                else:
                    return "NA", "NA", "NA", "NA","NA"
    if counter == 0:
        return "NA", "NA", "NA", "NA","NA"
