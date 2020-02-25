from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
from ggplot import *
from echoanalysis_tools import *


def computeframetime_gdcm(data):
    counter = 0
    for i in data:
        if i.split(" ")[0] == '(0018,1063)':
            frametime = i.split(" ")[2][1:-1]
            counter = 1
        elif i.split(" ")[0] == '(0018,0040)':
            framerate = i.split("[")[1].split(" ")[0][:-1]
            frametime = str(1000 / eval(framerate))
            counter = 1
        elif i.split(" ")[0] == '(7fdf,1074)':
            framerate = i.split(" ")[3]
            frametime = str(1000 / eval(framerate))
            counter = 1
        elif i.split(" ")[0] == '(0018,1065)': #frame time vector
            framevec = i.split(" ")[2][1:-1].split("\\")
            frametime = framevec[10] #arbitrary frame
            counter = 1
    if counter == 1:
        ft = frametime
        return ft
    else:
        return None

def smooth3(y):
    box_pts = 3
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotstrain(fl, fr, ft, hr):
    mpl.rc('figure', figsize=(12, 8))
    mpl.rc('image', cmap='gray')
    mpl.use('pdf')
    if not (fr == "None" or fl == "None"):
        sr_l = fl['strainrate']
        sr_r = fr['strainrate']
        sr = np.hstack((sr_r, sr_l))
    elif fr == "None":
        sr = fl['strainrate']
    elif fl == "None":
        sr = fr['strainrate']
    driftcorrect = True
    window = int((60 / hr) / (ft / 1000))
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

def process_gls(videofile, ft, hr, fl, fr):
    weightlist = []
    glslist = []
    nonelist = ['nan', 'None', None]
    if not (fr == "None" or fl == "None"):
        if fr.has_key('strainrate') and fl.has_key('strainrate'):
            weight = np.mean((fl['L_size'], fr['L_size']))
            if fr['strainrate'].shape[0] == fl['strainrate'].shape[0]:
                gls, N = plotstrain(fl, fr, ft, hr)
                badframepct_L, badframepct_R = fl['badframepct'], fr['badframepct']
            else:
                return "NA", "NA", "NA", "NA","NA" 
    elif not fl == "None":
        if fl.has_key('strainrate'):
            weight = fl['L_size']
            gls, N = plotstrain(fl, fr, ft, hr)
            badframepct_L, badframepct_R = fl['badframepct'], "NA"
        else:
            return "NA", "NA", "NA", "NA","NA" 
    elif not fr == "None":
        if fr.has_key('strainrate'):
            weight = fr['L_size']
            gls, N = plotstrain(fl, fr, ft, hr)
            badframepct_L, badframepct_R = "None", fr['badframepct']
        else:
            return "NA", "NA", "NA", "NA","NA" 
    else:
        return "NA", "NA", "NA", "NA","NA" 
    if not gls in nonelist:
        return gls, weight, N, badframepct_L, badframepct_R
    else:
        return "NA", "NA", "NA", "NA", "NA"
