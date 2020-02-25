# coding: utf-8
from __future__ import division
import moments as mom
from skimage.morphology import convex_hull_image
import trackpy as tp
import pims
import cv2
import pickle
from output_cropped_forstrain_v2 import outputcropped
from optparse import OptionParser
from process_gls_fcn_v2 import process_gls
from multiprocessing import Pool
import shutil
from echoanalysis_tools import *

parser=OptionParser()

def computemoments_image(image):
    chull = convex_hull_image(image)
    image[chull] += 3
    xbar, ybar, cov = mom.inertial_axis(image)
    params = extractparameters(xbar, ybar, cov)   
    return params


def computemoments(points, nrow, ncol):
    image = np.zeros((nrow, ncol))
    for i in points:
        if i[1] < nrow and i[0] < ncol:
            image[np.int(i[1]), np.int(i[0])] = 3.0 #first coordinate is row
    chull = convex_hull_image(image)
    image[chull] += 3
    xbar, ybar, cov = mom.inertial_axis(image)
    params = extractparameters(xbar, ybar, cov)
    return params

def extractparameters(x_bar, y_bar, cov):
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    out1 = make_lines(eigvals, eigvecs, mean, -1)
    height = (out1[0][0] - out1[0][2], out1[1][0] - out1[1][2])
    center = (out1[0][1], out1[1][1])
    out2 = make_lines(eigvals, eigvecs, mean, 0)
    width = (out2[0][0] - out2[0][2], out2[1][0] - out2[1][2])
    theta = angle_between((1, 0), height)
    return np.linalg.norm(height), np.linalg.norm(width), theta, center, out1, out2

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def computeoverlap_resize(frames, x, y, frameindex1, frameindex2, windowx, windowy, method, scale):
    '''
    Computer location of max overlap - remember output is in terms of dy, dx
    If it moves upwards (i.e. systole) dY will be negative
    If it moves inwards (i.e. systole) dX will be positive
    
    :param x: 
    :param y: 
    :param frameindex1: 
    :param frameindex2: 
    :param windowx: 
    :param windowy: 
    :param method: 
    :param scale: 
    :return: 
    '''
    data1 = np.array(frames[frameindex1])
    data2= np.array(frames[frameindex2])
    a = 2
    b = a+1
    y = np.int(round(y, 0))
    x = np.int(round(x, 0))
    yspace1 = np.int(round(a*windowy, 0))
    xspace1 = np.int(round(a*windowx, 0))
    yspace2 = np.int(round(b*windowy, 0))
    xspace2 = np.int(round(b*windowx, 0))
    ydiff = yspace2 - yspace1
    xdiff = xspace2 - xspace1
    template = data1[(y - yspace1):(y + yspace1), (x - xspace1):(x + xspace1)]
    template = cv2.resize(template, (0,0), fx = scale, fy = scale)
    w, h = template.shape[::-1]
    img = data2[(y - yspace2):(y + yspace2), (x - xspace2):(x + xspace2)]
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)
    method = eval(method) 
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        score = min_val
    else:
        top_left = max_loc
        score = max_val
    movey = (top_left[1] - scale*ydiff)/scale
    movex = (top_left[0] - scale*xdiff)/scale
    return movey, movex, score

def compute_L(point, center, theta):
    '''
    note that for x and y coordinates, 0,0 is top left of image; positive y difference is towards apex
    :param coordinates of data
    :param center: 
    :param theta: angle of deviation from vertical
    :return: projection of norm along the direction of the wall
    '''
    center = np.array(center)
    center[1] = -center[1]
    point = np.array(point)
    point[1] = -point[1]
    angle = angle_between((1,0), center - point)
    return np.linalg.norm(center - point)*np.cos((theta - angle)*(np.pi/180))

def compute_dLdT(dx, dy, theta):
    point = np.array((dx, dy)) 
    angle = angle_between((1,0), point)
    dL =  np.linalg.norm(point)*np.cos(np.deg2rad(theta - angle))
    dT =  np.linalg.norm(point)*np.cos(np.deg2rad(theta - angle - 90))
    return dL, dT

def computestrain(frames, datamat, partno, dxmax, dymax, center, theta, frame): 
    x = datamat.iloc[partno]['x']
    y = datamat.iloc[partno]['y']
    L = compute_L((x,y), center, theta)
    if L > 10*dxmax:
        mod = 0.4
    elif L > -3*dxmax:
        mod = 0.7
    else:
        mod = 1
    windowx = int(round(mod*dxmax))
    windowy = int(round(mod*dymax))
    dy, dx, score = computeoverlap_resize(frames, x, y, frame, frame+1, windowx, windowy, 'cv2.TM_CCOEFF_NORMED', 4)
    dL, dT = compute_dLdT(dx, dy, theta)
    return L, dL, dT, score

def computefit(Ldat, dLdat, dTdat, weights):
    z = np.polyfit(Ldat, dLdat, 3) 
    w = np.polyfit(Ldat, dTdat, 1)
    a1, b1, c1, d1 = z[3], z[2], z[1], z[0] 
    a2, b2 = w[1], w[0]
    p1 = np.poly1d(z)
    p2 = np.polyder(p1)
    return a1, b1, c1, d1, a2, b2, z, p1, p2

def updateparams(center, theta, a1, a2, b2):
    xnew = center[0] + a1*np.cos(np.deg2rad(theta) - a2*np.sin(np.deg2rad(theta)))
    ynew = center[1] + a2*np.cos(np.deg2rad(theta)) + a1*np.sin(np.deg2rad(theta))
    thetanew = theta + np.arctan(b2)
    return xnew, ynew, thetanew

def findpart(frames, frameno, minthresh, ns, verbose, nrow, ncol):
    dtarget = 3
    ns = 1
    sep = dtarget  #smaller value leads to fewer points
    f = tp.locate(frames[frameno], noise_size = ns, separation = sep, smoothing_size = \
                  dtarget + 1, diameter = dtarget, minmass= minthresh , topn = \
                  100, invert=False)
    if not f is None:
        ymin, ymax = np.min(f['y']), np.max(f['y'])
        fsep = f
        nopart = fsep.shape[0]
        if not nopart == 0:
            part_data = np.array((fsep['x'].values, fsep['y'].values)).T
            center= computemoments(part_data, nrow, ncol)[3]
            return fsep, center, ymin, ymax
        else:
            return None, None, None, None
    else:
        return None, None, None, None

def testpart(frames, frameno, minthresh, ns, verbose):
    dtarget = 3
    ns = 1
    sep = dtarget  #smaller value leads to fewer points
    f = tp.locate(frames[frameno], noise_size = ns, separation = sep, smoothing_size = \
                  dtarget + 1, diameter = dtarget, minmass= minthresh , topn = 100, invert=False)
    nopart = f.shape[0]
    return nopart

def processframe(frames, frameno, fsep, center, theta, scorethresh, dxmax,
                 dymax):
    framedict = {}
    framedict['L'] = []
    framedict['dT'] = []
    framedict['dL'] = []
    framedict['score'] = []
    xthresh = np.median(fsep['x'])
    for j in range(0, fsep.shape[0]):
        L, dL, dT, score = computestrain(frames, fsep, j, dxmax, dymax, center, theta, frameno)
        if score > scorethresh:
            framedict['L'].append(L)
            framedict['dT'].append(dT)
            framedict['dL'].append(dL)
            framedict['score'].append(score)
    return framedict

def computepartno(minmass, frames, ns):
    partnolist = []
    for choice in range(0, 30, 10): 
        partno =  testpart(frames, choice, minmass, ns, False)
        partnolist.append(partno)
    return np.median(partnolist)

def outputStrain_window(frames, framelo, framehi, minmass, scorethresh,
                        dxmax, dymax, tmpDir, videoFile, nrow,
                        ncol, direction):
    framealldict = {}
    success = 0
    badcount = 0
    windowsize = framehi - framelo + 1
    for choice in range(framelo, framehi):
        partlist = []
        params = computemoments_image(frames[framelo])
        theta = params[2]
        center = params[3]
        #print("theta", direction, theta)
        ns = 1
        fsep, centerpart, ylow, yhi = findpart(frames, choice, minmass, ns,
                                               False, nrow, ncol)
        if fsep is None:
            break
        if choice == framelo:
            yupper = -ylow + center[1] #apex
            ylower = -yhi + center[1] #base
        partlist.append(fsep.shape[0])
        framealldict[choice] = processframe(frames, choice, fsep, center, theta,  \
                                            scorethresh, dxmax, dymax)
        scorethreshnew = scorethresh
        L_list = []
        L = framealldict[choice]['L']
        Lorig = L
        L_list.append(len(L))
        if len(L) >= 10:
            success = 1
            framealldict[choice] = processframe(frames, choice, fsep, center, theta,  \
                                            scorethreshnew, dxmax, dymax)
            dL = framealldict[choice]['dL']
            dT = framealldict[choice]['dT']
            w = framealldict[choice]['score']
            a1, b1, c1, d1, a2, b2, z, p1, p2 = computefit(L, dL, dT, w)
        else:
            badcount = badcount + 1
            if choice == framelo:
                break
        if choice == framelo:
            sr = p2(np.arange(ylower, yupper, 10))
            vel = p1(np.arange(ylower, yupper, 10))
        else:
            srnew = p2(np.arange(ylower, yupper, 10))
            velnew = p1(np.arange(ylower, yupper, 10))
            sr = np.vstack((sr, srnew))
            vel = np.vstack((vel, velnew))
        xnew, ynew, thetanew = updateparams(center, theta, a1, a2, b2)
        center = (xnew, ynew)
        theta = thetanew
    print("bad frame percentage", badcount/windowsize, videoFile, direction)
    if success == 1:
        np.set_printoptions(precision = 3)
        alldict = {}
        alldict['strainrate'] = sr
        alldict['L_size'] = np.median(L_list)
        alldict['velocity'] = vel
        alldict['partno'] = partlist
        alldict['badframepct'] = badcount/windowsize
        outdict =  tmpDir + "/" + videoFile + "_velocity_strainrate_strain_" \
        +str(scorethresh) + "_" + str(framelo) + "_" + str(framehi) + "_" + \
        direction + ".pkl"
        out = open(outdict, 'w')
        pickle.dump(alldict,  out)
        out.close()
    return 1

def initializeStrain(view):
    measureDict = {}
    driftcorrect = True
    segmentDir = "./segment/" +  view
    if os.path.exists(segmentDir):
        allfiles = os.listdir(segmentDir)
        for fileName in allfiles:
            if "la.npy" in fileName:
                print(fileName)
                filePrefix = fileName.split("_la")[0]
                frameno = eval(filePrefix.split("Image")[1].split("-")[1].split(".dcm")[0])
                videoFile = "Image-" + str(frameno) + ".dcm"
                try:
                    output = outputcropped(videoFile, view)
                    #output = 1
                    if output == 1:
                        measureDict[filePrefix] = {}
                except (IOError, EOFError, KeyError) as e:
                    print(e)
    return measureDict

def outputStrainpool((rawDir, tmpDir, filePrefix, scorethresh)):
        frameno = filePrefix.split("Image")[1].split("-")[1].split(".dcm")[0]
        videoFile = "Image-" + frameno + ".dcm"
        print(videoFile)
        try:
            ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(rawDir, videoFile) 
            print(ft, hr, nrow, ncol, x_scale, y_scale)
            if not (ft==None or hr == None or x_scale == None or y_scale == None):
                window =  int(((60 /hr) / (ft / 1000))) #approximate number of frames per cardiac cycle
                deltax = x_scale
                deltay = y_scale
                if window > 10:
                    xvel = 8 #(cm/s)
                    yvel = 8 
                    dxmax = (ft/1000)*(xvel)/deltax
                    dymax = (ft/1000)*(yvel)/deltay
                    #print("dxmax", dxmax, "dymax", dymax)
                    for direction in ["left", "right"]:
                        videodir = tmpDir + "/" + videoFile + "/maskedimages_" + direction + "/" 
                        print(videodir)
                        frames = pims.ImageSequence(videodir + '/*.png', as_grey=True)
                        nrow = frames[0].shape[0]
                        ncol = frames[0].shape[1]
                        minmass = 130
                        partlimit = 100
                        def adjustcurrentpart(partlimit, minmass, frames):
                            currentpart = computepartno(minmass, frames, 1)
                            if currentpart < partlimit and minmass >= 80:
                                minmass = minmass - 10
                                adjustcurrentpart(partlimit, minmass, frames)
                        adjustcurrentpart(partlimit, minmass, frames)
                        end = len(frames) - 1
                        print("size of segment", len(np.where(frames[0] > 0)[0]))
                        print("comparison", 0.005*nrow*ncol)
                        if not len(np.where(frames[0] > 0)[0]) < 0.005*nrow*ncol: #heuristic filter for bad segmentation
                            framelo = 0
                            framehi = end
                            print(videoFile, direction + " side")
                            outputStrain_window(frames, framelo, framehi, minmass, 
                                            scorethresh, dxmax, dymax, tmpDir, 
                               videoFile, nrow, ncol, direction)
        except (IOError, EOFError, KeyError) as e:
                print(videoFile, e)
        return 1

def extractmetadata(dicomDir, videoFile):
    command = 'gdcmdump ' + dicomDir + "/" + videoFile
    pipe = subprocess.Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    a = computedeltaxy_gdcm(data)
    if not a[0] == None:
        x_scale, y_scale = a
    else:
        x_scale, y_scale = None, None
    hr = computehr_gdcm(data)
    b = computexy_gdcm(data)
    if not b[0] == None:
        nrow, ncol = b
    else:
        nrow, ncol = None, None
    ft = computeft_gdcm_strain(data)
    return ft, hr, nrow, ncol, x_scale, y_scale


def createglslists(rawDir, badthresh_L, badthresh_R, tmpDir, direction):
    print("creating glslists")
    donedict = {}
    glslist = []
    glslist_unique = []
    allfiles = os.listdir(tmpDir)
    pkldir = tmpDir
    for pklfile in allfiles:
        if "velocity" in pklfile:
          filePrefix = pklfile.split("_left")[0].split("_right")[0]
          print(filePrefix)
          if not filePrefix in donedict.keys():
            leftfile = pkldir + "/" + filePrefix + "_left.pkl"
            ritefile = pkldir + "/" + filePrefix + "_right.pkl"
            videoFile = pklfile.split("_velocity")[0]
            ft, hr, nrow, ncol, x_scale, y_scale = extractmetadata(rawDir, videoFile) 
            if direction == "left":
                if os.path.exists(leftfile):
                    fl = open(leftfile)
                    fl = pickle.load(fl)
                    fr = "None"
                else:
                    fl = "None"
                    fr = "None"
            elif direction == "right":
                if os.path.exists(ritefile):
                    fr = open(ritefile)
                    fr = pickle.load(fr)
                    fl = "None"
                else:
                    fr = "None"
                    fl = "None"
            elif direction == "both":
                if os.path.exists(ritefile) and os.path.exists(leftfile):
                    fr = open(ritefile)
                    fr = pickle.load(fr)
                    fl = open(leftfile)
                    fl = pickle.load(fl)
                else:
                    fr = "None"
                    fl = "None"
            gls, Lweight, N, badframepct_L,\
            badframepct_R = process_gls(videoFile,\
                               ft, hr, fl, fr)
            donedict[filePrefix] = ''
            if not gls == "NA":
                goodcount = 0
                if Lweight >= 7:
                    if direction == "both":
                        if (badframepct_L < badthresh_L and badframepct_R < \
                        badthresh_R):
                            goodcount = 1
                    if direction == "left":
                        if (badframepct_L < badthresh_L):
                            goodcount = 1
                    if direction == "right":
                        if (badframepct_R < badthresh_R):
                            goodcount = 1
                if goodcount == 1:
                    for i in range(0, N):
                        glslist.append(gls)
                    glslist_unique.append(gls)
    return glslist, glslist_unique

tteDir = "./dicomsample/"
outFile = "teststudy_strain_gls_L_R.txt"
tmpDir = "./straintmp/"

rawDir = tteDir

if os.path.exists(tmpDir):
    shutil.rmtree(tmpDir)
else:
    os.makedirs(tmpDir)

time.sleep(5)
scorethresh = 0.85
measureDicta4c = initializeStrain("a4c")
measureDicta2c = initializeStrain("a2c")
measureDict = dict(measureDicta4c.items() + measureDicta2c.items())
print(measureDict.items())
pool = Pool()                         
job_args = [(rawDir, tmpDir, fileName, scorethresh) for i, fileName in
            enumerate(measureDict.keys())]
pool.map(outputStrainpool, job_args)  
pool.close()

badthresh_L = 0.10
badthresh_R = 0.10

def iterategls(badthresh_L, badthresh_R, tmpDir, direction):
    glslist, glslist_unique  = createglslists(rawDir, badthresh_L, badthresh_R, tmpDir, direction)
    if len(glslist_unique) < 2 and badthresh_L < 0.6:
        badthresh_L = badthresh_L + 0.10
        badthresh_R = badthresh_R + 0.10
        glslist, glslist_unique, badthresh_L, badthresh_R = iterategls(badthresh_L, badthresh_R, tmpDir, direction)
    return glslist, glslist_unique, badthresh_L, badthresh_R

glslist_L, glslist_L_unique, badthresh_L_L, badthresh_L_R = iterategls(badthresh_L, badthresh_R, tmpDir, "left")
glslist_R, glslist_R_unique, badthresh_R_L, badthresh_R_R = iterategls(badthresh_L, badthresh_R, tmpDir, "right")
glslist_L_R, glslist_L_R_unique, badthresh_B_L, badthresh_B_R = iterategls(badthresh_L, badthresh_R,
                                              tmpDir, "both")
out = open(outFile, 'w')
out.write("direction\tgls_0\tgls_25\tgls_50\tgls_75\tgls_100\tnomeas\tnostudies\tbadthresh_L\tbadthrsh_R\n")
a = str(np.nanpercentile(glslist_L, 0))
b = str(np.nanpercentile(glslist_L, 25))
c = str(np.nanpercentile(glslist_L, 50))
d = str(np.nanpercentile(glslist_L, 75))
e = str(np.nanpercentile(glslist_L, 100))
f = str(np.nanstd(glslist_L_unique))
g = str(len(glslist_L))
h = str(len(glslist_L_unique))
print(a, b, c, d, e, f, g, h, str(badthresh_L_L))
out.write("left" + "\t" + a + "\t" + b + "\t" + c + "\t" + d + "\t" + e + "\t" + f + "\t" +
          g + "\t" + h + "\t" + str(badthresh_L_L) + "\t" + str(badthresh_L_R) + "\n")

a = str(np.nanpercentile(glslist_R, 0))
b = str(np.nanpercentile(glslist_R, 25))
c = str(np.nanpercentile(glslist_R, 50))
d = str(np.nanpercentile(glslist_R, 75))
e = str(np.nanpercentile(glslist_R, 100))
f = str(np.nanstd(glslist_R_unique))
g = str(len(glslist_R))
h = str(len(glslist_R_unique))
print(a, b, c, d, e, f, g, h, str(badthresh_R_R))
out.write("right" + "\t" + a + "\t" + b + "\t" + c + "\t" + d + "\t" + e + "\t" + f + "\t" +
          g + "\t" + h + "\t" + str(badthresh_R_L) + "\t" + str(badthresh_R_R) + "\n")

a = str(np.nanpercentile(glslist_L_R, 0))
b = str(np.nanpercentile(glslist_L_R, 25))
c = str(np.nanpercentile(glslist_L_R, 50))
d = str(np.nanpercentile(glslist_L_R, 75))
e = str(np.nanpercentile(glslist_L_R, 100))
f = str(np.nanstd(glslist_L_R_unique))
g = str(len(glslist_L_R))
h = str(len(glslist_L_R_unique))
print(a, b, c, d, e, f, g, h, str(badthresh_B_R))
out.write("both" + "\t" + a + "\t" + b + "\t" + c + "\t" + d + "\t" + e + "\t" + f + "\t" +
          g + "\t" + h + "\t" + str(badthresh_B_L) + "\t" + str(badthresh_B_R) + "\n")
out.close()
#shutil.rmtree(tmpDir)
