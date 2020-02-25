# coding: utf-8
from __future__ import division
import moments as mom
from echoanalysis_tools import remove_periphery
import numpy as np
from skimage.morphology import convex_hull_image
import trackpy as tp
import pims
import cv2
import pickle

def preprocess(dicomdir, videofile):
    npydir = dicomdir + "/unet/"
    lvo_segs = np.load(npydir + "/" + videofile + "_lvo.npy")
    lvo_segs = remove_periphery(lvo_segs)
    params = computemoments_image(lvo_segs[0])
    theta, center = params[2], params[3]
    lvlength, lvwidth = params[0], params[1]
    return center, lvlength, lvwidth, theta

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
    :return: projection of norm along the direction of the myocardium
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
        mod = 0.6
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
    '''
    method of updating center and angle based on polynomial fit
    not sure if this is better than computing moments on segmented data for every frame
    '''
    xnew = center[0] + a1*np.cos(np.deg2rad(theta) - a2*np.sin(np.deg2rad(theta)))
    ynew = center[1] + a2*np.cos(np.deg2rad(theta)) + a1*np.sin(np.deg2rad(theta))
    thetanew = theta + np.arctan(b2)
    return xnew, ynew, thetanew

def findpart(frames, frameno, minthresh, ns, verbose, nrow, ncol):
    '''
    using trackpy to locate particles
    '''
    dtarget = 3
    ns = 1
    sep = dtarget  #smaller value leads to fewer points
    f = tp.locate(frames[frameno], noise_size = ns, separation = sep, smoothing_size = \
                  dtarget + 1, diameter = dtarget, minmass= minthresh , topn = \
                  200, invert=False)
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
                  dtarget + 1, diameter = dtarget, minmass= minthresh , topn = 200, invert=False)
    nopart = f.shape[0]
    return nopart

def processframe(frames, frameno, fsep, center, theta, scorethresh, dxmax,
                 dymax):
    '''

    '''
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

def outputstrain_window(frames, framelo, framehi, minmass, scorethresh,
                        dxmax, dymax, dicomdir, videofile, nrow, ncol, direction):
    framealldict = {}
    success = 0
    badcount = 0
    windowsize = framehi - framelo + 1
    print(videofile)
    for choice in range(framelo, framehi):
        partlist = []
        params = computemoments_image(frames[framelo])
        theta = params[2]
        center = params[3]
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
        L_list = []
        L = framealldict[choice]['L']
        Lorig = L
        L_list.append(len(L))
        if len(L) >= 15 or choice == framelo:
            success = 1
            framealldict[choice] = processframe(frames, choice, fsep, center, theta,  \
                                            scorethresh, dxmax, dymax)
            dL = framealldict[choice]['dL']
            dT = framealldict[choice]['dT']
            w = framealldict[choice]['score']
            a1, b1, c1, d1, a2, b2, z, p1, p2 = computefit(L, dL, dT, w)
        else:
            badcount = badcount + 1
        if choice == framelo:
            sr = p2(np.arange(ylower, yupper, 10))
            vel = p1(np.arange(ylower, yupper, 10))
        else:
            srnew = p2(np.arange(ylower, yupper, 10))
            velnew = p1(np.arange(ylower, yupper, 10))
            sr = np.vstack((sr, srnew))
            vel = np.vstack((vel, velnew))
        '''
        #alternate method to update location of center and angle; compare to using image moments
        xnew, ynew, thetanew = updateparams(center, theta, a1, a2, b2)
        center = (xnew, ynew)
        theta = thetanew
        '''
    print("bad frame percentage", badcount/windowsize, videofile, direction)
    if success == 1:
        np.set_printoptions(precision = 3)
        alldict = {}
        alldict['strainrate'] = sr
        alldict['L_size'] = np.median(L_list)
        alldict['velocity'] = vel
        alldict['badframepct'] = badcount/windowsize
        alldict['partno'] = partlist
        outdir = dicomdir 
        outdict =  outdir + "/" + videofile + "_velocity_strainrate_strain_" +str(scorethresh) + "_" + str(framelo) + "_" + str(framehi) + "_" + direction + ".pkl"
        out = open(outdict, 'w')
        pickle.dump(alldict,  out)
        out.close()
    return 1

def outputstrain(dicomdir, videofile, scorethresh, ft, nrow, ncol, window, x_scale, y_scale):
    '''

    '''
    print("computing strain", dicomdir, videofile)
    center, lvlength, lvwidth, thetainit = preprocess(dicomdir, videofile)
    deltax = x_scale
    deltay = y_scale
    xvel = 8 #(cm/s) myocardial velocity
    yvel = 8
    dxmax = (ft/1000)*(xvel)/deltax #maximum movement anticipated
    dymax = (ft/1000)*(yvel)/deltay #maximum movement anticipated
    for direction in ["left", "right"]:
        videodir = dicomdir + "/maskedimages_" + \
        direction + "/" + videofile
        print("reading in data")
        frames = pims.ImageSequence(videodir + '/*.png', as_grey=True)
        print("data read")
        minmass = 130
        partlimit = 100
        def adjustcurrentpart(partlimit, minmass, frames):
            '''
            recursive method to obtain enough trackable particles
            '''
            currentpart = computepartno(minmass, frames, 1)
            if currentpart < partlimit and minmass >= 80:
                minmass = minmass - 10
                adjustcurrentpart(partlimit, minmass, frames)
        adjustcurrentpart(partlimit, minmass, frames)
        end = len(frames) - 1
        if not len(np.where(frames[0] > 0)[0]) < 0.01*nrow*ncol: #heuristic filter for bad segmentation
            framelo = 0
            framehi = end
            outputstrain_window(frames, framelo, framehi, minmass, 
                        scorethresh, dxmax, dymax, dicomdir,
                            videofile, nrow, ncol, direction)
    return 1
