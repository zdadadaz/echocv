# coding: utf-8
import sys
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from scipy.misc import imread, imsave, imresize
import cv2
import os
import dicom
import time

def computehr_gdcm(data):
    '''
    identifies heart rate for a given video
    '''
    hr = "None"
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0018,1088)':
            hr = i.split("[")[1].split("]")[0]
    return eval(hr)

def computexy_gdcm(data):
    '''
    returns number of rows and columns
    '''
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0028,0010)':
            rows = i.split(" ")[2]
        elif i.split(" ")[0] == '(0028,0011)':
            cols = i.split(" ")[2]
    return int(rows), int(cols)

def computebsa_gdcm(data):
    '''
    dubois, height in m, weight in kg
    :param data: 
    :return: 
    '''
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0010,1020)':
            h = i.split("[")[1].split("]")[0]
        elif i.split(" ")[0] == '(0010,1030)':
            w = i.split("[")[1].split("]")[0]
    return 0.20247 * (eval(h)**0.725) * (eval(w)**0.425)

def computedeltaxy_gdcm(data):
    '''
    returns the number of cm per pixel in the x and y direction
    0.012 threshold is included as heuristic because (0018,602) code includes other portions of image
    '''
    xlist = []
    ylist = []
    for i in data:
        i = i.lstrip()
        if i.split(" ")[0] == '(0018,602c)':
            deltax = i.split(" ")[2]
            if np.abs(eval(deltax)) > 0.012:
                xlist.append(np.abs(eval(deltax)))
        if i.split(" ")[0] == '(0018,602e)':
            deltay = i.split(" ")[2]
            if np.abs(eval(deltax)) > 0.012:
                ylist.append(np.abs(eval(deltay)))
    return np.min(xlist), np.min(ylist)

def remove_periphery(imgs):
    '''
    retains central cone-shaped area of echo images
    '''
    imgs_ret = []
    for img in imgs:
        image = img.astype('uint8').copy()
        fullsize = image.shape[0] * image.shape[1]
        image[image > 0 ] = 255
        image = cv2.bilateralFilter(image, 11, 17, 17)
        thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = cnts[1]
        areas = []
        for i in range(0, len(contours)):
            areas.append(cv2.contourArea(contours[i]))

        if len(areas) == 0:
            imgs_ret.append(img)
        else:
            select = np.argmax(areas)
            roi_corners_clean = []
            roi_corners = np.array(contours[select], dtype = np.int32)
            for i in roi_corners:
                roi_corners_clean.append(i[0])
            hull = cv2.convexHull(np.array([roi_corners_clean], dtype = np.int32))
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, hull, 1)
            imgs_ret.append(img*mask)
    return np.array(imgs_ret)

def computeft_gdcm(data):
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
        return eval(ft)
    else:
        return None

def output_imgdict(imagefile):
    '''
    converts raw dicom to numpy arrays; some dicom images are YBR_FULL_422 compression; others are uncompressed
    '''
    ds = imagefile
    if len(ds.pixel_array.shape) == 4: #format 3, nframes, nrow, ncol
        nframes = ds.pixel_array.shape[1]
        maxframes = nframes * 3
    elif len(ds.pixel_array.shape) == 3: #format nframes, nrow, ncol
        nframes = ds.pixel_array.shape[0]
        maxframes = nframes * 1
    nrow = int(ds.Rows)
    ncol = int(ds.Columns)
    ArrayDicom = np.zeros((nrow, ncol), dtype=ds.pixel_array.dtype)
    imgdict = {}
    for counter in range(0, maxframes, 3):  
        k = counter % nframes
        j = (counter) // nframes
        m = (counter + 1) % nframes
        l = (counter + 1) // nframes
        o = (counter + 2) % nframes
        n = (counter + 2) // nframes
        if len(ds.pixel_array.shape) == 4: #this is typical YBR_FULL_422
            a = ds.pixel_array[j, k, :, :]
            b = ds.pixel_array[l, m, :, :]
            c = ds.pixel_array[n, o, :, :]
            d = np.vstack((a, b))
            e = np.vstack((d, c))
            g = e.reshape(3 * nrow * ncol, 1)
            y = g[::3]
            u = g[1::3]
            v = g[2::3]
            y = y.reshape(nrow, ncol)
            u = u.reshape(nrow, ncol)
            v = v.reshape(nrow, ncol)
            ArrayDicom[:, :] = ybr2gray(y, u, v)
            ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name for most studies
            counter = counter + 1
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter / 3)
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
        elif len(ds.pixel_array.shape) == 3:
            ArrayDicom[:, :] = ds.pixel_array[counter, :, :]
            ArrayDicom[0:int(nrow / 10), 0:int(ncol)] = 0  # blanks out name for most studies
            counter = counter + 1
            ArrayDicom.clip(0)
            nrowout = nrow
            ncolout = ncol
            x = int(counter / 3)
            imgdict[x] = imresize(ArrayDicom, (nrowout, ncolout))
    return imgdict

def create_imgdict_from_dicom(directory, filename):
    """
    convert compressed DICOM format into numpy array
    """
    targetfile = os.path.join(directory, filename)
    temp_directory = os.path.join(directory, "image")
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    ds = dicom.read_file(targetfile, force = True)
    if ("NumberOfFrames" in  dir(ds)) and (ds.NumberOfFrames>1):
        outrawfile = os.path.join(temp_directory, filename + "_raw")
        command = 'gdcmconv -w ' + os.path.join(directory, filename) + " "  + outrawfile
        subprocess.Popen(command, shell=True)
        time.sleep(10)
        if os.path.exists(outrawfile):
            ds = dicom.read_file(outrawfile, force = True)
            imgdict = output_imgdict(ds)
        else:
            print(outrawfile, "missing")
    return imgdict

def create_mask(imgs):
    '''
    removes static burned in pixels in image
    '''
    from scipy.ndimage.filters import gaussian_filter
    diffs = []
    for i in range(len(imgs) - 1):
        temp = np.abs(imgs[i] - imgs[i + 1])
        temp = gaussian_filter(temp, 10)
        temp[temp <= 50] = 0
        temp[temp > 50] = 1

        diffs.append(temp)

    diff = np.mean(np.array(diffs), axis=0)
    diff[diff >= 0.5] = 1
    diff[diff < 0.5] = 0
    return diff

def ybr2gray(y, u, v):
    '''
    conversion of ybr to grayscale
    '''
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return np.array(gray, dtype="int8")

def extractmetadata(dicomdir, videofile):
    command = 'gdcmdump ' + dicomdir + "/" + videofile
    pipe = subprocess.Popen(command, stdout=PIPE, stderr=None, shell=True)
    text = pipe.communicate()[0]
    data = text.split("\n")
    a = computedeltaxy_gdcm(data)
    if not a == None:
        x_scale, y_scale = a
    else:
        x_scale, y_scale = None, None
    hr = computehr_gdcm(data)
    b = computexy_gdcm(data)
    if not b == None:
        nrow, ncol = b
    else:
        nrow, ncol = None, None
    ft = computeft_gdcm(data)
    bsa = None
    try:
        bsa = computebsa_gdcm(data)
    except Exception as e:
        print e, "bsa"
    return bsa, ft, hr, nrow, ncol, x_scale, y_scale
