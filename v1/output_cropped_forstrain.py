# -*- coding: utf-8 -*-
import os
import dicom
from echoanalysis_tools import *
import cv2
import numpy as np
from scipy.misc import imread, imsave, imresize

def outputcropped(dicomdir, videofile, x_scale, y_scale, nrow, ncol):
    outdir_left = dicomdir + "/maskedimages_left/"
    if not os.path.exists(outdir_left):
        os.path.exists(outdir_left)
    outdir_right = dicomdir + "/maskedimages_right/"
    if not os.path.exists(outdir_right):
        os.path.exists(outdir_right)
    outdir_left = outdir_left + videofile
    outdir_right = outdir_right + videofile
    if not os.path.exists(outdir_left):
        os.makedirs(outdir_left)
    if not os.path.exists(outdir_right):
        os.makedirs(outdir_right)
    npydir = dicomdir + "/unet/"
    lvo_segs = np.load(npydir + "/" + videofile + "_lvo.npy")
    lv_segs = np.load(npydir + "/" + videofile + "_lv.npy")
    imgdict = create_imgdict_from_dicom(dicomdir, videofile)
    mask = create_mask(imgdict)
    kernel_i = np.ones((20,20), np.uint8) #narrower than 20,20
    kernel_o = np.ones((25,25), np.uint8) #narrower than 20,20
    for i in range(0, len(lv_segs)):
        lv_seg = lv_segs[i].copy()
        lvo_seg = lvo_segs[i].copy()
        lv_seg = imresize(lv_seg.copy(), (nrow, ncol), interp='nearest')
        lv_seg_dilate = cv2.dilate(lv_seg, kernel_i, iterations = 2)
        lvo_seg = imresize(lvo_seg.copy(), (nrow, ncol), interp='nearest')
        lvo_seg_dilate = cv2.dilate(lvo_seg, kernel_o, iterations = 1)
        y_i, x_i = np.where(lv_seg_dilate > 0)
        y_o, x_o = np.where(lvo_seg_dilate > 0)
        outer = np.transpose((y_o, x_o))
        inner = np.transpose((y_i, x_i))
        iset =  set([tuple(x) for x in inner])
        oset =  set([tuple(x) for x in outer])
        overlap = np.array([x for x in iset & oset])
        y = overlap[:,0]
        x = overlap[:,1]
        y_left_hi = y[(x < np.percentile(x, 43)) & (y < np.percentile(y, 50))]
        y_left_lo = y[(x < np.percentile(x, 35)) & (y > np.percentile(y, 50)) &
                     (y < np.percentile(y, 90))]
        x_left_hi = x[(x < np.percentile(x, 43)) & (y < np.percentile(y, 50))]
        x_left_lo = x[(x < np.percentile(x, 35)) & (y > np.percentile(y, 50)) &
                     (y < np.percentile(y, 90))]
        y_rite_hi = y[(x > np.percentile(x, 43)) & (y < np.percentile(y, 50))]
        y_rite_lo = y[(x > np.percentile(x, 35)) & (y > np.percentile(y, 50)) &
                     (y < np.percentile(y, 90))]
        x_rite_hi = x[(x > np.percentile(x, 43)) & (y < np.percentile(y, 50))]
        x_rite_lo = x[(x > np.percentile(x, 35)) & (y > np.percentile(y, 50)) &
                     (y < np.percentile(y, 90))]
        y_left = np.hstack((y_left_lo, y_left_hi))
        x_left = np.hstack((x_left_lo, x_left_hi))
        y_rite = np.hstack((y_rite_lo, y_rite_hi))
        x_rite = np.hstack((x_rite_lo, x_rite_hi))
        points_left = np.fliplr(np.transpose((y_left,x_left)))  
        points_right = np.fliplr(np.transpose((y_rite,x_rite)))  
        newimage_left = isolate_obj(imgdict[i],points_left)
        outfile_left = outdir_left + "/" + videofile + "_" + str(i) + ".png"
        cv2.imwrite(outfile_left, newimage_left)   
        newimage_right = isolate_obj(imgdict[i],points_right)
        outfile_right = outdir_right + "/" + videofile + "_" + str(i) + ".png"
        cv2.imwrite(outfile_right, newimage_right)   
    return 1

def isolate_obj(inputimage, points):
    roi_corners = np.array(points, dtype = np.int32)
    hull = roi_corners
    channel_count = 1
    mask = np.zeros(inputimage.shape, dtype=np.uint8)
    for i in roi_corners:
        mask[i[1], i[0]] = 255
    ignore_mask_color = (255,)*channel_count
    masked_image = cv2.bitwise_and(inputimage, mask)
    return(masked_image)
