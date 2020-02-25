# -*- coding: utf-8 -*-
import os
from echoanalysis_tools import create_mask, create_imgdict_from_dicom
import cv2
import numpy as np
from scipy.misc import imresize

outDir = "./strainOutput/"
segDir = "./segment/"

def outputcropped(videoFile, view):
        flag = 1
        print(videoFile, view)
        outDir = "./straintmp/" + videoFile
        dicomdir = "./dicomsample/"
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        outDir_left = outDir + "/maskedimages_left/"
        if not os.path.exists(outDir_left):
            os.makedirs(outDir_left)
        outDir_right = outDir + "/maskedimages_right/"
        if not os.path.exists(outDir_right):
            os.makedirs(outDir_right)
        npydir = "./segment/" + view
        lvo_segs = np.load(npydir + "/" + videoFile +
                          "_lvo.npy")
        lv_segs = np.load(npydir + "/" + videoFile +
                          "_lv.npy")
        imgdict = create_imgdict_from_dicom(dicomdir, videoFile)
        nrow = imgdict[0].shape[0]
        ncol = imgdict[0].shape[1]
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
            if len(overlap.shape) == 2:
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
                outfile_left = outDir_left + "/" + videoFile + "_" + str(i) + ".png"
                cv2.imwrite(outfile_left, newimage_left)   
                newimage_right = isolate_obj(imgdict[i],points_right)
                outfile_right = outDir_right + "/" + videoFile + "_" + str(i) + ".png"
                cv2.imwrite(outfile_right, newimage_right)   
            else:
                flag = 0
        if flag == 1:
            return 1
        else:
            return 0
        return 1

def isolate_obj(inputimage, points):
    roi_corners = np.array(points, dtype = np.int32)
    hull = roi_corners
    channel_count = 1
    mask = np.zeros(inputimage.shape, dtype=np.uint8)
    for i in roi_corners:
        mask[i[1], i[0]] = 255
    ignore_mask_color = (255,)*channel_count
    #cv2.fillConvexPoly(mask, hull, ignore_mask_color)
    masked_image = cv2.bitwise_and(inputimage, mask)
    return(masked_image)
