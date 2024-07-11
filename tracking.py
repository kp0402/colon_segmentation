# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 00:01:07 2022

@author: zhanz173

adapted from:
    https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""
import numpy as np
import cv2 as cv

#global setting
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 20,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

#-------------------------------------

def define_tracking_region(frame,r):
    assert (frame.ndim == 2), "gray scale image expected"
    mask = np.zeros_like(frame)
    mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
    return cv.goodFeaturesToTrack(frame, mask=mask,**feature_params)
        
def dectect_outlier(data):
    threshold = 2
    max_iter = 5
    mean = np.mean(data)
    std = np.std(data)
    z = np.abs((data-mean))/(std+1e-6)
    outlier = z > threshold
    
    while std > 5 and max_iter > 0:
        filtered_data = data[outlier == False]
        mean = np.mean(filtered_data)
        std = np.std(filtered_data)
        z = np.abs((data-mean))/(std+1e-6)
        threshold = min(max(z),threshold)
        outlier = z >= threshold
        max_iter -= 1
    

    filtered_data = data[outlier == False]
    mean = np.mean(filtered_data)
     
    return outlier, mean

def replace_tracking_point(p0,p1,ptr_scores, replacement, ptr):
    dx = p1[:,0] - p0[:,0]
    outlier, movement = dectect_outlier(dx)
    ptr_scores += 1-outlier
    ptr_scores -= 2*(abs(dx) < 1) 
    
    p0[:,0] = p0[:,0] * outlier +  p1[:,0] * (1 - outlier)
    p0[:,1] = p0[:,1] * outlier +  p1[:,1] * (1 - outlier)
    
    if replacement:
        i = np.argmin(ptr_scores)
        if len(ptr) == 0:
            p0[i] = p0[i] + 50 * np.random.uniform((p0[i].shape))
            ptr_scores[i] = sum(ptr_scores)/len(ptr_scores)
        else:
            p0[i] = ptr.pop(0)
            ptr_scores[i] = sum(ptr_scores)/len(ptr_scores)
  
    return p0,ptr_scores, movement

def tracking(new_frame, old_frame, tracking_points, scores, replace=False, replacements=None):
    p1, _, _ = cv.calcOpticalFlowPyrLK(old_frame, new_frame, 
                                            tracking_points, None, **lk_params)
    tracking_points, scores, dx = replace_tracking_point(tracking_points,p1, scores,replace, replacements)

    return tracking_points, scores, dx
