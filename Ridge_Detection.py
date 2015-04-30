# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:24:55 2015

@author: benamy
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:20:36 2015

@author: benamy
"""


import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter1d, gaussian_laplace
import itertools as itt
import math
from math import sqrt, hypot, log
from numpy import arccos
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
# from ._hessian_det_appx import _hessian_matrix_det
from skimage.transform import integral_image
from scipy.signal import argrelextrema
from skimage.morphology import dilation, remove_small_objects
from scipy.signal import convolve2d
from scipy.ndimage import label, labeled_comprehension
from scipy.stats import percentileofscore

from numpy.random import random_integers

from scipy.ndimage.filters import (gaussian_filter1d, gaussian_laplace,
                                   generic_filter1d, maximum_filter1d)
 
              
def ridge_region_vert(ridges, shape):
    '''
    ridges are tuples like (row, col, sigma, max_value)
    ridge width is sqrt(2) * sigma
    '''    
    img = np.zeros(shape, dtype=float)
    
    for r in ridges:
        width = sqrt(2) * r[2]    
        #width = r[2]
        lower_bound = max(0, round(r[1] - width))
        upper_bound = min(shape[1]-1, round(r[1] + width))
        img[r[0], lower_bound:upper_bound] = r[3]
    return img
        
def ridge_region_horiz(ridges, shape):
    '''
    ridges are tuples like (row, col, sigma, max_value)
    ridge width is sqrt(2) * sigma
    '''    
    img = np.zeros(shape, dtype=float)
    
    for r in ridges:
        width = sqrt(2) * r[2]        
        #width = r[2]        
        lower_bound = max(0, round(r[0] - width))
        upper_bound = min(shape[0]-1, round(r[0] + width))
        img[lower_bound:upper_bound, r[1]] = r[3]
    return img

def convex(a, perc_back):
    '''
    Assumes that noise from background pixels is centered around 0.
    '''
    noise_center = percentileofscore(a.flat, 0, kind = 'mean')
    dispersion = np.percentile(a.flat, noise_center - 25 * perc_back)
    return a <= 0 * (dispersion / 2)

def ridge_height_threshold(a, perc_back):  
    return 0.02
    '''
    Assumes that noise from background pixels is centered around 0.    
    '''
    '''    
    noise_center = percentileofscore(a.flat, 0, kind = 'mean')
    dispersion = np.percentile(a.flat, 
                               (noise_center + 25 + 0.25 * perc_back))
    #return dispersion    
    return 0
    '''
    
    

def ridge_lines(img, min_ridge_length = 10):
    return remove_small_objects(img, min_ridge_length, connectivity = 2)

def find_ridges(image_gray, min_sigma = 0.7071, max_sigma = 30, 
                        sigma_ratio = 1.6, min_ridge_length = 15, 
                        low_threshold = 0.001, high_threshold = 0.005, 
                        figure=False):
    #preliminaries
    img_dark_removed = flatten_background(image_gray, 0.9)
    dark_pixels = image_gray < background_threshold(image_gray, 0.9)
    #bright_pixels = image_gray > foreground_threshold(image_gray, 1)
    perc_background = percent_background(image_gray)  
    
    abs_isobel = np.abs(ndimage.sobel(image_gray, axis=0))
    abs_jsobel = np.abs(ndimage.sobel(image_gray, axis=1))
    vertical_slopes = abs_isobel > threshold_otsu(abs_isobel)
    horizontal_slopes = abs_jsobel > threshold_otsu(abs_jsobel)    
    
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                          for i in range(k + 1)])

    gaussian_blurs_h = [gaussian_filter1d(img_dark_removed, s, 0) \
                        for s in sigma_list]
    gaussian_blurs_v = [gaussian_filter1d(img_dark_removed, s, 1) \
                        for s in sigma_list]


    # computing difference between two successive Gaussian blurred images
    image_cube_h = np.ma.zeros((image_gray.shape[0], image_gray.shape[1], k),
                               mask = False)    
    image_cube_v = np.ma.zeros((image_gray.shape[0], image_gray.shape[1], k),
                               mask = False)    
    exclusion = np.zeros((image_gray.shape[0], image_gray.shape[1], k), 
                         dtype = bool) 
    exclusion_layer = np.copy(dark_pixels)
    for i in range(k):
        image_cube_h[:,:,i] = ((gaussian_blurs_h[i] - gaussian_blurs_h[i + 1]))
        image_cube_v[:,:,i] = ((gaussian_blurs_v[i] - gaussian_blurs_v[i + 1]))
        exclusion_layer = (exclusion_layer | 
                           convex(image_cube_h[:,:,i], perc_background) |
                           convex(image_cube_v[:,:,i], perc_background))
        exclusion[:,:,i] = exclusion_layer
    
    # Find horizontal ridges first 
    footprint_h = np.ones((3,1,3), dtype=bool)
    image_cube_h_norm = normalize(image_cube_h)
    maxima_h = peak_local_max(image_cube_h_norm, indices=False, min_distance=1, 
                    threshold_rel=0, threshold_abs=0, exclude_border = False,
                    footprint = footprint_h)
    maxima_h = maxima_h & (~exclusion) & (image_cube_h >= low_threshold)
    locations_h = np.amax(maxima_h, axis=-1)
    max_values_h = np.ma.amax(image_cube_h, axis = -1)    
    sigmas_h = np.argmax(image_cube_h, axis=-1)
    sigmas_h = min_sigma * np.power(sigma_ratio, sigmas_h)
    
    
    footprint_v = np.ones((1,3,3), dtype=bool)
    image_cube_v_norm = normalize(image_cube_v)
    maxima_v = peak_local_max(image_cube_v_norm, indices=False, min_distance=1, 
                    threshold_rel=0, threshold_abs=0,
                    footprint = footprint_v)
    maxima_v = maxima_v & (~exclusion) & (image_cube_v >= low_threshold)
    locations_v = np.amax(maxima_v, axis=-1)
        
    max_values_v = np.ma.amax(image_cube_v, axis = -1)    
    sigmas_v = np.argmax(image_cube_v, axis=-1)
    sigmas_v = min_sigma * np.power(sigma_ratio, sigmas_v)  
    
    # Filter peaks below low_threshold
    #locations_h = (locations_h & (max_values_h >= low_threshold))
    # Filter horizontal ridges
    locations_h = locations_h & (~horizontal_slopes)
    # Ridges need to either be prominent or highly connected
    locations_h = (locations_h & 
            #((max_values_h > ridge_height_threshold(image_cube_h, 
            #                                        perc_background)) |
            #((max_values_h >= threshold_otsu(max_values_h)) |
            #((max_values_h > np.percentile(max_values_h[bright_pixels], 90)) |            
            ((max_values_h >= high_threshold) | 
                (ridge_lines(locations_h, min_ridge_length))))
    
    # Filter peaks below low_threshold
    #locations_v = locations_v & (max_values_h >= low_threshold)
    # Filter vertical ridges
    locations_v = locations_v & (~ vertical_slopes)
    # Ridges need to either be prominent or highly connected
    locations_v = ((locations_v) & 
            #((max_values_v > ridge_height_threshold(image_cube_h, 
            #                                        perc_background)) |
            #((max_values_v >= threshold_otsu(max_values_v)) |
            #((max_values_v > np.percentile(max_values_v[bright_pixels], 67)) |
            ((max_values_h >= high_threshold) |
                (ridge_lines(locations_v, min_ridge_length))))
    
    # Aggregate information about maxima of horizontal ridges    
    indices_h = np.argwhere(locations_h)    
    sigmas_h = sigmas_h[locations_h]    
    max_values_h = max_values_h[locations_h]     
    maxima_h = np.hstack((indices_h, sigmas_h[:,np.newaxis], 
                          max_values_h[:,np.newaxis]))
    
    # Aggregate information about maxima of vertical ridges    
    indices_v = np.argwhere(locations_v)    
    sigmas_v = sigmas_v[locations_v]    
    max_values_v = max_values_v[locations_v]     
    maxima_v = np.hstack((indices_v, sigmas_v[:,np.newaxis], 
                          max_values_v[:,np.newaxis]))    
        
    # Prioritize horizontal regions
    horizontal_regions = ridge_region_horiz(maxima_h, image_gray.shape)
    locations_v = locations_v & (horizontal_regions == 0)       
    
    return (locations_h, locations_v)
