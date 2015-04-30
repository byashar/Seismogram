# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 11:54:50 2015

@author: benamy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage
from skimage.filter import sobel
from skimage.draw import circle
from scipy.ndimage import label
from skimage import color
from skimage import io
from skimage.morphology import dilation, erosion, closing
from scipy.stats import percentileofscore

image = plt.imread('/home/benamy/Documents/Seismogram Project/Images/target.png')
image_gray = color.rgb2gray(image)

def gray_dec2int(a):
    a = a * 255
    a = np.round(a)
    a = a.astype(uint8)
    return a    

def normalize(a):
    b = a - np.amin(a)
    b = b.astype(float) / np.amax(b)
    return b
    
def show_me(a):
    io.imsave('/home/benamy/Documents/Seismogram Project/Images/show.png', 
          normalize(a))

def show_me_true(a):
    io.imsave('/home/benamy/Documents/Seismogram Project/Images/show.png', a)
    
def background_intensity(a, prob_background = 1):
    if np.amax(a) <= 1:
        bins = np.linspace(0, 1, num = 257)
        a = np.round(255 * a) / 255
    else:
        bins = np.arange(257)
    hist, bin_edges = np.histogram(a, bins = bins)  
    # Pad counts with 1 (to eliminate zeros)    
    hist = hist + 1
    i = np.argmax(hist)
    background_count = np.zeros((256))
    background_count[0:(i + 1)] = hist[0:(i + 1)]
    background_count[(i + 1):(2 * i + 1)] = hist[(i - 1)::-1]
    probabilities = background_count / hist
    return bin_edges[np.argmin(probabilities >= prob_background) - 1]

def percent_background(a):
    if np.amax(a) <= 1:
        bins = np.linspace(0, 1, num = 257)
        a = np.round(255 * a) / 255
    else:
        bins = np.arange(257)
    hist, bin_edges = np.histogram(a, bins = bins)  
    i = np.argmax(hist)
    num_background = 2 * np.sum(hist[0:i]) + hist[i]
    return num_background / a.size

def noise_boundaries(a, perc_back):
    '''
    Assumes that noise from background pixels is centered around 0.
    '''
    noise_center = percentileofscore(a.flat, 0, kind = 'mean')
    noise_edges = [noise_center - 25 * perc_back, 
                   noise_center + 25 * perc_back]
    dispersion = np.percentile(a.flat, noise_edges)
    noise_boundaries = 4 * np.asarray(dispersion)
    return noise_boundaries

def gray_to_photon_count(image_gray, B_max = 1.01, B_min = 0):
    if B_min == 0:
        B_min = background_intensity(image_gray)
    image_gray = np.maximum(image_gray, B_min)
    image_photons = np.log(B_max - B_min) - np.log(B_max - image_gray)
    image_photons = normalize(image_photons)
    return image_photons

def mark_coords(shape, coords):
    markers = np.zeros(shape, dtype=bool)    
    for x in coords:
        markers[x[0],x[1]] = True
    return markers

def color_markers(marker_image, background, marker_color=[1,0,0]):
    if background.ndim == 2:
        background = color.gray2rgb(background)
    image_color = np.ndarray(np.r_[marker_image.shape,3])
    image_color[:,:] = marker_color
    marker_image = np.dstack((marker_image,marker_image,marker_image))
    overlay = np.where(marker_image, image_color, background)
    marker_image = marker_image[:,:,0]
    return overlay
    
def draw_circle(image,coords,radius):
    rr, cc = circle(coords[0], coords[1], radius)
    image[rr, cc] = True
