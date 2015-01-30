# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:38:19 2015

@author: benamy
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import convolve2d

from skimage.morphology import medial_axis
from skimage import color
from skimage import io
from skimage.draw import circle

def find_intersections(image_bin, figure=True):
    '''    
    Finds the intersections of traces by skeletonizing the image.
    
    Parameters
    -------------
    image_bin : 2-D Boolean numpy array
        A binary image.
    figure : bool
        If True, function outputs a binary image with 
        pixels in intersection regions set to True. If set to False,
        function outputs a list of circular regions making up the 
        intersections.
    
    Returns
    ----------
    
    '''
    image_skel,dist = medial_axis(image_bin, return_distance = True)
    image_dead_ends = find_dead_ends(image_skel)
    dentrites = get_all_pixel_paths(image_skel,image_dead_ends, 
                                    max_path_length = 50)
    for d in dentrites:
        pix_i = d[0]    
        pix_f = d[-1]
        dentrite_displacement = np.sqrt( (pix_i[0] - pix_f[0]) ** 2 + 
                                     (pix_i[1] - pix_f[1]) ** 2)
        if dentrite_displacement < 2.5 * dist[pix_f[0], pix_f[1]]:
            remove_pixels(image_skel,d)
    
    intersections, degrees = find_junctions(image_skel)
    radii = get_intersection_sizes(intersections,dist)
    image_intersections = mark_coords(image_skel.shape, intersections)
    expand_junctions(image_intersections, intersections,radii)    

def find_dead_ends(skeleton):
    '''
    Finds dead ends in the skeletonized image. It does this by finding
    pixels in the skeleton (labeled True in the image) that are 
    adjacent to exactly one other pixel in the skeleton. Adjacent, in this
    case, includes the four diagonal pixels. 
    
    Parameters
    ------------
    skeleton : 2-D Boolean numpy array
        A binary image of the skeleton.
        
    Returns
    --------
    dead_end_indices : ndarray
        A 2-D array with two columns, listing the indices of the 
        dead-end pixels in the image.
    '''
    skeleton = skeleton.astype(int)    
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    dead_ends = convolve2d(skeleton, kernel, mode='same')
    dead_ends = dead_ends == 11
    dead_end_indices = np.argwhere(dead_ends)
    return dead_end_indices

def find_junctions(skeleton):
    ''' 
    Finds junctions in the skeletonized image. It does this by finding
    pixels in the skeleton (labeled True in the image) that are
    adjacent to three or more other pixels in the skeleton. Adjacent,
    in this case, includes the four diagonal pixels.
    
    Parameters
    ------------
    skeleton : 2-D Boolean numpy array
        A binary image of the skeleton.
        
    Returns
    --------
    junctions : ndarray
        A 2-D array with two columns, listing the indices of the 
        pixels determined to be junctions.
    degrees : numpy array
        A 1-D array, with the same number of rows as junctions, 
        listing the degree (number of adjacent pixels in skeleton) 
        of each junction. 
    '''
    skeleton = skeleton.astype(int)    
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    connectivity = convolve2d(skeleton,kernel, mode='same')
    connectivity = connectivity - 10    
    junctions = connectivity >= 3
    junctions = np.argwhere(junctions)
    degrees = connectivity[junctions[:,0], junctions[:,1]]
    return (junctions, degrees)

def get_pixel_path(pixel_array, curr_pixel, prev_pixel=np.array([-1,-1]), \
                    max_path_length=-1):
    '''
    Follows and records a 1-pixel-wide path in a skeletonized image until
    it runs into a junction or a dead end. It does this by jumping from
    pixel to adjacent pixel until there are is more than one adjacent 
    pixel (not counting the pixel it jumped from) or there are no more 
    adjacent pixels. 
    
    Parameters
    -------------
    pixel_array : 2-D Boolean array
        The skeletonized image.
    curr_pixel : numpy array
        The indices (row, col) of the pixel from which to trace a path.
    prev_pixel : numpy array, optional
        The indices (row, col) of the previous pixel in the path. Specify
        to follow a pixel path, starting in the middle of the pixel path.
        Enables recursion.
    max_path_length : int
        The maximum length of a pixel path. Function returns the pixel
        path if this limit has been reached. 
        
    Returns 
    ---------
    connectivity : 
    '''    
    pixel_path = []
    connectivity = 0
    if max_path_length == 0:
        return [0, []]
    image_shape = np.shape(pixel_array)        
    
    # Check the surrounding 8 pixels 
    pixels_to_explore = np.array([[-1,-1], [-1,0], [-1,1],
                                  [0,-1],          [0,1],
                                  [1,-1],  [1,0],  [1,1]]) \
                        + curr_pixel    
    for x in pixels_to_explore:
        # If the indices correspond to a pixel in the image that's
        # not the previous pixel...        
        if x[0] >= 0 and x[0] < image_shape[0] \
        and x[1] >= 0 and x[1] < image_shape[1] \
        and (x != prev_pixel).any():
            # And the pixel is part of the skeleton            
            if pixel_array[x[0],x[1]]:
                pixel_path.append(x)
    
    connectivity = len(pixel_path)
    if connectivity == 1:
        next_pixel = pixel_path[0]
        pixel_path = [curr_pixel]            
        connectivity, extension = get_pixel_path(pixel_array, next_pixel,
                                                 curr_pixel, max_path_length-1)
        pixel_path = pixel_path + extension
        return [connectivity, pixel_path]
    else:
         return [connectivity, [curr_pixel]]

def get_all_pixel_paths(pixel_array, dead_ends, max_path_length=-1):
    '''
        
    '''
    paths = []    
    for d_e in dead_ends:
        conn, path = get_pixel_path(pixel_array, d_e, max_path_length)
        if conn >= 2:
            paths.append(path)
    return paths

def remove_pixels(pixel_array, pixels_to_remove):
    for p in pixels_to_remove:
        pixel_array[p[0],p[1]] = False
        
        
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

def expand_junctions(image, junctions, radii):
    for center,radius in zip(junctions,radii):
        draw_circle(image,center,radius)

def get_intersection_sizes(intersections,distance_transform):
    sizes = []
    for i in intersections:
        sizes.append(distance_transform[i[0],i[1]])
    return sizes

def output_intersections(header, *args):
    intersection_data = zip(*args)
    filepath = os.path.dirname(os.getcwd()) + '/intersections.csv'
    with open(filepath,'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(intersection_data)

intersections, degrees = find_junctions(image_skel)
radii = get_intersection_sizes(intersections,dist)
image_intersections = mark_coords(image_skel.shape, intersections)
expand_junctions(image_intersections, intersections,radii)
image_intersections = color_markers(image_intersections, color.gray2rgb(image_gray))

image_intersections = color_markers(image_skel,image_intersections,marker_color=[0,1,0])


output_intersections(['Row', 'Column', 'Degree', 'Radius'], \
    intersections[:,0], intersections[:,1], degrees, radii)