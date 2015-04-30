# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:21:14 2014

@author: benamy
"""
import numpy as np
from matplotlib.pyplot import scatter
from scipy.interpolate import LSQUnivariateSpline
from scipy.interpolate import UnivariateSpline

class segment:
    '''
    Segment docstring placeholder
    '''    
    
    def __init__(self, pixel_coords, image_shape):
        self.set_region(pixel_coords, image_shape)

    def set_region(self, pixel_coords, image_shape):
        self.region = region(pixel_coords, image_shape)
        self.size = self.region.size
        self.domain = self.region.region_domain
        self.range = self.region.region_range
        self.set_linear_fit()
    
    def binary_image(self):
        return self.region.binary_image()
    
    def add_ridge_line(self, pixel_coords):
        self.ridge_line = pixel_coords
        self.set_pixel_series()
    
    def add_horizontal_ridges(self, coords):
        self.ridge_line_h = coords        
        
    def add_vertical_ridges(self, coords):
        self.ridge_line_v = coords
    
    def set_pixel_series(self):
        self.pixel_series = ridge_line_to_series(self.ridge_line)
        #self.set_center_line()
    
    def plot_ridge_line(self):
        scatter(self.ridge_line[:,1], (-self.ridge_line[:,0]))
    
    def plot_pixel_series(self):    
        scatter(self.pixel_series[:,1], (-self.pixel_series[:,0]))
    
    def set_linear_fit(self):
        self.linear_fit = linear_fit(self.region.coords)
        
    def set_center_line(self):
        self.center_line = series_to_center_line(self.ridge_line)

    def plot_center_line(self):
        plot(self.center_line[:,1], (-self.center_line[:,0]), '-')

class region:
    ''' The coordinates of a collection of pixels.
    '''
    #coords = np.array([], dtype=int, ndmin=2)
    #dims = np.array([0,0], dtype=int)
    #ID = 0
    centroid = np.array([0,0], dtype=int)
    region_domain = np.array([0,0], dtype=int)
    region_range = np.array([0,0], dtype=int)
    size = 0

    
    def __init__(self, pixel_coords, image_shape, region_ID=0):    
        self.coords = pixel_coords
        self.dims = image_shape
        self.ID = region_ID
        self.calc_properties()
    
    def calc_properties(self):
        self.region_domain[0] = np.amin(self.coords[:,1])
        self.region_domain[1] = np.amax(self.coords[:,1])
        self.region_range[0] = np.amin(self.coords[:,0])
        self.region_range[1] = np.amax(self.coords[:,0])
        self.centroid = np.mean(self.coords, axis=0)
        self.size = self.coords.shape[0]
        
    def add_pixels(self, pixel_coords):
        pixel_list = list(self.coords)
        pixel_list = pixel_list + list(pixel_coords)
        pixel_list = map(tuple, pixel_list)
        pixel_list = list(set(pixel_list)) # remove duplicates
        pixel_list = np.asarray(tuple(pixel_list), dtype=int)
        self.coords = pixel_list
        self.calc_properties()        
        
    def remove_pixels(self, pixel_coords):
        curr_pixels = set(map(tuple, list(self.coords)))
        pixels_to_remove = set(map(tuple, list(pixel_coords)))
        new_pixel_list = curr_pixels - pixels_to_remove
        new_pixel_list = np.asarray(tuple(new_pixel_list), dtype=int)
        self.coords = new_pixel_list
        self.calc_properties()
        
    def binary_image(self):
        img = np.zeros(self.dims, dtype=bool)
        for p in self.coords:
            img[p[0], p[1]] = True
        return img
    
    def is_in_region(self, pixel_coords):
        matches = np.logical_and(self.coords[:,0] == pixel_coords[0],
                                 self.coords[:,1] == pixel_coords[1])
        return np.any(matches)

    def pixel_row(self, row):
        return self.coords[(self.coords[:,0] == row),:]

    def pixel_column(self, col):
        return self.coords[(self.coords[:,1] == col),:]

class region:
    ''' The coordinates of a collection of pixels.
    '''
    
    def __init__(self, region, origin, image_shape, region_ID=0):    
        self.region = region
        self.origin = origin
        self.image_shape = image_shape
        self.ID = region_ID
        self.calc_properties()
    
    def calc_properties(self):
        self.region_domain[0] = self.origin[1]
        self.region_domain[1] = self.origin[1] + self.region.shape[1]
        self.region_range[0] = self.origin[0] 
        self.region_range[1] = self.origin[0] + self.region.shape[0]
        self.centroid = np.mean(self.coords, axis=0)
        self.size = self.coords.shape[0]
        
    def add_pixels(self, pixel_coords):
        pixel_list = list(self.coords)
        pixel_list = pixel_list + list(pixel_coords)
        pixel_list = map(tuple, pixel_list)
        pixel_list = list(set(pixel_list)) # remove duplicates
        pixel_list = np.asarray(tuple(pixel_list), dtype=int)
        self.coords = pixel_list
        self.calc_properties()        
        
    def remove_pixels(self, pixel_coords):
        curr_pixels = set(map(tuple, list(self.coords)))
        pixels_to_remove = set(map(tuple, list(pixel_coords)))
        new_pixel_list = curr_pixels - pixels_to_remove
        new_pixel_list = np.asarray(tuple(new_pixel_list), dtype=int)
        self.coords = new_pixel_list
        self.calc_properties()
        
    def binary_image(self):
        img = np.zeros(self.dims, dtype=bool)
        for p in self.coords:
            img[p[0], p[1]] = True
        return img
    
    def is_in_region(self, pixel_coords):
        matches = np.logical_and(self.coords[:,0] == pixel_coords[0],
                                 self.coords[:,1] == pixel_coords[1])
        return np.any(matches)

    def pixel_row(self, row):
        return self.coords[(self.coords[:,0] == row),:]

    def pixel_column(self, col):
        return self.coords[(self.coords[:,1] == col),:]

class intersection:
    pass

class node:
    node_type = '' # intersection or dead end
    location = (0,0)
    radius = 0
    degree = 0
    segment_IDs = []

    def connect_segment(self, seg):
        self.segment_IDs.append(seg.ID)
        self.segment_IDs.sort()
        self.degree = self.degree + 1
    
    def get_segments(self):
        return segment_IDs

class pixel_path:    
    ID = 0    
    pixel_path = []
    end1 = (0,0)
    end2 = (0,0)
    end1_type = ''
    end2_type = ''
    length = 0
    displacement = 0
    path_domain = (0,0)   # min and max column values
    path_range = (0,0)  # min and max row values
    
    def __init__(self, pixels, ID):
        if len(pixels) > 0:
            self.pixel_path = pixels
            self.ID = ID
            self.calc_properties()

    def calc_properties(self):
        self.length = len(self.pixel_path)
        self.end1 = self.pixel_path[0]
        self.end2 = self.pixel_path[-1]
        self.displacement = sqrt((self.end1[0] - self.end2[0])^2 + 
                                 (self.end1[1] - self.end2[1])^2)
        self.path_domain[0] = min(self.pixel_path, key=lambda x: x[1])
        self.path_domain[1] = max(self.pixel_path, key=lambda x: x[1])
        self.path_range[0] = min(self.pixel_path, key=lambda x: x[0])
        self.path_range[1] = max(self.pixel_path, key=lambda x: x[0])

    
    def extend_path(self, pixels):
        '''Input should be a list of tuples representing pixels. 
        '''
        self.pixel_path = self.pixel_path + pixels
        self.calc_properties()
    
    def binary_image(self, dims):
        img = np.zeros(dims, dtype=bool)
        for p in self.pixel_path:
            img[p[0], p[1]] = True
        return img

class pixel_series:
    coords = np.array([], dtype=int, ndmin=2)
    ID = 0
    series_domain = np.array([0,0], dtype=int)
    series_range = np.array([0,0], dtype=int)
    length = 0
    end1_linear_fit = np.array([0,0])
    end2_linear_fit = np.array([0,0])
    end1_quadratic_fit = np.array([0,0,0])
    end2_quadratic_fit = np.array([0,0,0])
    
    END_LENGTH = 6
    
    def __init__(self, pixels, ID):
        if len(pixels) > 0:
            self.pixel_series = pixels
            self.ID = ID
            self.calc_properties()

    def calc_properties(self):
        self.length = self.coords.shape[0]
        self.series_domain[0] = np.amin(self.coords[:,1])
        self.series_domain[1] = np.amax(self.coords[:,1])
        self.series_range[0] = np.amin(self.coords[:,0])
        self.series_range[1] = np.amax(self.coords[:,0])
        
    def get_segment(self, col1, col2):
        i1 = col1 - self.series_domain[0]
        i2 = col2 - self.series_domain[0]
        if i1 >= 0 and i2 < self.length:
            return self.coords[i1:i2,:]
        else:
            return

    def get_linear_fits(self):
        if self.length >= END_LENGTH:
            end = self.get_segment(self.series_domain[0], 
                                   self.series_domain[0] + END_LENGTH - 1)
            self.end1_linear_fit = linear_fit(end)
            end = self.get_segment(self.series_domain[1] - END_LENGTH + 1, 
                                   self.series_domain[1])
            self.end2_linear_fit = linear_fit(end)
            
    def get_quadratic_fits(self):
        if self.length >= END_LENGTH:
            end = self.get_segment(self.series_domain[0], 
                                   self.series_domain[0] + END_LENGTH - 1)
            self.end1_quadratic_fit = quadratic_fit(end)
            end = self.get_segment(self.series_domain[1] - END_LENGTH + 1, 
                                   self.series_domain[1])
            self.end2_quadratic_fit = quadratic_fit(end)
            
class center_line:
    
    def __init__(self, spline, domain):
        self.spline = spline
        self.domain = domain
    
    # def all_pixels(self):

    def pixels(self, x):
        df_dx = self.spline(x, 1)        
        f_l = self.spline(x)
        f_r = self.spline(x + 0.5)
        if df_dx == 0:
            return np.array([[x, f_l, 1]])
        if df_dx > 0:
            left_cell = np.floor(f_l)
            left_cell_value = f_l - np.floor(f_l)
            right_cell = np.ceil(f_r)
            right_cell_value = np.ceil(f_r) - f_r
        if df_dx < 0:
            left_cell = np.ceil(f_l)
            left_cell_value = np.ceil(f_l) - f_l
            right_cell = np.floor(f_r)
            right_cell_value = f_r - np.floor(f_r)
        pixels = np.arange(left_cell, right_cell, np.sign(df_dx))
        cell_value = np.abs(1 / df_dx)
        cell_value[0] = left_cell_value
        cell_value[-1] = right_cell_value
        

        
        num_divs = np.abs(np.round(df_dx))
        div_size = 1 / num_divs
        divs = np.arange((x - 0.5 + (div_size / 2)), (x + 0.5), div_size)
        y = self.spline(divs)

    '''
    Got lazy. The function below was the beginning of a more complicated method

    def pixels(self, x):
        df_dx = self.spline(x, 1)
        num_divs = np.abs(np.round(df_dx))
        div_size = 1 / num_divs
        divs = np.arange((x - 0.5 + (div_size / 2)), (x + 0.5), div_size)
        y = self.spline(divs)
        y_u = np.ceil(y)
        y_d = np.floor(y)
    '''
        
        
'''
Linear regression functions
'''
def linear_fit(coords):
    y = coords[:,0]    
    x0 = np.ones_like(coords[:,1])
    x1 = coords[:,1]
    A = array([x1, x0])
    w = np.linalg.lstsq(A.T, y)[0]
    return w

def quadratic_fit(coords):
    y = coords[:,0]    
    x0 = np.ones_like(coords[:,1])
    x1 = coords[:,1]
    x2 = coords[:,1] ^ 2
    A = array([x2, x1, x0])
    w = np.linalg.lstsq(A.T, y)[0]
    return w

'''
Analysis functions
'''
def ridge_line_to_series(coords):
    if coords.size == 0:
        return np.array([[0,0]])
    domain = (min(coords[:,1]), max(coords[:,1]))
    series = []    
    for x in range(domain[0], domain[1] + 1):
        y = coords[coords[:,1] == x][:,0]
        if y.size > 0:
            y = np.mean(y)
            series.append(np.array([y, x]))
    series = np.asarray(series)
    return series

def vertical_ridge_line_to_series(coords):
    if coords.size == 0:
        return np.array([[0,0]])
    domain = (min(coords[:,1]), max(coords[:,1]))    
    series = []    
    if (domain[1] - domain[0]) < 2:
        return np.array([0,0])    
    # Exclude first and last columns of vertical ridge pixels
    for x in range(domain[0] + 1, domain[1]):
        y = coords[coords[:,1] == x][:,0]
        if y.size > 0:
            y = np.mean(y)
            series.append(np.array([y, x]))
    series = np.asarray(series)
    return series

def series_to_center_line(series, points_per_knot = 5):
    domain = (min(series[:,1]), max(series[:,1]))    
    num_points = series.size / 2    
    if (domain[1] - domain[0]) < 5:
        return
    num_knots = floor((domain[1] - domain[0]) / points_per_knot)
    points_per_knot = float(num_points) / num_knots
    knots = linspace(domain[0] + points_per_knot, domain[1] - points_per_knot, 
                     num_knots - 1)
    fit = LSQUnivariateSpline(series[:,1], series[:,0], t= knots, 
                              bbox = domain)
    x = np.arange(domain[0], domain[1]+1)
    y = fit(x)
    return np.hstack((y[:, np.newaxis], x[:, np.newaxis]))

class intersection(region):
    pass



class timing_mark(region):
    pass


class gap(region):
    pass
    

