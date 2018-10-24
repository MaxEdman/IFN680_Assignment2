'''

This module contains functions
- to load the original image dataset
- to generate random homographies
- to warp randomly images with random homographies

'''

import numpy as np

import random

from tensorflow.contrib import keras


from tensorflow.contrib.keras import backend as K

from skimage import transform

#------------------------------------------------------------------------------

def load_dataset():
    '''
    Load the dataset, shuffled and split between train and test sets
    and return the numpy arrays  x_train, y_train, x_test, y_test
    The dtype of all returned array is uint8
    
    @returnInstructions: 

        x_train, y_train, x_test, y_test
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    return x_train, y_train, x_test, y_test

#------------------------------------------------------------------------------
    
def random_homography(variation, image_side):
    '''
    Generate a random homography.  
    The large the value of variation the more deformation is applied.
    
    The homography is defined by 4 random points.

       @param
       
           variation:    percentage (in decimal notation from 0 to 1)
                         relative size of a circle region where centre is projected
                         
           image_side:   
                         length of the side of an input square image in pixels
       
       @return
       
           tform:        object from skimage.transfrm
    
    '''
    d = image_side * variation
    
    top_left =    (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Top left corner
    bottom_left = (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))   # Bottom left corner
    top_right =   (random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))     # Top right corner
    bottom_right =(random.uniform(-0.5*d, d), random.uniform(-0.5*d, d))  # Bottom right corner

    tform = transform.ProjectiveTransform()

    tform.estimate(np.array((
            top_left,
            (bottom_left[0], image_side - bottom_left[1]),
            (image_side - bottom_right[0], image_side - bottom_right[1]),
            (image_side - top_right[0], top_right[1])
        )), np.array((
            (0, 0),
            (0, image_side),
            (image_side, image_side),
            (image_side, 0)
        )))       

    return tform

#------------------------------------------------------------------------------
  
    
def random_deform(image, rotation, variation):
    '''
    Apply a random warping deformation to the in
    
    '''
    image_side = image.shape[0]
    assert image.shape[0]==image.shape[1]
    cval = 0
    rhom = random_homography(variation, image_side)
    image_warped = transform.rotate(
        image, 
        random.uniform(-rotation, rotation), 
        resize = False,
        mode='constant', 
        cval=cval)
    image_warped = transform.warp(image_warped, rhom, mode='constant', cval=cval)
    return image_warped

  
#------------------------------------------------------------------------------
    
    
