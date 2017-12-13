import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot

# Upsamples the image and quantizes it to 1 bit. Uses linear interpolation.
# Thresh sets the threshold for the quantization
def upsampleAndQuantize(im, thresh):
    shape = im.shape
    large = np.zeros((shape[0]*2,shape[1]*2),bool)
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            large[2*i,2*j] = im[i,j]>thresh;
            if(i<shape[0]-1 and j<shape[1]-1):
                large[2*i,2*j+1] = (im[i,j]+im[i,j+1])>(2*thresh)
                large[2*i+1,2*j] = (im[i,j]+im[i+1,j])>(2*thresh)
                large[2*i+1,2*j+1] = (im[i,j]+im[i+1,j+1]+im[i+1,j]+im[i,j+1])>(4*thresh)
    return large

# Upsamples the image by 2 in each dimension using linear interpolation
def upsampleAndInterpolate(im):
    shape = im.shape
    large = np.zeros((shape[0]*2,shape[1]*2))
    for i in range(0,shape[0]):
        for j in range(0,shape[1]):
            large[2*i,2*j] = im[i,j];
            if(i<shape[0]-1 and j<shape[1]-1):
                large[2*i,2*j+1] = (im[i,j]+im[i,j+1])/2
                large[2*i+1,2*j] = (im[i,j]+im[i+1,j])/2
                large[2*i+1,2*j+1] = (im[i,j]+im[i+1,j+1]+im[i+1,j]+im[i,j+1])/4
    return large

# Completes a simple noise shaping operation
def noiseShape(im,val,thresh):
    quantized = np.zeros((im.shape[0]+2,im.shape[1]+2))
    middle_mat = np.zeros((im.shape[0]+2,im.shape[1]+2))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            middle_mat[i+1,j+1] = (im[i,j]-(val*quantized[i,j+1])/2.-(val*quantized[i+1,j])/2.)+middle_mat[i,j+1]/2.+middle_mat[i+1,j]/2.
            if middle_mat[i+1,j+1]>thresh:
                quantized[i+1,j+1] = 1;
            else:
                quantized[i+1,j+1] = -1;
    return quantized[1:quantized.shape[0]-2,1:quantized.shape[0]-2]

# Applies a low pass filter to the image
def lpf(im):
    lrg = np.zeros((im.shape[0]+4,im.shape[1]+4))
    lrg[2:im.shape[0]+2,2:im.shape[0]+2] = im;
    fin = np.zeros((im.shape[0],im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            fin[i,j] = np.sum(lrg[i:i+4,j:j+4])/25.
    return fin

# Removes every other value in the image
def decimate(im):
    sm = im[0::2,0::2]
    return sm

#im_disp = matplotlib.pyplot.imshow(im_large)
#matplotlib.pyplot.show()
