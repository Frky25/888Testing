from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

from NoiseShapeLibrary import *

data_set = input_data.read_data_sets('input_data', reshape=False)
images = data_set.test.images
labels = data_set.test.labels

import numpy as np
import matplotlib.pyplot as plt

imagenum = 127
model_parameters = np.load('model_parameters.npy').item()
plt.figure(1)
plt.title("Class label: %d" % 0)
# Your code for the input image here
w = len(images[imagenum][0])
h = len(images[imagenum])
img = images[imagenum].reshape((h,w))
img -= img.flatten().mean()
plt.imshow(img)
    
plt.axis('off')

l1_filter = np.zeros((5, 5, 1, 4))
num_rows = l1_filter.shape[2]
num_cols = l1_filter.shape[3]
plt.figure(2)
test1 = model_parameters['l1_w']
test2 = model_parameters['l1_b']
for x in range(num_cols):
    for y in range(num_rows):
        plt.subplot(num_rows, num_cols, y*num_cols + x + 1)
        w = l1_filter.shape[0]
        h = l1_filter.shape[1]
        img = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                img[i][j] = model_parameters['l1_w'][i][j][y][x]
        plt.imshow(img)
        plt.title("In: %d, Out: %d" % (y, x))
        plt.axis('off')
        
def inside(x, y, w, h):
    return x>=0 and y>=0 and x<w and y<h

def conv(x, W, b):
    in_h = len(x)
    in_w = len(x[0])
    in_c = len(x[0][0])
    conv_h = len(W)
    c_h_s = -(conv_h-1)//2
    conv_w = len(W[0])
    c_w_s = -(conv_h-1)//2
    conv_c = len(W[0][0][0])
    output = np.zeros((in_h,in_w,conv_c))
    for xi in range(in_w):
        for yi in range(in_h):
            for ci in range(conv_c):
                output[yi][xi][ci] = b[ci]
                for ii in range(conv_w):
                    for ji in range(conv_h):
                        for ki in range(in_c):
                            if inside(xi+ii+c_w_s, yi+ji+c_h_s, in_w, in_h):
                                output[yi][xi][ci] += x[yi+ji+c_h_s][xi+ii+c_w_s][ki]*W[ji][ii][ki][ci]
    
    return output

def fc(x, W, b):
    in_l = len(x)
    out_l = len(W[0]) 
    output = np.zeros(out_l)
    for i in range(out_l):
        output[i] = b[i]
        for j in range(in_l):
            output[i] += x[j]*W[j][i]
    
    return output

def relu(x):
    if isinstance(x,np.ndarray):#recurse
        output = np.zeros_like(x)
        for i in range(len(x)):
            output[i] = relu(x[i])
        return output
    else:#just a number
        if x<0:
            return 0
        else:
            return x   

def pool2(x, dh, dw):
    in_h = len(x)
    in_w = len(x[0])
    in_c = len(x[0][0])
    out_h = in_h//dh
    out_w = in_w//dw
    output = np.zeros((out_h,out_w,in_c))
    for c in range(in_c):
        for i in range(out_h):
            for j in range(out_w):
                yi = i*dh
                xi = j*dw
                output[i][j][c] = max(x[yi][xi][c],x[yi+1][xi][c],x[yi][xi+1][c],x[yi+1][xi+1][c])
    
    return output

def flatten(x):
    output = x.flatten()
    
    return output

def cutoff(x, cutoff):
    num_chan = x.shape[2]
    w = x.shape[0]
    h = x.shape[1]
    img = np.zeros((h,w,num_chan))
    for k in range(num_chan):
        for i in range(h):
            for j in range(w):
                if x[i][j][k] > cutoff[k]:
                    img[i][j][k] = 1.0
                else:
                    img[i][j][k] = -1.0
    return img

def lp(x):
    num_chan = x.shape[2]
    w = x.shape[0]
    h = x.shape[1]
    im = np.zeros((h,w))
    imd = decimate(im)
    w = imd.shape[0]
    h = imd.shape[1]
    imd = np.zeros((h,w))
    img = np.zeros((h,w,num_chan))
    for k in range(num_chan):
        w = x.shape[0]
        h = x.shape[1]
        for i in range(h):
            for j in range(w):
                im[i][j] = x[i][j][k]
        imd = decimate(lpf(im))
        w = imd.shape[0]
        h = imd.shape[1]
        for i in range(h):
            for j in range(w):
                img[i][j][k] = imd[i][j]
        
    return img
    

plt.figure(3)
img = images[imagenum]
img -= img.flatten().mean()
b=model_parameters['l1_b']
l1 = conv(img, model_parameters['l1_w'], model_parameters['l1_b'])
res = relu(l1)
num_chan = res.shape[2]
for x in range(num_chan):
    plt.subplot(num_rows, num_cols, x + 1)
    w = res.shape[0]
    h = res.shape[1]
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = res[i][j][x]
    mi = min(img.flatten())
    ma = max(img.flatten())
    print(str(x)+' '+str(mi)+' ' + str(ma))
    plt.imshow(img)
    plt.title("In: %d, Out: %d" % (y, x))
    plt.axis('off')
    
    
img = images[imagenum].reshape((h,w))
img -= img.flatten().mean()
im_large = upsampleAndInterpolate(img)
shaped = noiseShape(im_large,0.5,0)
imshaped = shaped.reshape((55,55,1))
plt.figure(4)
matplotlib.pyplot.imshow(shaped)
plt.figure(5)
w = model_parameters['l1_w'].shape[0]
h = model_parameters['l1_w'].shape[1]
num_chan = model_parameters['l1_w'].shape[3]
l1_filter = np.zeros((9, 9, 1, 4))
print(test2[0])
print(test2[1])
print(test2[2])
print(test2[3])
for x in range(num_chan):
    plt.subplot(num_rows, num_cols, x + 1)
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = model_parameters['l1_w'][i][j][0][x]
    im_large = upsampleAndInterpolate(img)
    shaped = noiseShape(im_large,0.5,0)
    matplotlib.pyplot.imshow(shaped)
    for i in range(9):
        for j in range(9):
            l1_filter[i][j][0][x] = shaped[i][j]
res = conv(imshaped, l1_filter, np.zeros(4))
cut = (-b)*10+5
print(cut)
res = cutoff(res,cut)
res = lp(res)*4
res = relu(res)
plt.figure(6)
w = res.shape[0]
h = res.shape[1]
for x in range(num_chan):
    plt.subplot(num_rows, num_cols, x + 1)
    w = res.shape[0]
    h = res.shape[1]
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = res[i][j][x]
    plt.imshow(img)
    mi = min(img.flatten())
    ma = max(img.flatten())
    print(str(x)+' '+str(mi)+' ' + str(ma))
    plt.title("In: %d, Out: %d" % (y, x))
    plt.axis('off')