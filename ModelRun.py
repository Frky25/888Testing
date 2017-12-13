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

model_parameters = np.load('model_parameters.npy').item()

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

l1_filter = np.zeros((5, 5, 1, 4))
h = l1_filter.shape[0]
w = l1_filter.shape[1]
num_rows = l1_filter.shape[2]
num_cols = l1_filter.shape[3]
for x in range(num_rows):
    for y in range(num_cols):
        img = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                img[i][j] = model_parameters['l1_w'][i][j][x][y]
        im_large = upsampleAndInterpolate(img)
        shaped = noiseShape(im_large,0.5,0)
        for i in range(h):
            for j in range(w):
                l1_filter[i][j][x][y] = shaped[i][j]
                
b = model_parameters['l1_b']
cut = (-b)*10+5
                
def run_inference(image):
    # Your code for defining the correct network topology here
    h = image.shape[0]
    w = image.shape[1]
    img = image.reshape((h,w))
    img -= img.flatten().mean()
    im_large = upsampleAndInterpolate(img)
    shaped = noiseShape(im_large,0.5,0)
    h = shaped.shape[0]
    w = shaped.shape[1]
    imshaped = shaped.reshape((h,w,1))
    l1 = conv(imshaped, l1_filter, np.zeros(4))
    l1_cut = cutoff(l1, cut)
    l1_lpf = lp(l1_cut)*4.0
    l2 = relu(l1_lpf)
    l3 = pool2(l2, 2, 2)
    l4 = conv(l3, model_parameters['l4_w'], model_parameters['l4_b'])
    l5 = relu(l4)
    l6 = pool2(l5, 2, 2)
    l7 = fc(flatten(l6), model_parameters['l7_w'], model_parameters['l7_b'])
    l8 = relu(l7)
    l9 = fc(l8, model_parameters['l9_w'], model_parameters['l9_b'])
    output_class = np.argmax(l9)
    return output_class

def run_model(image):
    # Your code for defining the correct network topology here
    # img = image - image.flatten().mean()
    l1 = conv(image, model_parameters['l1_w'], model_parameters['l1_b'])
    l2 = relu(l1)
    l3 = pool2(l2, 2, 2)
    l4 = conv(l3, model_parameters['l4_w'], model_parameters['l4_b'])
    l5 = relu(l4)
    l6 = pool2(l5, 2, 2)
    l7 = fc(flatten(l6), model_parameters['l7_w'], model_parameters['l7_b'])
    l8 = relu(l7)
    l9 = fc(l8, model_parameters['l9_w'], model_parameters['l9_b'])
    output_class = np.argmax(l9)
    return output_class

icorrect, total = (0, 1000)
for n in range(min(total, int(images.shape[0]))):
    inference = run_inference(images[n])
    if labels[n] == inference:
        icorrect += 1
    if n%25 == 0:
        print("%d: Label: %d, Inference: %d" % (n, labels[n], inference))
print("IAccuracy: %d/%d (%.2g)" % (icorrect, total, float(icorrect)/total))


