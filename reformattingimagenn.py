"""
Praneeth Medepalli
ECE 302: Dr. Stanley Chan
Oct. 11, 2019
Project 1 Code (Reformatting test image to create and flatten patches
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image

def patchMatrixGenerator(patches):
    numPatches = patches.shape[0]

    output = np.empty([numPatches,64])
    
    for i in range(numPatches):
        z = patches[i].flatten('F')
        output[i,:] = z
    
    return output


train_cat = np.matrix(np.loadtxt('data/train_cat.txt',delimiter=','))
train_grass = np.matrix(np.loadtxt('data/train_grass.txt',delimiter=','))

Y = plt.imread('data/cat_grass.jpg') / 255

#one_image = load_sample_image('')
patches = image.extract_patches_2d(Y, (8, 8))
output = patchMatrixGenerator(patches)

np.savetxt('data/cat_grass_pixels.txt',output)

#outMatrix = patchMatrixGenerator(Y)