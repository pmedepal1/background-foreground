"""
Praneeth Medepalli
ECE 302: Dr. Stanley Chan
Oct. 11, 2019
Project 1 Code (Parts 1-3)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time



#Testing notes - numpy_matrix.shape returns dimensions of matrix
#print(train_cat.shape)

#mu_cat_test = np.array([])
#for i in range(64):
#    avg = np.mean(train_cat[i])
#    mu_cat_test = np.append(mu_cat_test, avg)
# Doing this but formatting into 64x1 vector instead of 64 element array
#mu_cat = np.mean(train_cat,1)
#cov_cat = np.cov(train_cat)


def my_training(train_cat,train_grass):
    mu_cat = np.mean(train_cat,1)
    mu_grass = np.mean(train_grass,1)
    
    Sigma_cat = np.cov(train_cat)
    Sigma_grass = np.cov(train_grass)
    
    
    return mu_cat, mu_grass, Sigma_cat, Sigma_grass


def my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass):

    detCatRoot = np.linalg.det(Sigma_cat) ** (0.5)
    detGrassRoot = np.linalg.det(Sigma_grass) ** (0.5)
    
    invCat = np.linalg.pinv(Sigma_cat)
    invGrass = np.linalg.pinv(Sigma_grass)
    
    d = 64
    
    factorCat = 1 / ( (2*math.pi)**(d/2) * detCatRoot )
    factorGrass = 1 / ( (2*math.pi)**(d/2) * detGrassRoot )
    
    M = Y.shape[0]
    N = Y.shape[1]

    Output = np.zeros((M-8,N-8))
    
    for i in range(M-8):
        for j in range(N-8):
            z_patch = Y[i:i+8,j:j+8]
            z = z_patch.flatten('F')
            z = z.reshape((64,1))
            vectDiffCat = z - mu_cat
            vectDiffGrass = z - mu_grass
            
            transposeCat = np.transpose(vectDiffCat)
            transposeGrass = np.transpose(vectDiffGrass)
            
            powerCat = ( (-0.5) * transposeCat * invCat * vectDiffCat )[0,0]
            powerGrass = ( (-0.5) * transposeGrass * invGrass * vectDiffGrass )[0,0]
            
            fzcat = factorCat * math.exp(powerCat)
            fzgrass = factorGrass * math.exp(powerGrass)
            
            
            fcatz = K_cat * fzcat
            fgrassz = K_grass * fzgrass
            #fcatz = fzcat
            #fgrassz = K_grass * K_cat * 700 * fzgrass
            #Between 1000 and 1050 is optimal ratio - 6.38% Try <900, try finding optimal
            if fcatz > fgrassz:
                Output[i,j] = 1
                
    return Output

def mean_absolute_error(Output, Actual):
    diffMatrix = abs(Output - Actual)
    mae = np.mean(diffMatrix)
    
    return mae


train_cat = np.matrix(np.loadtxt('data/train_cat.txt',delimiter=','))
train_grass = np.matrix(np.loadtxt('data/train_grass.txt',delimiter=','))

Y = plt.imread('data/cat_grass.jpg') / 255


[mu_cat, mu_grass, Sigma_cat, Sigma_grass] = my_training(train_cat,train_grass)

K_cat = train_cat.shape[1]
K_grass = train_grass.shape[1]

start_time = time.time()
OutputPic = my_testing(Y, mu_cat, mu_grass, Sigma_cat, Sigma_grass, K_cat, K_grass)
print('My runtime is %.2f seconds' % (time.time() - start_time))

plt.imshow(OutputPic*255, cmap = 'gray')

numRows = OutputPic.shape[0]
numCols = OutputPic.shape[1]
A = plt.imread('data/truth.png')
Actual = A[0:numRows,0:numCols]
#plt.imshow(Actual, cmap = 'gray')
mae = mean_absolute_error(OutputPic, Actual) * 100

print('Mean Absolute Error is %.2f%%' % mae)







        
        