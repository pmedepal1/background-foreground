# Have to train network to see what a patch of cat is and what a patch of grass is, then feed each patch of
# actual image to identify cat or grass

# Although I borrowed only 2 code sections to use here, I wish to 
# include the license that the program creators wrote
# for Copyright purposes
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Code taken from clothes classification example - importing classes and libraries
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# End of shared code 

from google.colab import drive
drive.mount('/content/gdrive')

filein = "train_cat.txt"
%cd /content/gdrive/My\ Drive
train_cat = np.rot90(np.matrix(np.loadtxt('train_cat.txt',delimiter=',')))
train_grass = np.rot90(np.matrix(np.loadtxt('train_grass.txt',delimiter=',')))

train_set = np.append(train_cat,train_grass,axis=0)

# Code taken from clothes example. model is the neural network that is created.
# Two layers are added. The classification program had one input layer, one
# hidden layer that uses a relu activation function, and a softmax layer to 
# output a classification format

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
              
model.fit(train_set, labels, epochs=10)
# End of shared code

test_picture = np.matrix(np.loadtxt('cat_grass_pixels.txt',delimiter=' '))

predictions = model.predict(test_picture)

def mean_absolute_error(Output, Actual):
  diffMatrix = abs(Output - Actual)
  mae = np.mean(diffMatrix)
    
  return mae

# Function to convert predictions array back into an image
def array_to_matrix(numberRows, numberCols, predictions,Y):

  Output = np.zeros((numberRows,numberCols))

  for i in range(numberRows):
    for j in range(numberCols):
      arr_pos = i * numberCols + j # Orientation of predictions is columns, not rows
      probVal = np.argmin(predictions[arr_pos]) # using argmin bc need to flip, as 1 = grass, 0 = cat for predictions, opposite for actual image
      Output[i,j] = probVal
  
  return Output

    
Y = plt.imread('cat_grass.jpg') / 255
numberRows = 368
numberCols = 493
Output = array_to_matrix(numberRows, numberCols, predictions,Y)
plt.imshow(Output*255, cmap = 'gray')

A = plt.imread('truth.png')
Actual = A[0:numberRows,0:numberCols]
#plt.imshow(Actual, cmap = 'gray')
mae = mean_absolute_error(Output, Actual) * 100

print('Mean Absolute Error is %.2f%%' % mae)