# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:20:10 2022

@author: Leng
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Xrange,Yrange = 128,128
X, Y = np.linspace(0, Xrange-1,Xrange,dtype=int), np.linspace(0, Yrange-1,Yrange,dtype=int)

model = tf.keras.models.load_model(r".\Images\ModelSaved-without6789")

#model.summary()

layer_outputs = [layer.output for layer in model.layers[1:2]]
activation_model =tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

FieldSource = np.load('.\\Images\\validation_data\\Source\\source0.npy')   
FieldTarget = np.load('.\\Images\\validation_data\\Target\\target0.npy')   
FieldSource = FieldSource.reshape(-1,Xrange,Yrange,5)
activations = activation_model.predict(FieldSource) 

FieldTargetAmp = FieldTarget[:,:,0]
FieldTargetPha = FieldTarget[:,:,1]


layer_names = []
for layer in model.layers[1:2]:
    layer_names.append(layer.name)
    
#images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    for feature in range(n_features):
        channel_image = layer_activation[:,:,feature]
        plt.contourf(X,Y,channel_image, 50, cmap='rainbow')
        #plt.title("display")
        plt.show()
    #size = layer_activation.shape[1]
    #n_cols = n_features//images_per_row
    # display_grid = np.zeros((size*n_cols, images_per_row*size))
    # for col in range(n_cols):
    #     for row in range(images_per_row):
    #         channel_image = layer_activation[0,:,:,col*images_per_row+row]
    #         display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channel_image
    #         scale = 1./size
    #         plt.figure(figsize = (scale*display_grid.shape[1],scale*display_grid.shape[0]))
    #         plt.title(layer_name)
    #         plt.grid(False)
    #         #plt.imshow(display_grid, aspect = 'auto',cmap = 'viridis')
            