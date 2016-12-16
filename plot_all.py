#!/usr/bin/env python

import keras
from Custom import Deconvolution2D

keras.layers.Deconvolution2D = Deconvolution2D

import keras.models
import keras.backend as K
import numpy as np
from util import name, latent_dim
import matplotlib.pyplot as plt
import tensorflow as tf

pre_encoder = keras.models.load_model(name+'/pre.h5')
encoder = keras.models.load_model(name+'/encoder.h5')
encoder_style = keras.models.load_model(name+'/encoder0.h5')
encoder_digit = keras.models.load_model(name+'/encoder1.h5')
discriminator_style = keras.models.load_model(name+'/discriminator0.h5')
discriminator_digit = keras.models.load_model(name+'/discriminator1.h5')
decoder = keras.models.load_model(name+'/decoder.h5')
autoencoder = keras.models.load_model(name+'/model.h5')

from util import mnist
x_train,y_train, x_test,y_test = mnist()

def plot_grid(images,name="plan.png"):
    l = len(images)
    w = 6
    h = max(l//6,1)
    plt.figure(figsize=(20, h*2))
    for i,image in enumerate(images):
        # display original
        ax = plt.subplot(h,w,i+1)
        plt.imshow(image.reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)

x_test_reconstructed = autoencoder.predict(x_test)
n = 24  # how many digits we will display
x = np.array([x_test,x_test_reconstructed])[:,:n] # 2,24,784
x = x.reshape((2,4,6,784))      # divide by 6 rows (4 groups)
x = np.einsum('igjp->gijp', x)  # 4,2,6,784
x = x.reshape((48,784))
plot_grid(x,"autoencoding.png")

def plot_latent(latent,color,name,size=(6,6)):
    plt.figure(figsize=size)
    plt.scatter(latent[:, 0], latent[:, 1], c=color)
    plt.colorbar()
    plt.savefig(name)

style_test = encoder_style.predict(x_test)
plot_latent(style_test,y_test,"style.png")

style_train = encoder_style.predict(x_train)
plot_latent(style_train,y_train,"style-train.png")

digit_test = encoder_digit.predict(x_test)

print digit_test[:10]

result_test = discriminator_digit.predict(digit_test)

print result_test[:10]

labels_test = np.argmax(digit_test,1)

added = np.einsum('xb->bx',np.array([style_test[:,0] + labels_test*10, style_test[:,1]]))

plot_latent(added,y_test,"style-label.png",(60,6))



# if latent_dim == 2:
#     from scipy.stats import norm
#     # display a 2D manifold of the digits
#     n = 30  # figure with 15x15 digits
#     digit_size = 28
#     figure = np.zeros((digit_size * n, digit_size * n))
#     grid_x = np.linspace(z_train[:,0].min(), z_train[:,0].max(), n)
#     grid_y = np.linspace(z_train[:,1].min(), z_train[:,1].max(), n)
#     for i, x in enumerate(grid_x):
#         for j, y in enumerate(grid_y):
#             z_sample = np.tile(np.array([[x, y]]), 1).reshape(1, 2)
#             # print z_sample
#             x_decoded = decoder.predict(z_sample, batch_size=1)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 j * digit_size: (j + 1) * digit_size,
#                 i * digit_size: (i + 1) * digit_size,] = digit
#     plt.figure(figsize=(10, 10))
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig("manifold.png")

