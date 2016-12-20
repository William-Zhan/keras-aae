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

def plot_latent(latent,color,name,size=(6,6)):
    plt.figure(figsize=size)
    plt.scatter(latent[:, 0], latent[:, 1], c=color)
    axes = plt.gca()
    axes.set_xlim([-10,10])
    axes.set_ylim([-10,10])
    plt.colorbar()
    plt.savefig(name)

def plot_latent_nolimit(latent,color,name,size=(6,6)):
    plt.figure(figsize=size)
    plt.scatter(latent[:, 0], latent[:, 1], c=color)
    axes = plt.gca()
    plt.colorbar()
    plt.savefig(name)

def plot_digit(digit,color,name):
    max_label=digit.shape[1]
    plt.figure(figsize=(30,10))
    for i in range(max_label):
        plt.scatter(((np.random.random_sample(digit.shape[0])-0.5)*0.9+i),
                    digit[:,i],
                    # s=2,
                    c=color)
    axes = plt.gca()
    plt.colorbar()
    plt.savefig(name)
    
if __name__ == '__main__':

    pre_encoder = keras.models.load_model(name+'/pre.h5')
    encoder = keras.models.load_model(name+'/encoder.h5')
    decoder = keras.models.load_model(name+'/decoder.h5')
    autoencoder = keras.models.load_model(name+'/model.h5')

    from util import mnist
    x_train,y_train, x_test,y_test = mnist()

    encoder_digit = keras.models.load_model(name+'/encoder0.h5')
    discriminator_digit = keras.models.load_model(name+'/discriminator0.h5')

    digit_test = encoder_digit.predict(x_test)

    print digit_test[:10]
    print np.sum(digit_test,axis=1)

    result_test = discriminator_digit.predict(digit_test)
    
    print result_test[:10]

    plot_digit(digit_test,y_test,"digit-test.png")
    
    x_test_reconstructed = autoencoder.predict(x_test)
    n = 24  # how many digits we will display
    x = np.array([x_test,x_test_reconstructed])[:,:n] # 2,24,784
    x = x.reshape((2,4,6,784))      # divide by 6 rows (4 groups)
    x = np.einsum('igjp->gijp', x)  # 4,2,6,784
    x = x.reshape((48,784))
    plot_grid(x,"autoencoding.png")
