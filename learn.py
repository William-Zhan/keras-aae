#!/usr/bin/env python

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, Merge
from keras.models import Model
from Custom import Sequential, Residual, Deconvolution2D, Latent
from keras.constraints import maxnorm
from keras import regularizers
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from util import train, train_stack, retrieve_stack, name
import numpy as np
import os
import tensorflow as tf

conv_common = {'activation':'relu', 'border_mode':'same'}

pre_encoder = Sequential(
    (784,),
    [
        Dense(3000, activation='relu'),
        BN(),
        Dense(3000, activation='relu'),
        BN(),
    ])

def gaussian_distribution (z):
    return K.random_normal(shape=K.shape(z), mean=0., std=0.1)

style = Latent(2,gaussian_distribution,'linear')

def categorical_distribution (z):
    uni = K.random_uniform(shape=(K.shape(z)[0],), low=0, high=6, dtype='int32')
    return K.one_hot(uni, 6)

digit = Latent(6, categorical_distribution, 'sigmoid')

latent_layers = [style,digit]

dimensions = len(latent_layers)

decoder = Sequential(
    (reduce(lambda x, y: x+y, map(lambda x: x.dim, latent_layers)),),
    [
        Dense(3000, activation='relu'),
        BN(),
        Dense(3000, activation='relu'),
        Dense(784,  activation='sigmoid'),
    ])

x = Input((784,))
z1 = pre_encoder(x)

latent_nodes = np.array(map(lambda l: l(z1), latent_layers))

zs  = list(latent_nodes[:,0])
d1s = tuple(latent_nodes[:,1])
d2s = tuple(latent_nodes[:,2])

def concatenate(zs):
    import tensorflow as tf
    return tf.concat(1, zs)

z  = Lambda(concatenate)(zs)

y = decoder(z)

encoder     = Model(x,z)
encoders    = map(lambda (z): Model(x,z), zs)
discriminators = map(lambda l: l.discriminator, latent_layers)
autoencoder = Model(x,y)
aae = Model(input=x,output=(y,)+d1s+d2s)

from keras.objectives import binary_crossentropy
def bc(weight):
    return lambda x,y: weight * binary_crossentropy(x,y)

from keras.optimizers import Adam, RMSprop
aae.compile(optimizer=RMSprop(lr=0.0001),
            loss=list(('mse',) +
                      tuple([bc( 1)]*(2*dimensions))))
aae.summary()

def aae_train (model, name, epoch=128,computational_effort_factor=8,noise=False):
    from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
    from util import mnist, plot_examples
    batch_size = epoch * computational_effort_factor
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,_, x_test,_ = mnist()
    if noise:
        x_input = add_noise(x_train)
    else:
        x_input = x_train
    try:
        model.fit(x_input,
                  list((x_train,) +
                       tuple(map(lambda x: np.ones([x_input.shape[0],1]),  range(dimensions))) +
                       tuple(map(lambda x: np.zeros([x_input.shape[0],1]), range(dimensions)))),
                  nb_epoch=epoch,
                  batch_size=(batch_size//1),
                  shuffle=True,
                  callbacks=[TensorBoard(log_dir="{0}".format(name)),
                             CSVLogger("{0}/log.csv".format(name),append=True),
                             EarlyStopping(
                                 monitor='loss',
                                 patience=20,verbose=1,mode='min',min_delta=0.0001),
                             ReduceLROnPlateau(
                                 monitor='loss',
                                 factor=0.7,
                                 patience=3,verbose=1,mode='min',epsilon=0.0001)
                  ])
    except KeyboardInterrupt:
        print ("learning stopped")
    plot_examples(name,autoencoder,x_test)

aae_train(aae, name, 1024, 8)
aae.compile(optimizer=Adam(lr=0.01),
            loss=list(('mse',) + tuple(['binary_crossentropy']*(2*dimensions))))

pre_encoder.save(name+"/pre.h5")
autoencoder.save(name+"/model.h5")
encoder.save(name+"/encoder.h5")
decoder.save(name+"/decoder.h5")
for i, e in enumerate(encoders):
    e.save(name+"/encoder"+str(i)+".h5")
for i, e in enumerate(discriminators):
    e.save(name+"/discriminator"+str(i)+".h5")
