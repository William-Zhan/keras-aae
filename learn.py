#!/usr/bin/env python

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, Merge
from keras.models import Model
from Custom import Sequential, Residual, Deconvolution2D, Latent
from keras.constraints import maxnorm
from keras import regularizers
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from util import train, train_stack, retrieve_stack, name
from keras.optimizers import Adam, RMSprop
import numpy as np
import os
import tensorflow as tf

conv_common = {'activation':'relu', 'border_mode':'same'}

pre_encoder = Sequential(
    (784,),
    [
        Dense(1000, activation='relu'),
        BN(),
        Dense(1000, activation='relu'),
        BN(),
    ])

def gaussian_distribution (z):
    return K.random_normal(shape=K.shape(z), mean=0., std=0.1)

style = Latent(2,gaussian_distribution,'linear')

def categorical_distribution (z):
    uni = K.random_uniform(shape=(K.shape(z)[0],), low=0, high=6, dtype='int32')
    return K.one_hot(uni, 6)

digit = Latent(6, categorical_distribution, 'sigmoid')

# latent_layers = [style,digit]
latent_layers = [style]

dimensions = len(latent_layers)

decoder = Sequential(
    (reduce(lambda x, y: x+y, map(lambda x: x.dim, latent_layers)),),
    [
        Dense(1000, activation='relu'),
        BN(),
        Dense(1000, activation='relu'),
        Dense(784,  activation='sigmoid'),
    ])

x = Input((784,))
z1 = pre_encoder(x)

latent_nodes = np.array(map(lambda l: l(z1), latent_layers))

zs = list(latent_nodes[:,0])
ds = list(latent_nodes[:,1:].transpose().flatten())
print zs
print ds
def concatenate(tensors):
    import tensorflow as tf
    return tf.concat(1, tensors)

z = Lambda(concatenate)(zs)
y = decoder(z)
d = Lambda(concatenate)(ds)

print z
print d

encoder     = Model(x,z)
encoders    = map(lambda (z): Model(x,z), zs)
# discriminator  = Model(z,d)
# discriminators = map(lambda (z,d): Model(z,d), zs, ds)
discriminators = map(lambda l: l.discriminator, latent_layers)
autoencoder = Model(x,y)

from keras.objectives import binary_crossentropy
def bc(weight):
    return lambda x,y: weight * binary_crossentropy(x,y)


aae_r = Model(input=x,output=y)
aae_r.compile(optimizer=Adam(lr=0.001), loss='mse')
aae_d = Model(input=x,output=d)
aae_d.compile(optimizer=Adam(lr=0.001), loss=bc(1))
aae_g = Model(input=x,output=d)
aae_g.compile(optimizer=Adam(lr=0.001), loss=bc(-1))

def aae_train (name, epoch=128,computational_effort_factor=8,noise=False):
    from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
    from keras.utils.generic_utils import Progbar
    from util import mnist, plot_examples
    batch_size = epoch * computational_effort_factor
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,_, x_test,_ = mnist()
    batch_pb = Progbar(x_train.shape[0], width=25)
    epoch_pb = Progbar(epoch,            width=25)
    if noise:
        x_input = add_noise(x_train)
    else:
        x_input = x_train
    d_train = np.concatenate((np.ones([x_input.shape[0],dimensions]),
                              np.zeros([x_input.shape[0],dimensions])),axis=1)
    d_train2 = np.zeros([x_input.shape[0],dimensions*2])
    try:
        def set_trainable(models,flag):
            for m in models:
                m.trainable=flag
        for e in range(epoch):
            for i in range(x_train.shape[0]//batch_size):
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                d_batch = d_train[i*batch_size:(i+1)*batch_size]
                d_batch2 = d_train2[i*batch_size:(i+1)*batch_size]
                encoder.trainable, decoder.trainable = True, True
                set_trainable(discriminators,False)
                r_loss = aae_r.train_on_batch(x_batch, x_batch)
                # 
                encoder.trainable, decoder.trainable = False, False
                set_trainable(discriminators,True)
                d_loss = aae_d.train_on_batch(x_batch, d_batch)
                # 
                encoder.trainable, decoder.trainable = True, False
                set_trainable(discriminators,False)
                g_loss = aae_g.train_on_batch(x_batch, d_batch2)
                #
                losses = [('r',r_loss),
                          ('d',d_loss),
                          ('g',-g_loss),
                          ('d-g',d_loss+g_loss),
                          ('r+d-g',r_loss+d_loss+g_loss)]
                batch_pb.update(i*batch_size,losses)
            print "\nEpoch {}/{}: {}".format(e,epoch,losses)
    except KeyboardInterrupt:
        print ("learning stopped")
    plot_examples(name,autoencoder,x_test)

aae_train(name, 1024, 2)

pre_encoder.save(name+"/pre.h5")
autoencoder.save(name+"/model.h5")
encoder.save(name+"/encoder.h5")
decoder.save(name+"/decoder.h5")
for i, e in enumerate(encoders):
    e.save(name+"/encoder"+str(i)+".h5")
for i, e in enumerate(discriminators):
    e.save(name+"/discriminator"+str(i)+".h5")
