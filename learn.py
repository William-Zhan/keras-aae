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
    return K.random_normal(shape=K.shape(z), mean=0., std=1.)

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
# ^^^ [[real,fake],[real,fake],...] -> [real,real...,fake,fake...]
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
aae_d.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
aae_g = Model(input=x,output=d)
aae_g.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')

def set_trainable(net, val):
    net.trainable = val
    if hasattr(net, 'layers'):
        for l in net.layers:
            set_trainable(l, val)

def aae_train (name, epoch=128,computational_effort_factor=8,noise=False):
    from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
    from keras.utils.generic_utils import Progbar
    from util import mnist, plot_examples
    batch_size = epoch * computational_effort_factor
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,_, x_test,_ = mnist()
    total = x_train.shape[0]
    if noise:
        x_input = add_noise(x_train)
    else:
        x_input = x_train
    real_train = np.ones([total,dimensions])
    fake_train = np.zeros([total,dimensions])
    r_loss, d_loss, g_loss = 0.,0.,0.
    try:
        for e in range(epoch):
            for i in range(total//batch_size):
                d = {'teach' : 0, 'deceive' : 0}
                batch_pb = Progbar(total, width=25)
                def update():
                    batch_pb.update(min((i+1)*batch_size,total),
                                    [('r',r_loss), ('d',d_loss), ('g',g_loss)])
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                real_batch = real_train[i*batch_size:(i+1)*batch_size]
                fake_batch = fake_train[i*batch_size:(i+1)*batch_size]
                set_trainable(encoder, True)
                set_trainable(decoder, True)
                map(lambda d:set_trainable(d,False), discriminators)
                r_loss = aae_r.train_on_batch(x_batch, x_batch)
                def teach():
                    d['teach'] += 1
                    set_trainable(encoder, False)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,True), discriminators)
                    d_loss = aae_d.train_on_batch(
                        x_batch, np.concatenate((real_batch,fake_batch),1))
                def deceive():
                    d['deceive'] += 1
                    set_trainable(encoder, True)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,False), discriminators)
                    g_loss = aae_g.train_on_batch(
                        x_batch, np.concatenate((real_batch,real_batch),1))
                teach()
                deceive()
                update()
                while abs(d_loss - g_loss) > 0.01 :
                    teach()
                    deceive()
                    update()
            print "\nEpoch {}/{}: {}".format(e,epoch,[('r',r_loss), ('d',d_loss), ('g',g_loss),
                                                      ('nt',d['teach']), ('nd',d['deceive'])])
    except KeyboardInterrupt:
        print ("learning stopped")

aae_train(name, 1000, 4)

pre_encoder.save(name+"/pre.h5")
autoencoder.save(name+"/model.h5")
encoder.save(name+"/encoder.h5")
decoder.save(name+"/decoder.h5")
for i, e in enumerate(encoders):
    e.save(name+"/encoder"+str(i)+".h5")
for i, e in enumerate(discriminators):
    e.save(name+"/discriminator"+str(i)+".h5")

# from util import plot_examples
# plot_examples(name,autoencoder,x_test)
