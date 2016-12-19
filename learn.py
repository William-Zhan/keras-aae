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
        # BN(),
        Dense(1000, activation='relu'),
        Dense(784,  activation='sigmoid'),
    ])

x = Input((784,))
z1 = pre_encoder(x)

latent_nodes = np.array(map(lambda l: l(z1), latent_layers))

zs = list(latent_nodes[:,0])
ds = list(latent_nodes[:,1:3].transpose().flatten())
ns = list(latent_nodes[:,3])
# ^^^ [[real,fake],[real,fake],...] -> [real,real...,fake,fake...]
print zs
print ds
def concatenate(tensors):
    import tensorflow as tf
    return tf.concat(1, tensors)

z = Lambda(concatenate)(zs)
d = Lambda(concatenate)(ds)
n = Lambda(concatenate)(ns)

y = decoder(z)

print z
print d

encoder     = Model(x,z)
encoders    = map(lambda (z): Model(x,z), zs)
discriminators = map(lambda l: l.discriminator, latent_layers)
autoencoder = Model(x,y)

noise = Model(x,ns)

aae_r = Model(input=x,output=y)
aae_r.compile(optimizer=Adam(lr=0.001), loss='mse')
aae_d = Model(input=x,output=d)
opt_d = Adam(lr=0.0001)
aae_d.compile(optimizer=opt_d, loss='binary_crossentropy')
aae_g = Model(input=x,output=d)
opt_g = Adam(lr=0.0001)
aae_g.compile(optimizer=opt_g, loss='binary_crossentropy')

def reduceLR(opt,ratio):
    old_lr = float(K.get_value(opt.lr))
    new_lr = 0.7 * old_lr
    K.set_value(opt.lr,new_lr)
    print "Reducing learning rate to {}".format(new_lr)

def set_trainable(net, val):
    net.trainable = val
    if hasattr(net, 'layers'):
        for l in net.layers:
            set_trainable(l, val)

def aae_train (name, epoch=128,computational_effort_factor=8):
    from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
    from keras.utils.generic_utils import Progbar
    from util import mnist, plot_examples
    batch_size = int(epoch * computational_effort_factor)
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,y_train, x_test,y_test = mnist()
    from plot_all import plot_latent
    plot_latent(noise.predict(x_train),np.zeros_like(y_train),"style-noise.png")
    x_train = x_train[:36000,:]   # for removing residuals
    total = x_train.shape[0]
    real_train = np.ones([total,dimensions])
    fake_train = np.zeros([total,dimensions])
    r_loss, d_loss, g_loss = 0.,0.,0.
    try:
        for e in range(epoch):
            d = {'discriminator' : 0, 'generator' : 0}
            for i in range(total//batch_size):
                batch_pb = Progbar(total, width=25)
                def update(force=False):
                    batch_pb.update(min((i+1)*batch_size,total),
                                    [('r',r_loss), ('d',d_loss), ('g',g_loss),
                                     # ('d-g',(d_loss-g_loss))
                                    ], force=force)
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                real_batch = real_train[i*batch_size:(i+1)*batch_size]
                fake_batch = fake_train[i*batch_size:(i+1)*batch_size]
                d_batch = np.concatenate((fake_batch,real_batch),1)
                g_batch = np.concatenate((real_batch,real_batch),1)
                set_trainable(encoder, True)
                set_trainable(decoder, True)
                map(lambda d:set_trainable(d,False), discriminators)
                r_loss = aae_r.train_on_batch(x_batch, x_batch)
                def test():
                    return \
                        aae_d.test_on_batch(x_batch, d_batch), \
                        aae_g.test_on_batch(x_batch, g_batch)
                def train_discriminator():
                    d['discriminator'] += 1
                    set_trainable(encoder, False)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,True), discriminators)
                    return aae_d.train_on_batch(x_batch, d_batch)
                def train_generator():
                    d['generator'] += 1
                    set_trainable(encoder, True)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,False), discriminators)
                    return aae_g.train_on_batch(x_batch, g_batch)
                if e > 20:
                    train_discriminator()
                    train_generator()
                d_loss, g_loss = test()
                update()
            print "Epoch {}/{}: {}".format(e,epoch,[('r',r_loss), ('d',d_loss), ('g',g_loss),
                                                    ('td',d['discriminator']),
                                                    ('tg',d['generator'])])
            if (e % 5) == 0:
                from plot_all import plot_latent, plot_latent_nolimit
                plot_latent(encoders[0].predict(x_test),y_test,"style-test-{}.png".format(e))
                plot_latent_nolimit(encoders[0].predict(x_test),y_test,"style2-test-{}.png".format(e))
    except KeyboardInterrupt:
        print ("learning stopped")

aae_train(name, 10000, 0.1)

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


# how to train the adversarial network?
# train_discriminator()
# train_generator()
# d_loss, g_loss = test()
# update(('d',d_loss))
# idea 1
# for j in range(10):
#     train_discriminator()
#     train_generator()
#     d_loss, g_loss = test()
#     update()

# idea 2
# while abs(d_loss-g_loss) > 0.01:
#     if d_loss > g_loss:
#         train_discriminator()
#     else:
#         train_generator()
#     d_loss, g_loss = test()
#     update()

# idea 3
# while True:
#     d_loss, g_loss = test()
#     if g_loss > 1.0:
#         break
#     train_discriminator()
# while True:
#     if train_generator() < 0.1:
#         break

# idea 4
# while True:
#     if train_discriminator() < 0.1:
#         break
# while True:
#     if train_generator() < 0.1:
#         break
# d_loss, g_loss = test()
# update()

# idea 5
#  and d_loss < 1. and g_loss < 1.

# a = np.concatenate((fake_batch,real_batch),1)
# np.random.shuffle(a)
# return aae_g.train_on_batch(x_batch, a)
# 
# while True:
#     adv_pb = Progbar(1000, width=20)
#     for j in range(10):
#         d_loss, g_loss = test()
#         adv_pb.update(j+1, [('r',r_loss), ('d',d_loss), ('g',g_loss),])
#         if d_loss > g_loss:
#             train_discriminator()
#         else:
#             train_generator()
#     if abs(d_loss-g_loss) < 0.1:
#         print "early stop"
#         break

# idea 6
# # adv_pb = Progbar(1000, width=20)
# for j in range(1000):
#     d_loss = train_discriminator()
#     # adv_pb.update(j+1, [('r',r_loss), ('d',d_loss), ('g',g_loss),])
#     if d_loss < 0.4:
#         break
# # adv_pb = Progbar(1000, width=20)
# for j in range(1000):
#     g_loss = train_generator()
#     # adv_pb.update(j+1, [('r',r_loss), ('d',d_loss), ('g',g_loss),])
#     if g_loss < 0.4:
#         break
