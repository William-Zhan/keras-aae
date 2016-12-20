#!/usr/bin/env python

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, Merge
from keras.models import Model
from Custom import Sequential, Residual, Deconvolution2D, Latent
from keras.constraints import maxnorm
from keras import regularizers
from keras.layers.normalization import BatchNormalization as BN
from keras import backend as K
from util import train, train_stack, retrieve_stack, name
from keras.optimizers import Adam, RMSprop, SGD
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
def gaussian_distribution_np (batch_size, latent_dim):
    return np.random.normal(0.,1.,(batch_size, latent_dim))

style = Latent(1,gaussian_distribution_np,'linear')

n_category = 6

def categorical_distribution (z):
    uni = K.random_uniform(shape=(K.shape(z)[0],), low=0, high=n_category, dtype='int32')
    return K.one_hot(uni, n_category)

def categorical_distribution_np (batch_size, latent_dim):
    a = np.random.random_integers(0,latent_dim-1,batch_size)
    b = np.zeros((batch_size, latent_dim))
    b[np.arange(batch_size), a] = 1
    return b

digit = Latent(n_category, categorical_distribution_np, 'softmax')

# latent_layers = [digit,style]
# latent_layers = [style]
latent_layers = [digit]

dimensions = len(latent_layers)

decoder = Sequential(
    (reduce(lambda x, y: x+y, map(lambda x: x.dim, latent_layers)),),
    [
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(784,  activation='sigmoid'),
    ])

x = Input((784,))
z1 = pre_encoder(x)

latent_nodes = np.array(map(lambda l: l(z1), latent_layers))

zs = list(latent_nodes[:,0])
d1s = list(latent_nodes[:,1])
d2s = list(latent_nodes[:,2])
ns = list(latent_nodes[:,3])

def concatenate(tensors):
    import tensorflow as tf
    return tf.concat(1, tensors)

z = Lambda(concatenate)(zs)
d1 = Lambda(concatenate)(d1s)
d2 = Lambda(concatenate)(d2s)
n = Lambda(concatenate)(ns)

y = decoder(z)

encoder     = Model(x,z)
encoders    = map(lambda (z): Model(x,z), zs)
discriminators = map(lambda l: l.discriminator, latent_layers)
autoencoder = Model(x,y)

aae_r = Model(input=x,output=y)
opt_r = Adam(lr=0.001)
aae_r.compile(optimizer=opt_r, loss='mse')
aae_d = Model(input=x,output=d1)
opt_d = Adam(lr=0.001)
aae_d.compile(optimizer=opt_d, loss='binary_crossentropy')
aae_g = Model(input=x,output=d1)
opt_g = Adam(lr=0.001)
aae_g.compile(optimizer=opt_g, loss='binary_crossentropy')
aae_nd = Model(input=ns,output=d2s)
opt_nd = Adam(lr=0.001)
aae_nd.compile(optimizer=opt_nd, loss='binary_crossentropy')
aae_ng = Model(input=ns,output=d2s)
opt_ng = Adam(lr=0.001)
aae_ng.compile(optimizer=opt_ng, loss='binary_crossentropy')

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

def aae_train (name, epoch=1000,batch_size=18000):
    from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
    from keras.utils.generic_utils import Progbar
    from util import mnist, plot_examples
    from plot_all import plot_digit
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,y_train, x_test,y_test = mnist()
    x_train = x_train[:36000,:]   # for removing residuals
    x_plot = x_test[:1000,:]
    y_plot = y_test[:1000]
    total = x_train.shape[0]
    real_train = np.ones([total,dimensions])
    fake_train = np.zeros([total,dimensions])
    r_loss, val_loss, d_loss, g_loss = 0.,0.,0.,0.
    plot_epoch = epoch//200
    pretraining = True
    try:
        pb = Progbar(epoch*(total//batch_size), width=25)
        for e in range(epoch):
            # val_loss = aae_r.evaluate(x_test, x_test)
            if (e % plot_epoch) == 0:
                plot_digit(encoders[0].predict(x_plot),y_plot,"digit-test-{}.png".format(e))
            for i in range(total//batch_size):
                def update():
                    pb.update(e*(total//batch_size)+i, [('r',r_loss), ('val',val_loss), ('d',d_loss), ('g',g_loss),])
                update()
                x_batch = x_train[i*batch_size:(i+1)*batch_size]
                n_batch = np.concatenate([l.sample(batch_size) for l in latent_layers],axis=1)
                real_batch = real_train[i*batch_size:(i+1)*batch_size]
                fake_batch = fake_train[i*batch_size:(i+1)*batch_size]
                def train_autoencoder():
                    set_trainable(encoder, True)
                    set_trainable(decoder, True)
                    map(lambda d:set_trainable(d,False), discriminators)
                    return aae_r.train_on_batch(x_batch, x_batch)
                def train_discriminator():
                    set_trainable(encoder, False)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,True), discriminators)
                    return aae_d.train_on_batch(x_batch, fake_batch) + \
                        aae_nd.train_on_batch(n_batch, real_batch)
                def train_generator():
                    set_trainable(encoder, True)
                    set_trainable(decoder, False)
                    map(lambda d:set_trainable(d,False), discriminators)
                    return aae_g.train_on_batch(x_batch, real_batch) + \
                        aae_ng.train_on_batch(n_batch, real_batch)
                r_loss = train_autoencoder()
                val_loss = aae_r.test_on_batch(x_batch, x_batch)
                if r_loss < 0.02 or not pretraining:
                    if pretraining:
                        pretraining = False
                        print "pretraining finished"
                    d_loss = train_discriminator()
                    g_loss = train_generator()
    except KeyboardInterrupt:
        print ("learning stopped")

aae_train(name, 1000, 1000)

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
