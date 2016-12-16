from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Flatten
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping

def mnist (labels = range(0,6)):
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    def conc (x,y):
        return np.concatenate((y.reshape([len(y),1]),x),axis=1)
    def select (x,y):
        selected = np.array([elem for elem in conc(x, y) if elem[0] in labels])
        return np.delete(selected,0,1), np.delete(selected,np.s_[1::],1).flatten()
    x_train, y_train = select(x_train, y_train)
    x_test, y_test = select(x_test, y_test)
    return x_train, y_train, x_test, y_test

def pow2 (x):
    return 2**x

def add_noise(x,noise_factor = 0.5):
    x += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
    return np.clip(x, 0., 1.)

def train (model, name, epoch=pow2(7),computational_effort_factor=8,noise=False):
    batch_size = epoch * computational_effort_factor
    print("epoch: {0}, batch: {1}".format(epoch, batch_size))
    x_train,_, x_test,_ = mnist()
    if noise:
        x_input = add_noise(x_train)
    else:
        x_input = x_train
    
    hf5 = "{0}/model.h5".format(name)
    model.fit(x_input, x_train,
              nb_epoch=epoch,
              batch_size=(batch_size//1),
              shuffle=True,
              validation_data=(x_test, x_test),
              callbacks=[TensorBoard(log_dir="{0}".format(name)),
                         CSVLogger("{0}/log.csv".format(name),append=True),
                         EarlyStopping(
                             monitor='loss',
                             patience=6,verbose=1,mode='min',min_delta=0.0001),
                         ReduceLROnPlateau(
                             monitor='loss',
                             factor=0.7,
                             patience=3,verbose=1,mode='min',epsilon=0.0001)
              ])
    model.save(hf5)
    plot_examples(name,model,x_test)
    
def plot_examples (name,model,x_test):
    # encode and decode some digits
    # note that we take them from the *test* set
    decoded_imgs = model.predict(x_test)

    # use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("{0}/examples.png".format(name))

def retrieve_stack(encoder,decoder,i,name="/tmp/"):
    x_in = Input(shape=encoder.layers[0].get_input_shape_at(0)[1:])
    h = x_in
    for l in encoder.layers[0:i]:
        h = l(h)
    partial_encoder = Model(x_in,h)
    stack_in = Input(shape=partial_encoder.get_output_shape_at(0)[1:])
    stack_out = decoder.layers[-i-1](encoder.layers[i](stack_in))
    stack = Model(input=stack_in,output=stack_out)
    stack.summary()
    x_in = Input(shape=partial_encoder.get_output_shape_at(0)[1:])
    h = x_in
    for l in [ decoder.layers[j] for j in range(-i,0) ]:
        h = l(h)
    partial_decoder = Model(x_in,h)
    return stack,partial_encoder,partial_decoder

def prefit(model,x_train,y_train,x_test,y_test,threshold):
    model.fit(x_train,y_train,
              nb_epoch=3,
              batch_size=256,
              shuffle=True,
              validation_data=(x_test, y_test))
    loss = model.evaluate(x_train,y_train)
    if loss>threshold:
        print ("Detected an unfortunate weights. Reinitializing...")
        reinitialize(model)
        prefit(model,x_train,y_train,x_test,y_test,threshold)

def reinitialize(layer):
    from keras.engine.topology import Container
    if isinstance(layer,Container):
        # print layer
        map(reinitialize,layer.layers)
        # print layer
    else:
        old_w = layer.weights
        print "from",layer.get_weights()
        layer.build(layer.input_shape)
        # print layer
        # print old_w
        # print layer.weights
        print "to",layer.get_weights()
        # print ("{0}->{1}".format(old_w,layer.weights))


def train_stack(encoder,decoder,i,name="/tmp/"):
    print "Generating Stack #{0}".format(i)
    stack,partial_encoder,partial_decoder = retrieve_stack(encoder,decoder,i,name)
    x_train,_, x_test,_ = mnist_0123()
    m_train = partial_encoder.predict(x_train)
    m_test = partial_encoder.predict(x_test)
    print "Pretraining Stack #{0} (inner layer)".format(i)
    stack.compile(optimizer=Adam(lr=0.004), loss='mse')
    # prefit(stack,m_train,m_train,m_test,m_test,0.05)
    stack.fit(m_train,m_train,
              nb_epoch=1024,
              batch_size=256,
              shuffle=True,
              validation_data=(m_test, m_test),
              callbacks=[TensorBoard(log_dir="{0}".format(name)),
                         CSVLogger("{0}/log.csv".format(name),append=True),
                         EarlyStopping(
                             patience=6,verbose=1,mode='min',min_delta=0.0002),
                         ReduceLROnPlateau(
                             factor=0.7,
                             patience=3,verbose=1,mode='min',epsilon=0.0002)
              ])
    print "Pretraining Stack #{0} (inner + upper layer)".format(i)
    partial = Sequential([partial_encoder,stack,partial_decoder])
    partial.compile(optimizer=Adam(lr=0.001), loss='mse')
    partial.fit(x_train,x_train,
                nb_epoch=1024,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir="{0}".format(name)),
                           CSVLogger("{0}/log.csv".format(name),append=True),
                           EarlyStopping(
                               patience=6,verbose=1,mode='min',min_delta=0.0002),
                           ReduceLROnPlateau(
                               factor=0.7,
                               patience=3,verbose=1,mode='min',epsilon=0.0002)
                ])
    return stack,partial_encoder,partial_decoder

name = "learn"
latent_dim = 2
precision = 16
