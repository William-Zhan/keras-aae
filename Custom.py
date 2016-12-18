# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model

def Sequential (input_shape, array):
    x = Input(input_shape)
    h = x
    for layer in array:
        h = layer(h)
    return Model(x,h)

def Residual (layer):
    x = Input(layer.get_input_shape_at(0))
    r = layer(x)
    f = Lambda(lambda (x,r): x+r)(x,r)
    return Model(x,f)

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils.np_utils import conv_output_length, conv_input_length
from keras.layers import Convolution2D

class Deconvolution2D(Convolution2D):
    '''Transposed convolution operator for filtering windows of two-dimensional inputs.
    The need for transposed convolutions generally arises from the desire
    to use a transformation going in the opposite direction of a normal convolution,
    i.e., from something that has the shape of the output of some convolution
    to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with said convolution. [1]

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    To pass the correct `output_shape` to this layer,
    one could use a test model to predict and observe the actual output shape.

    # Examples

    ```python
        # apply a 3x3 transposed convolution with stride 1x1 and 3 output filters on a 12x12 image:
        model = Sequential()
        model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), border_mode='valid', input_shape=(3, 12, 12)))
        # Note that you will have to change the output_shape depending on the backend used.

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12))
        # For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)
        # Theano GPU: (None, 3, 13, 13)
        # Theano CPU: (None, 3, 14, 14)
        # TensorFlow: (None, 14, 14, 3)

        # apply a 3x3 transposed convolution with stride 2x2 and 3 output filters on a 12x12 image:
        model = Sequential()
        model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25), subsample=(2, 2), border_mode='valid', input_shape=(3, 12, 12)))
        model.summary()

        # we can predict with the model and print the shape of the array.
        dummy_input = np.ones((32, 3, 12, 12))
        # For TensorFlow dummy_input = np.ones((32, 12, 12, 3))
        preds = model.predict(dummy_input)
        print(preds.shape)
        # Theano GPU: (None, 3, 25, 25)
        # Theano CPU: (None, 3, 25, 25)
        # TensorFlow: (None, 25, 25, 3)
    ```

    # Arguments
        nb_filter: Number of transposed convolution filters to use.
        nb_row: Number of rows in the transposed convolution kernel.
        nb_col: Number of columns in the transposed convolution kernel.
        output_shape: Output shape of the transposed convolution operation.
            tuple of integers (nb_filter, nb_output_rows, nb_output_cols)
            Formula for calculation of the output shape [1], [2]:
                o = s (i - 1) + a + k - 2p, \quad a \in \{0, \ldots, s - 1\}
                where:
                    i - input size (rows or cols),
                    k - kernel size (nb_filter),
                    s - stride (subsample for rows or cols respectively),
                    p - padding size,
                    a - user-specified quantity used to distinguish between
                        the s different possible output sizes.
             Because a is not specified explicitly and Theano and Tensorflow
             use different values, it is better to use a dummy input and observe
             the actual output shape of a layer as specified in the examples.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano/TensorFlow function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
        subsample: tuple of length 2. Factor by which to oversample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.

    # References
        [1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285 "arXiv:1603.07285v1 [stat.ML]")
        [2] [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
        [3] [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    '''
    def __init__(self, nb_filter, nb_row, nb_col, output_shape,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1),
                 dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise Exception('Invalid border mode for Deconvolution2D:', border_mode)

        self.output_shape_ = output_shape

        super(Deconvolution2D, self).__init__(nb_filter, nb_row, nb_col,
                                              init=init, activation=activation,
                                              weights=weights, border_mode=border_mode,
                                              subsample=subsample, dim_ordering=dim_ordering,
                                              W_regularizer=W_regularizer, b_regularizer=b_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              W_constraint=W_constraint, b_constraint=b_constraint,
                                              bias=bias, **kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = self.output_shape_[1]
            cols = self.output_shape_[2]
        elif self.dim_ordering == 'tf':
            rows = self.output_shape_[0]
            cols = self.output_shape_[1]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        output = K.deconv2d(x, self.W, (K.shape(x)[0],)+tuple(self.output_shape_),
                            strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_shape': (None,)+self.output_shape_}
        base_config = super(Deconvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Latent(object):
    def __init__(self, latent_dim, sample_fn, activation='linear'):
        from keras.layers.normalization import BatchNormalization as BN
        self.dim = latent_dim
        self.sampler = Lambda(sample_fn)
        self.encoder = Dense(latent_dim, activation=activation)
        self.discriminator = Sequential(
            (latent_dim,),
            [
                # putting BN in the discriminator worsens the result
                Dense(1000, activation='relu'),
                BN(),
                Dense(1000, activation='relu'),
                Dense(1,    activation='sigmoid'),
            ])
    def __call__ (self,pre_z):
        import tensorflow as tf
        z = self.encoder(pre_z)
        n = self.sampler(z)
        with tf.variable_scope("discriminator") as scope:
            d1 = self.discriminator(z)
            tf.get_variable_scope().reuse_variables()
            d2 = self.discriminator(n)
        return z, d1, d2
