from __future__ import absolute_import

# tag::small_network[]
from keras.layers import LeakyReLU
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        Input(shape=input_shape),

        ZeroPadding2D(padding=3, data_format='channels_first'),  # <1>
        Conv2D(48, (7, 7), data_format='channels_first'),
        LeakyReLU(),

        ZeroPadding2D(padding=2, data_format='channels_first'),  # <2>
        Conv2D(32, (5, 5), data_format='channels_first'),
        LeakyReLU(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        LeakyReLU(),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(32, (5, 5), data_format='channels_first'),
        LeakyReLU(),

        Flatten(),
        Dense(512),
        LeakyReLU(),
    ]

# <1> We use zero padding layers to enlarge input images.
# <2> By using `channels_first` we specify that the input plane dimension for our features comes first.
# end::small_network[]
