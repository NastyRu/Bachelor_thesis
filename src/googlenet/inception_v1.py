from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, \
                         Concatenate, AveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

class_num = 11


class inceptionV1():
    def __init__(self):
        self.name = "inceptionV1"

    def architecture(self):
        input = Input(shape=(224, 224, 3))

        layer = Convolution2D(filters=64,
                              kernel_size=(7, 7),
                              strides=2,
                              padding='same',
                              activation='relu')(input)
        layer = MaxPooling2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(layer)

        layer = Convolution2D(filters=64,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              activation='relu')(layer)
        layer = Convolution2D(filters=192,
                              kernel_size=(3, 3),
                              strides=1,
                              padding='same',
                              activation='relu')(layer)
        layer = MaxPooling2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(layer)

        layer = self.inception(input=layer,
                               filters_1x1=64,
                               filters_3x3_reduce=96,
                               filters_3x3=128,
                               filters_5x5_reduce=16,
                               filters_5x5=32,
                               filters_pool_proj=32)
        layer = self.inception(input=layer,
                               filters_1x1=128,
                               filters_3x3_reduce=128,
                               filters_3x3=192,
                               filters_5x5_reduce=32,
                               filters_5x5=96,
                               filters_pool_proj=64)
        layer = MaxPooling2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(layer)

        layer = self.inception(input=layer,
                               filters_1x1=192,
                               filters_3x3_reduce=96,
                               filters_3x3=208,
                               filters_5x5_reduce=16,
                               filters_5x5=48,
                               filters_pool_proj=64)

        aux1 = self.auxiliary(layer)

        layer = self.inception(input=layer,
                               filters_1x1=160,
                               filters_3x3_reduce=112,
                               filters_3x3=224,
                               filters_5x5_reduce=24,
                               filters_5x5=64,
                               filters_pool_proj=64)
        layer = self.inception(input=layer,
                               filters_1x1=128,
                               filters_3x3_reduce=128,
                               filters_3x3=256,
                               filters_5x5_reduce=24,
                               filters_5x5=64,
                               filters_pool_proj=64)
        layer = self.inception(input=layer,
                               filters_1x1=112,
                               filters_3x3_reduce=144,
                               filters_3x3=288,
                               filters_5x5_reduce=32,
                               filters_5x5=64,
                               filters_pool_proj=64)

        aux2 = self.auxiliary(layer)

        layer = self.inception(input=layer,
                               filters_1x1=256,
                               filters_3x3_reduce=160,
                               filters_3x3=320,
                               filters_5x5_reduce=32,
                               filters_5x5=128,
                               filters_pool_proj=128)
        layer = MaxPooling2D(pool_size=(3, 3),
                             strides=2,
                             padding='same')(layer)

        layer = self.inception(input=layer,
                               filters_1x1=256,
                               filters_3x3_reduce=160,
                               filters_3x3=320,
                               filters_5x5_reduce=32,
                               filters_5x5=128,
                               filters_pool_proj=128)
        layer = self.inception(input=layer,
                               filters_1x1=384,
                               filters_3x3_reduce=192,
                               filters_3x3=384,
                               filters_5x5_reduce=48,
                               filters_5x5=128,
                               filters_pool_proj=128)
        layer = AveragePooling2D(pool_size=(7, 7),
                                 strides=1,
                                 padding='same')(layer)

        layer = Dropout(rate=0.4)(layer)
        layer = Dense(units=1000, activation='linear')(layer)
        output = Dense(units=class_num, activation='softmax')(layer)

        return Model(inputs=input, outputs=[output, aux1, aux2])

    def inception(self, input,
                  filters_1x1,
                  filters_3x3_reduce, filters_3x3,
                  filters_5x5_reduce, filters_5x5,
                  filters_pool_proj):
        conv_1x1 = Convolution2D(filters=filters_1x1,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same',
                                 activation='relu')(input)

        conv_3x3_reduce = Convolution2D(filters=filters_3x3_reduce,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding='same',
                                        activation='relu')(input)
        conv_3x3 = Convolution2D(filters=filters_3x3,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation='relu')(conv_3x3_reduce)

        conv_5x5_reduce = Convolution2D(filters=filters_5x5_reduce,
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding='same',
                                        activation='relu')(input)
        conv_5x5 = Convolution2D(filters=filters_5x5,
                                 kernel_size=(5, 5),
                                 strides=1,
                                 padding='same',
                                 activation='relu')(conv_5x5_reduce)

        maxpool = MaxPooling2D(pool_size=(3, 3),
                               strides=1,
                               padding='same')(input)
        maxpool_proj = Convolution2D(filters=filters_pool_proj,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same',
                                     activation='relu')(maxpool)

        output = Concatenate(axis=-1)([conv_1x1, conv_3x3,
                                       conv_5x5, maxpool_proj])
        return output

    def auxiliary(self, input):
        layer = AveragePooling2D(pool_size=(5, 5),
                                 strides=3,
                                 padding='same')(input)
        layer = Convolution2D(filters=128,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              activation='relu')(layer)
        layer = Dense(units=256,
                      activation='relu')(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=class_num,
                      activation='softmax')(layer)
        return layer
