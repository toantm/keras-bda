#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.utils.generic_utils import Progbar
import numpy as np

from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.regularizers import l2

import sys
sys.setrecursionlimit(2 ** 25)

from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(1337)

K.set_image_dim_ordering('th')


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake.
    fake = Dense(1, activation='sigmoid', name='generation')(features)

    return Model(input=image, output=fake)


def rnpa_bottleneck_layer(input_tensor, nb_filters, filter_sz, stage,
                          init='glorot_normal', reg=0.0, use_shortcuts=True):
    nb_in_filters, nb_bottleneck_filters = nb_filters

    bn_name = 'bn' + str(stage)
    conv_name = 'conv' + str(stage)
    relu_name = 'relu' + str(stage)
    merge_name = '+' + str(stage)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    if stage > 1:  # first activation is just after conv1
        x = BatchNormalization(axis=1, name=bn_name + 'a')(input_tensor)
        x = Activation('relu', name=relu_name + 'a')(x)
    else:
        x = input_tensor

    x = Convolution2D(
        nb_bottleneck_filters, 1, 1,
        init=init,
        W_regularizer=l2(reg),
        bias=False,
        name=conv_name + 'a'
    )(x)

    # batchnorm-relu-conv, from nb_bottleneck_filters to nb_bottleneck_filters via FxF conv
    x = BatchNormalization(axis=1, name=bn_name + 'b')(x)
    x = Activation('relu', name=relu_name + 'b')(x)
    x = Convolution2D(
        nb_bottleneck_filters, filter_sz, filter_sz,
        border_mode='same',
        init=init,
        W_regularizer=l2(reg),
        bias=False,
        name=conv_name + 'b'
    )(x)

    # batchnorm-relu-conv, from nb_in_filters to nb_bottleneck_filters via 1x1 conv
    x = BatchNormalization(axis=1, name=bn_name + 'c')(x)
    x = Activation('relu', name=relu_name + 'c')(x)
    x = Convolution2D(nb_in_filters, 1, 1,
                      init=init, W_regularizer=l2(reg),
                      name=conv_name + 'c'
                      )(x)

    # merge
    if use_shortcuts:
        x = merge([x, input_tensor], mode='sum', name=merge_name)

    return x


def ResNetPreAct(input_shape=(1, 28, 28), nb_classes=10,
                 layer1_params=(5, 64, 2),
                 res_layer_params=(3, 16, 3),
                 final_layer_params=None,
                 init='glorot_normal', reg=0.0, use_shortcuts=True
                 ):
    """
    Source: https://gist.github.com/JefferyRPrice/c1ecc3d67068c8d9b3120475baba1d7e

    Return a new Residual Network using full pre-activation based on the work in
    "Identity Mappings in Deep Residual Networks"  by He et al
    http://arxiv.org/abs/1603.05027

    The following network definition achieves 92.0% accuracy on CIFAR-10 test using
    `adam` optimizer, 100 epochs, learning rate schedule of 1e.-3 / 1.e-4 / 1.e-5 with
    transitions at 50 and 75 epochs:
    ResNetPreAct(layer1_params=(3,128,2),res_layer_params=(3,32,25),reg=reg)

    Removed max pooling and using just stride in first convolutional layer. Motivated by
    "Striving for Simplicity: The All Convolutional Net"  by Springenberg et al
    (https://arxiv.org/abs/1412.6806) and my own experiments where I observed about 0.5%
    improvement by replacing the max pool operations in the VGG-like cifar10_cnn.py example
    in the Keras distribution.

    Parameters
    ----------
    input_dim : tuple of (C, H, W)

    nb_classes: number of scores to produce from final affine layer (input to softmax)

    layer1_params: tuple of (filter size, num filters, stride for conv)

    res_layer_params: tuple of (filter size, num res layer filters, num res stages)

    final_layer_params: None or tuple of (filter size, num filters, stride for conv)

    init: type of weight initialization to use

    reg: L2 weight regularization (or weight decay)

    use_shortcuts: to evaluate difference between residual and non-residual network
    """

    sz_L1_filters, nb_L1_filters, stride_L1 = layer1_params
    sz_res_filters, nb_res_filters, nb_res_stages = res_layer_params

    use_final_conv = (final_layer_params is not None)
    if use_final_conv:
        sz_fin_filters, nb_fin_filters, stride_fin = final_layer_params
        sz_pool_fin = input_shape[1] / (stride_L1 * stride_fin)
    else:
        sz_pool_fin = input_shape[1] / (stride_L1)

    img_input = Input(shape=input_shape, name='cifar')

    x = Convolution2D(
        nb_L1_filters, sz_L1_filters, sz_L1_filters,
        border_mode='same',
        subsample=(stride_L1, stride_L1),
        init=init,
        W_regularizer=l2(reg),
        bias=False,
        name='conv0'
    )(img_input)
    x = BatchNormalization(axis=1, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)

    for stage in range(1, nb_res_stages + 1):
        x = rnpa_bottleneck_layer(
            x,
            (nb_L1_filters, nb_res_filters),
            sz_res_filters,
            stage,
            init=init,
            reg=reg,
            use_shortcuts=use_shortcuts
        )

    x = BatchNormalization(axis=1, name='bnF')(x)
    x = Activation('relu', name='reluF')(x)

    if use_final_conv:
        x = Convolution2D(
            nb_fin_filters, sz_fin_filters, sz_fin_filters,
            border_mode='same',
            subsample=(stride_fin, stride_fin),
            init=init,
            W_regularizer=l2(reg),
            name='convF'
        )(x)

    x = AveragePooling2D((sz_pool_fin, sz_pool_fin), name='avg_pool')(x)

    x = Flatten(name='flat')(x)
    x = Dense(nb_classes, activation='softmax', name='fc10')(x)

    return Model(img_input, x, name='rnpa')

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 100
    batch_size = 100
    latent_size = 100
    nb_classes = 10
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    opt = SGD(lr=0.01)
    discriminator.compile(
        optimizer=opt,
        loss='binary_crossentropy')

    # build the classifier
    resnet = ResNetPreAct(layer1_params=(3, 128, 2), res_layer_params=(3, 32, 25), reg=0.0)

    resnet.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake_img = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    resnet.trainable = False

    fake = discriminator(fake_img)
    aux = resnet(fake_img)

    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]

    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    # fo = open("accuracy_save.txt", "wb")

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        epoch_resnet_loss =[]

        for index in range(nb_batches):
            progress_bar.update(index)

            # index = 1
            # generate a new batch of noise
            noise = np.random.normal(loc=0.0, scale=1, size=(batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, 10, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            # print(label_batch.shape)
            # print(sampled_labels.shape)   

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
            aux_y = np_utils.to_categorical(aux_y, 10)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, y))
            #
            epoch_resnet_loss.append(resnet.train_on_batch(X, aux_y))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.normal(loc=0.0, scale=1, size=(2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size).reshape(-1, 1)
            aux_sampled_labels = np_utils.to_categorical(sampled_labels, 10)

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels], [trick, aux_sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.normal(loc=0.0, scale=1, size=(nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)

        aux_y = np.concatenate((y_test, sampled_labels), axis=0)
        aux_y = np_utils.to_categorical(aux_y, 10)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(X, y, verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        resnet_test_loss = resnet.evaluate(X, aux_y, verbose=False)

        resnet_train_loss = np.mean(np.array(epoch_resnet_loss), axis=0)

        # # evaluate the test classification accuracy
        #
        # (loss, accuracy) = resnet.evaluate(X_test, Y_test, verbose=0)
        #
        # # show the accuracy on the testing set
        # print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
        #
        # fo.write('Test accuracy at the ' + str(epoch+1) + '-th iteration is: ' + str(accuracy) + '\n')


        # make new noise
        noise = np.random.normal(loc=0.0, scale=1, size=(2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test).reshape(-1, 1)
        aux_sampled_labels = np_utils.to_categorical(sampled_labels, 10)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels],
            [trick, aux_sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['resnet'].append(resnet_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        test_history['resnet'].append(resnet_test_loss)

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        # discriminator.save_weights(
        #    'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)
        resnet.save_weights(
            'params_resnet_epoch_{0:03d}.hdf5'.format(epoch), True)



    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))

    # evaluate the test classification accuracy
    (loss, accuracy) = resnet.evaluate(X_test, Y_test,
                                       batch_size=batch_size, verbose=0)

    # show the accuracy on the testing set
    print("\n [INFO] Test accuracy: {:.2f}%".format(accuracy * 100))

    # fo.close()


