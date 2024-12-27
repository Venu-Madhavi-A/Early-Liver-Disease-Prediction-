import keras
import cv2 as cv
import numpy as np
from keras import activations
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D
from Evaluation import evaluation




def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters


def dense_net(num_of_class=None):
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = BatchNormalization()(x)
    dense_x = Activation('relu')(x)

    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)

    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)

    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)
    weight = np.asarray(model.get_weights()[30])
    return model


def Model_DENSENET(train_data,test_data,train_target, test_target):
    IMG_SIZE = [32, 32, 3]
    Data1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Data1[i, :, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Datas = Data1.reshape(Data1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = dense_net(num_of_class=train_target.shape[1])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Datas, train_target, steps_per_epoch=1, epochs=1)
    # f1 = np.asarray(model.get_weights()[30].ravel())  # Fully Connected Layer Dense FC2
    # Feat1 = cv.resize(f1, (f1.shape[0], Target.shape[0]))
    # return Feat1
    pred = model.predict(test_data)
    activation = activations.relu(train_data).numpy()
    # activation = activation.eval(session=tf.compat.v1.Session())
    Eval = evaluation(pred.reshape(-1, 1), test_target)
    return activation, pred,Eval


def Model__DENSENET(train_data,test_data,train_target, test_target,sol):
    IMG_SIZE = [32, 32, 3]
    Data1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Data1[i, :, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Datas = Data1.reshape(Data1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = dense_net(num_of_class=train_target.shape[1])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Datas, train_target, steps_per_epoch=1, epochs=sol)
    # f1 = np.asarray(model.get_weights()[30].ravel())  # Fully Connected Layer Dense FC2
    # Feat1 = cv.resize(f1, (f1.shape[0], Target.shape[0]))
    # return Feat1
    pred = model.predict(test_data)
    activation = activations.relu(train_data).numpy()
    # activation = activation.eval(session=tf.compat.v1.Session())
    Eval = evaluation(pred.reshape(-1, 1), test_target)
    return activation, pred,Eval