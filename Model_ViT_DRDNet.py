import torch
import torchvision.transforms as transforms
from vit_pytorch import ViT
import numpy as np
from Evaluation import evaluation
import cv2 as cv
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Activation, Dropout, GlobalAveragePooling2D, \
    BatchNormalization, concatenate, AveragePooling2D

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
    x = Conv2D(24, (3, 3), kernel_initializer='he_uniform', padding='same', dilation_rate= (1,1), use_bias=False)(input_img)

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



def Model_ViT_DRDNet(image, Target, sol = None):
    if sol is None:
        sol = [5, 5, 100]
    # Load pre-trained Vision Transformer model
    model = ViT(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Extract features using the Vision Transformer model
    with torch.no_grad():
        features = model(input_tensor)

    Feat = np.asarray(features)

    per = round(len(Feat) * 0.75)
    train_data = features[:per, :]
    train_target = Target[:per, :]
    test_data = features[per:, :]
    test_target = Target[per:, :]

    IMG_SIZE = [32, 32, 3]
    Data1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Data1[i, :, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Datas = Data1.reshape(Data1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = dense_net(num_of_class=train_target.shape[1])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Datas, train_target, steps_per_epoch=1, epochs=1)
    pred = model.predict(test_data)
    Eval = evaluation(pred.reshape(-1, 1), test_target)
    return Eval, pred


