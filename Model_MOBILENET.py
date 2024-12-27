import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax
from keras.models import Model
import numpy as np
import os
import cv2 as cv
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from Evaluation import evaluation


def print_in_color(txt_msg,fore_tupple,back_tupple,):

    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
    print(msg .format(mat))
    print('\33[0m')
    return

def get_bs(dir,b_max):
    # dir is the directory containing the samples, b_max is maximum batch size to allow based on your memory capacity
    # you only want to go through test and validation set once per epoch this function determines needed batch size ans steps per epoch
    length=0
    dir_list=os.listdir(dir)
    for d in dir_list:
        d_path=os.path.join (dir,d)
        length=length + len(os.listdir(d_path))
    batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=b_max],reverse=True)[0]
    return batch_size,int(length/batch_size)


def MobileNet(train_data, train_target, test_data, sol):
    img_size = 224  # use 224 X 224 images compatible with mobilenet model
    lr = .01  # specify initial learning rate
    mobile = tf.keras.applications.mobilenet.MobileNet( include_top=False, input_shape=(img_size, img_size,3), pooling='max', weights='imagenet', dropout=.5)
    for layer in mobile.layers:
        layer.trainable=False
    x=mobile.layers[-1].output # this is the last layer in the mobilenet model the global max pooling layer
    x=keras.layers.BatchNormalization(axis=-1, momentum=3, epsilon=0.001 )(x)
    x=Dense(sol[1], activation='relu')(x)
    x=Dropout(rate=.3, seed = 123)(x)
    x=Dense(64, activation='relu')(x)
    x=Dropout(rate=.3, seed = 123)(x)
    predictions=Dense(train_target.shape[1], activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=predictions)

    model.compile(Adamax(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])


    class LRA(keras.callbacks.Callback):
        best_weights = model.get_weights()  # set a class vaiable so weights can be loaded after training is completed

        def __init__(self, patience=2, threshold=.95, factor=.5):
            super(LRA, self).__init__()
            self.patience = patience  # specifies how many epochs without improvement before learning rate is adjusted
            self.threshold = threshold  # specifies training accuracy threshold when lr will be adjusted based on validation loss
            self.factor = factor  # factor by which to reduce the learning rate
            self.lr = float(
                tf.keras.backend.get_value(model.optimizer.lr))  # get the initiallearning rate and save it in self.lr
            self.highest_tracc = 0.0  # set highest training accuracy to 0
            self.lowest_vloss = np.inf  # set lowest validation loss to infinity
            self.count = 0
            msg = '\n Starting Training - Initializing Custom Callback'
            print_in_color(msg, (244, 252, 3), (55, 65, 80))

        def on_epoch_end(self, epoch =sol[1], logs=None):  # method runs on the end of each epoch
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
            v_loss = logs.get('val_loss')  # get the validation loss for this epoch
            acc = logs.get('accuracy')  # get training accuracy
            if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
                if acc > self.highest_tracc:  # training accuracy improved in the epoch
                    msg = f'\n training accuracy improved from  {self.highest_tracc:7.2f} to {acc:7.2f} learning rate held at {lr:9.6f}'
                    print_in_color(msg, (0, 255, 0), (55, 65, 80))
                    self.highest_tracc = acc  # set new highest training accuracy
                    LRA.best_weights = model.get_weights()  # traing accuracy improved so save the weights
                    count = 0  # set count to 0 since training accuracy improved
                    if v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:  # training accuracy did not improve check if this has happened for patience number of epochs if so adjust learning rate
                    if self.count >= self.patience - 1:
                        self.lr = lr * self.factor  # adjust the learning by factor
                        tf.keras.backend.set_value(model.optimizer.lr, self.lr)  # set the learning rate in the optimizer
                        self.count = 0  # reset the count to 0
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                        msg = f'\nfor epoch {epoch + 1} training accuracy did not improve for {self.patience} consecutive epochs, learning rate adjusted to {lr:9.6f}'
                        print_in_color(msg, (255, 0, 0), (55, 65, 80))
                    else:
                        self.count = self.count + 1
                        msg = f'\nfor  epoch {epoch + 1} training accuracy did not improve, patience count incremented to {self.count}'
                        print_in_color(msg, (255, 255, 0), (55, 65, 80))
            else:  # training accuracy is above threshold so adjust learning rate based on validation loss
                if v_loss < self.lowest_vloss:  # check if the validation loss improved
                    msg = f'\n for epoch {epoch + 1} validation loss improved from  {self.lowest_vloss:7.4f} to {v_loss:7.4}, saving best weights'
                    print_in_color(msg, (0, 255, 0), (55, 65, 80))
                    self.lowest_vloss = v_loss  # replace lowest validation loss with new validation loss
                    LRA.best_weights = model.get_weights()  # validation loss improved so save the weights
                    self.count = 0  # reset count since validation loss improved
                else:  # validation loss did not improve
                    if self.count >= self.patience - 1:
                        self.lr = self.lr * self.factor
                        msg = f' \nfor epoch {epoch + 1} validation loss failed to improve for {self.patience} consecutive epochs, learning rate adjusted to {self.lr:9.6f}'
                        self.count = 0  # reset counter
                        print_in_color(msg, (255, 0, 0), (55, 65, 80))
                        tf.keras.backend.set_value(model.optimizer.lr, self.lr)  # set the learning rate in the optimizer
                    else:
                        self.count = self.count + 1  # increment the count
                        msg = f' \nfor epoch {epoch + 1} validation loss did not improve patience count incremented to {self.count}'
                        print_in_color(msg, (255, 255, 0), (55, 65, 80))



    model.set_weights(LRA.best_weights)
    weight = model.get_weights()[-1] # features of the fully connected layer
   # Modified Transfer Learning

    # model.fit(train_data, train_target)
    pred = np.round(model.predict(train_data)).astype('int')
    return pred

def Model_MOBILENET(train_data, train_target, test_data,Test_Tar, sol = None):
    IMG_SIZE = [224,224,3]
    if sol is None:
        sol = [150,10]
    Feat1 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] , IMG_SIZE[2]))
    Feat = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    for i in range(test_data.shape[0]):
        Feat1[i, :,:,:] = np.resize(test_data[i], (IMG_SIZE[0] , IMG_SIZE[1], IMG_SIZE[2]))
    predict = MobileNet(Feat1, train_target, test_data, sol)
    Eval = evaluation(predict, Test_Tar)
    return Eval,predict

