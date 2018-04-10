################
# Library
################
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
from skimage.transform import resize
import os
from keras.layers import Input, UpSampling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import backend as K

from keras.models import Model, load_model

from keras.callbacks import TensorBoard
from keras.applications import VGG16

################
# Constant
################













################
# Functions
################
class MNIST:
    def __init__(self):
        # set parameters
        self.IMG_ROW = 28
        self.IMG_COL = 28
        self.IMG_CHA = 1
        self.IMG_CLA = 10
        self.IMG_SHAPE = (self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        self.IMG_MIN = 0
        self.IMG_MAX = 1
        # loading data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # 255 degree to [0,1]
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        # input reshape 
        X_train = X_train.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        X_test = X_test.reshape(-1, self.IMG_ROW, self.IMG_COL, self.IMG_CHA)
        # one hot encoding
        y_train = np_utils.to_categorical(y_train, self.IMG_CLA)
        y_test = np_utils.to_categorical(y_test, self.IMG_CLA)
        # assign to variable
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        
        
        
class Classifier:
    
    def __init__(self, input_shape, session=None):
        
        # init with VGG16
        vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        # make a reference to VGG's input layer
        inp = vgg.input
        
        # make a new softmax layer with num_classes neurons
        flatten = Flatten()(vgg.layers[-2].output)
        out = Dense(10)(flatten)
        
        # loss
        def fn(correct, predicted):
            return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                           logits=predicted)
        # classifier
        self.model = Model(inp, out)
        
        # compile and print summary
        self.model.compile(optimizer='adadelta', loss=fn, metrics=['accuracy'])
        print(self.model.summary())

    def train(self, X, y, X_val, y_val, save_path='models/mnist_classifier'):
        if os.path.exists(save_path):
            self.restore(save_path)
        else:
            # fit model
            self.model.fit(X, y, epochs = 10, batch_size = 128,\
                            shuffle = True, validation_data = (X_val, y_val),\
                            callbacks = [TensorBoard(log_dir = '/tmp/classifier')])
        # save model
        self.model.save(save_path)

    def restore(self, path):
        self.model.load_weights(path)

    def predict(self, data):
        return self.model(data)

    
        

################
# Main
################
if __name__ == '__main__':
    # init
    mnist = MNIST()
    # classifier
    classifier = Classifier(input_shape=mnist.IMG_SHAPE)
    # training
    classifier.train(mnist.X_train, mnist.y_train, mnist.X_test, mnist.y_test,
                     save_path='models/classifier_trans')
        
        
        
        
        
        