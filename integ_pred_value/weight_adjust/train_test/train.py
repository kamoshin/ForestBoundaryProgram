import os
import numpy as np

import tensorflow as tf

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.initializers import TruncatedNormal, Constant

def makeDir(dir_path, mode='remove'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

"""
def step_decay(epochs):
    x = 1e-4
    if epochs >= 75 and epochs < 150: 
        x = 1e-5
        print("----{}----".format(x))
    elif epochs >= 150 and epochs < 225: 
        x = 1e-6
        print("----{}----".format(x))
    elif epochs >= 225: 
        x = 1e-7
        print("----{}----".format(x))
    return x
"""

def conv2d(filters, kernel_size, strides=(1, 1), padding='same', bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters, kernel_size, strides=strides, padding=padding,
        activation='relu', kernel_initializer=trunc, bias_initializer=cnst, **kwargs
    )   

def dense(units, activation='relu'):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units, activation=activation,
        kernel_initializer=trunc, bias_initializer=cnst,
    )

def Layer3NN(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=4))

    return model
    
class Dataset():

    def __init__(self, input_shape, train_dir, vali_dir):
        self.input_shape = input_shape
        self.train_dir = train_dir
        self.vali_dir = vali_dir

    def get_batch(self):
        x_train, y_train = self.load_data(self.train_dir)
        x_vali, y_vali = self.load_data(self.vali_dir)

        return x_train, y_train, x_vali, y_vali

    def load_data(self, data_dir):
        X = []
        Y = []

        #read images dir
        pred_value_list = os.listdir(data_dir+'X/')
        pred_label_list = os.listdir(data_dir+'Y/')
        pred_value_list = sorted(pred_value_list)
        pred_label_list = sorted(pred_label_list)

        for pred_value in pred_value_list:
            pred_value = np.load(data_dir+'X/'+pred_value)
            X.extend(pred_value)


        for pred_label in pred_label_list:
            pred_label = np.load(data_dir+'Y/'+pred_label)
            Y.extend(pred_label)

        X = np.asarray(X)
        Y = np.asarray(Y)

        print(X.shape)
        print(Y.shape)

        return X, Y

class Trainer():
    
    def __init__(self, log_dir, model, loss):
        self._target = model
        #self.lr = LearningRateScheduler(step_decay)
        self.optimizer=RMSprop(lr=1e-7)
        self._target.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.verbose = 1
        self.log_dir = log_dir
                
    def train(self, x_train, y_train, batch_size, epochs, validation_data):
        if os.path.exists(self.log_dir):
            import shutil
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)

        self._target.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=validation_data,
            callbacks=[
                TensorBoard(log_dir=self.log_dir),
                ModelCheckpoint(os.path.join(self.log_dir,'model_{epoch:02d}_{val_accuracy:.2f}.h5'), \
                    monitor='val_accuracy', period=20),
                #self.lr
            ],
            verbose=self.verbose
        )

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

log_dir = './../../../../Data/Models/weight_adjust_not_tree/'
makeDir(log_dir, mode='keep')

train_dir = './../../../../Data/pred_value/weight_adjust/train/not_tree/'
vali_dir = './../../../../Data/pred_value/weight_adjust/test/not_tree/'

input_value_shape = (4, 1)
num_classes = 1

dataset = Dataset(input_value_shape,train_dir,vali_dir)

model = Layer3NN(input_value_shape, num_classes)
#model = load_model('./../../Data/Report/'+'model_4950_0.77.h5')

x_train, y_train, x_vali, y_vali = dataset.get_batch()
trainer = Trainer(log_dir=log_dir,model=model, loss='mean_squared_error')

#trainer = Trainer(log_dir=log_dir,model=model, loss='categorical_crossentropy', optimizer=Adam(lr=1e-5))
trainer.train(x_train, y_train, batch_size=64, epochs=10000, validation_data=(x_vali, y_vali))

score = model.evaluate(x_vali, y_vali, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])