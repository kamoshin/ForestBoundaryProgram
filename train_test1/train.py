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
    # model = keras.models.Sequential([
    #     keras.layers.Conv2D(32, kernel_size=3,
    #                         activation="relu",
    #                         input_shape=input_shape),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.Dropout(0.25),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(128, activation="relu"),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(num_classes, activation="softmax")
    # ])
    model = Sequential()
    model.add(Dense(3, input_dim=8, activation='sigmoid'))
    model.add(dense(num_classes, activation='softmax'))

    return model

def AlexNet(input_shape, num_classes):
    model = Sequential()

    #conv1
    model.add(conv2d(96, 11, strides=(4, 4), padding='valid', bias_init=0,
        input_shape=input_shape))
    #pool1
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #conv2
    model.add(conv2d(256, 5)) 
    #pool2
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #conv3
    model.add(conv2d(384, 3, bias_init=0))
    #conv4
    model.add(conv2d(384, 3)) 
    #conv5
    model.add(conv2d(256, 3)) 
    #pool5keras.backend.set
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(BatchNormalization())

    #fc6
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    #fc7
    model.add(dense(4096))
    model.add(Dropout(0.5))

    #fc8
    model.add(dense(num_classes, activation='softmax'))

    return model
    
class Dataset():

    def __init__(self, image_shape, train_dir, vali_dir):
        self.image_shape = image_shape
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
        label_dir = os.listdir(data_dir)
        label_num = len(label_dir)
        print('label name :',label_dir,'  num :',label_num)
        
        for label_n in label_dir:
            set_label = []
            if label_n == 'japanese_cedar':
                set_label = [1,0,0,0]
            elif label_n == 'japanese_cypress':
                set_label = [0,1,0,0]
            elif label_n == 'other_tree':
                set_label = [0,0,1,0]
            elif label_n == 'not_tree':
                set_label = [0,0,0,1]
            else:
                print('!!!set_label error!!! check dataset directory')
                exit()
                
            dir_path = data_dir+label_n
            image_dir = os.listdir(dir_path)
            image_dir = sorted(image_dir, key=str.lower)
            image_num = len(image_dir)
            print('image num : ',image_num)
            
            read_num = 0    
            for image_n in image_dir:
                image = img_to_array(load_img(dir_path+'/'+image_n, target_size=(self.image_shape[0],self.image_shape[1])))
                norm_image = image / 255.0
                X.append(norm_image)

                Y.append(set_label)

                read_num += 1
                
                if read_num >= 20000:
                    break
        X = np.asarray(X)
        Y = np.asarray(Y)

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

log_dir = './../../Data/Models/few_data_add_pseudo_label/'
makeDir(log_dir, mode='keep')

# train_dir = './../../Data/trimed_img/fewdata_dataaugumentation/samplingtraindata1000_add_pseudo_label/'
# train_dir = './../../Data/trimed_img/unresize_randomsampling_train_add_pseudo_label/'
# vali_dir = './../../Data/trimed_img/fewdata_dataaugumentation/samplingtestdata/'

image_shape = (256, 256, 3)
num_classes = 4

dataset = Dataset(image_shape,train_dir,vali_dir)

model = AlexNet(image_shape, num_classes)
#model = load_model('./../../Data/Report/'+'model_4950_0.77.h5')

x_train, y_train, x_vali, y_vali = dataset.get_batch()
trainer = Trainer(log_dir=log_dir,model=model, loss='categorical_crossentropy')

#trainer = Trainer(log_dir=log_dir,model=model, loss='categorical_crossentropy', optimizer=Adam(lr=1e-5))
trainer.train(x_train, y_train, batch_size=64, epochs=10000, validation_data=(x_vali, y_vali))

score = model.evaluate(x_vali, y_vali, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])