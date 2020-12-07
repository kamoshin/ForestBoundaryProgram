import os
import numpy as np

from keras.preprocessing.image import img_to_array, load_img

from keras.models import Sequential, Model
from keras.initializers import TruncatedNormal, Constant
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout

def load_data(data_dir, image_shape):
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
            image = img_to_array(load_img(dir_path+'/'+image_n, target_size=image_shape))
            image = image / 255.0
            X.append(image)

            Y.append(set_label)

            read_num += 1
            
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y

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

def get_AlexNet(input_shape, num_classes):
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
    #pool5
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
