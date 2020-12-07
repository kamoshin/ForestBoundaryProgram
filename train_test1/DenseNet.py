import os
import numpy as np

import tensorflow as tf

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Add
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Dropout
from keras.initializers import TruncatedNormal, Constant

from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.nasnet import NASNetMobile, preprocess_input
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.densenet import DenseNet121, DenseNet201, preprocess_input

from colorama import Fore
def errorMessage(_m_list):
    for ms in _m_list:
        print(Fore.RED+ms)
    print(Fore.RESET)

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
                image = img_to_array(load_img(dir_path+'/'+image_n, target_size=self.image_shape))
                #norm_image = image / 255.0
                norm_image = preprocess_input(image)
                X.append(norm_image)

                Y.append(set_label)

                read_num += 1
                
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X, Y
"""
def NASNet(classes):
    base_model = NASNetMobile(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
"""
"""
def InceptionResNet(classes):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
"""
def DenseNet(classes):
    #base_model = DenseNet121(weights='imagenet', include_top=False)
    base_model = DenseNet201(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

#log_dir = './../../Experiments/NASNet/'
#log_dir = './../../Experiments/AlexNet/'
#log_dir = './../../Experiments/InceptionResNet299/'
log_dir = './../../Experiments/DenseNet201/'

train_dir = './../../Dataset/samplingtrain/'
test_dir = './../../Dataset/samplingtest/'
image_shape = (224, 224)
#image_shape = (299, 299)
num_classes = 4
#batch_size = 8
batch_size = 16
learning_rate = 1e-7

"""
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=360,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    horizontal_flip=0.5,
    vertical_flip=0.5,
    fill_mode='reflect'
)


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical'
)
"""
dataset = Dataset(image_shape,train_dir,test_dir)
x_train, y_train, x_test, y_test = dataset.get_batch()

#model = NASNet(num_classes)
#model = AlexNet((image_shape[0], image_shape[1], 3), num_classes)
model = DenseNet(num_classes)
from keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)

#model = AlexNet(input_shape, num_classes)
model.summary()
#exit()

model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    """
    model.fit_generator(
        train_generator,
        steps_per_epoch=142540//batch_size+1,
        epochs=10000,
        validation_data=test_generator,
        validation_steps=26968//batch_size+1,
        callbacks=[
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(os.path.join(log_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'), monitor='val_accuracy', period=10)
        ],
        verbose=1
    )
    """
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=10000,
        validation_data=(x_test, y_test),
        callbacks=[
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(os.path.join(log_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'), monitor='val_accuracy', period=50)
        ],
        verbose=1
    )
except KeyboardInterrupt:
    print('')
    errorMessage(["!! press Ctrl+C !!", "!! stop train process !!"])

model.save(os.path.join(log_dir, 'end_model.h5'))