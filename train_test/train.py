import os
import shutil
from colorama import Fore

import numpy as np

import tensorflow as tf
import keras

from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint

#from AlexNet import load_data, get_AlexNet
from DenseNet import load_data, get_DenseNet201
#from InceptionResNetV2 import load_data, get_InceptionResNetV2
#from NASNet import load_data, get_NASNetMobile

def errorMessage(_m_list):
    for ms in _m_list:
        print(Fore.RED+ms)
    print(Fore.RESET)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

#log_dir = './../../Experiments/AlexNet/'
log_dir = './../../Experiments/DenseNet64/'
#log_dir = './../../Experiments/NASNet/'
#log_dir = './../../Experiments/InceptionResNet299/'
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)

train_dir = './../../Dataset/samplingtrain64/'
test_dir = './../../Dataset/samplingtest64/'
image_shape = (224, 224, 3)
#image_shape = (299, 299, 3)
num_classes = 4
#batch_size = 8
batch_size = 16
learning_rate = 1e-7

x_train, y_train = load_data(train_dir, image_shape)
x_test, y_test = load_data(test_dir, image_shape)

#model = get_NASNetMobile(num_classes)
#model = get_AlexNet((image_shape[0], image_shape[1], 3), num_classes)
model = get_DenseNet201(num_classes)
model.summary()


from keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)

#exit()

model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=10000,
        validation_data=(x_test, y_test),
        callbacks=[
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(os.path.join(log_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'), monitor='val_accuracy', period=50),
            ModelCheckpoint(os.path.join(log_dir, 'model_best.h5'), monitor='val_accuracy', save_best_only=True)
        ],
        verbose=1
    )
except KeyboardInterrupt:
    print('')
    errorMessage(["!! press Ctrl+C !!", "!! stop train process !!"])

model.save(os.path.join(log_dir, 'end_model.h5'))
