import os
import shutil
from colorama import Fore

import numpy as np

import tensorflow as tf
import keras

from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

#from AlexNet import load_data, get_AlexNet
from DenseNet import get_DenseNet201
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
log_dir = './../../Experiments/DenseNet16/'
#log_dir = './../../Experiments/NASNet/'
#log_dir = './../../Experiments/InceptionResNet299/'
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)

train_dir = './../../Dataset/samplingtrain16/'
test_dir = './../../Dataset/samplingtest16/'
image_shape = (224, 224)
#image_shape = (299, 299, 3)
num_classes = 4
#batch_size = 8
batch_size = 16
learning_rate = 1e-7

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

#model = get_NASNetMobile(num_classes)
#model = get_AlexNet((image_shape[0], image_shape[1], 3), num_classes)
model = get_DenseNet201(num_classes)
model.summary()


from keras.utils import multi_gpu_model
model = multi_gpu_model(model, gpus=2)

#exit()

model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.fit_generator(
        train_generator,
        steps_per_epoch=390352//batch_size,
        epochs=10000,
        validation_data=test_generator,
        validation_steps=92900//batch_size,
        callbacks=[
            TensorBoard(log_dir=log_dir),
            ModelCheckpoint(os.path.join(log_dir, 'model_{epoch:02d}_{val_accuracy:.4f}.h5'), monitor='val_accuracy', period=50),
            ModelCheckpoint(os.path.join(log_dir, 'model_best.h5'), save_best_only=True)
        ],
        verbose=1
    )
except KeyboardInterrupt:
    print('')
    errorMessage(["!! press Ctrl+C !!", "!! stop train process !!"])

model.save(os.path.join(log_dir, 'end_model.h5'))
