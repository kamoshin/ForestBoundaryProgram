import os
import csv
import shutil
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

class Dataset():

    def __init__(self, image_shape, nolabel_dir):
        self.image_shape = image_shape
        self.nolabel_dir = nolabel_dir

    def get_batch(self):
        x = self.load_data(self.nolabel_dir)

        return x

    def load_data(self, data_dir):
        X = []

        image_dir = os.listdir(data_dir)
        image_dir = sorted(image_dir, key=str.lower)
        image_num = len(image_dir)
        print('image num : ',image_num)
        
        read_num = 0    
        for image_n in image_dir:
            image_path_list.append(image_n)
            image = img_to_array(load_img(data_dir+'/'+image_n, target_size=(self.image_shape[0],self.image_shape[1])))
            norm_image = image / 255.0
            X.append(norm_image)

            read_num += 1
            
            if read_num >= 15000:
                break
        X = np.asarray(X)

        return X

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

model1_path = './../../../Data/three_pronged/ensemble/Models/model1/model_2000_0.74.h5'
model2_path = './../../../Data/three_pronged/ensemble/Models/model2/model_9600_0.71.h5'
target_img_dir = './../../../Data/trimed_img/test/'

# load model
model1 = load_model(model1_path)
model2 = load_model(model2_path)
image_shape = (256, 256, 3)

# save_dir = './../../../Data/three_pronged/ensemble/pseudo_label_img/'
# photo_name_list = ['asaoka','kitagata','myodan', 'nishimataa', 'nishimatab',\
#                    'shijukuin', 'terachi', 'toriyao', 'uwadana', 'yoshikuraA']
label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']
high_line = 0.7
give_count = [0,0,0,0]
wrong_count = [0,0,0,0]

for i, label in enumerate(label_names):
    # load dataset
    image_path_list = []
    target_img_label = target_img_dir+'/'+label+'/'
    target_data = os.listdir(target_img_label)
    if len(target_data) == 0:
        continue
    dataset = Dataset(image_shape, target_img_label)
    x = dataset.get_batch()

    # predict
    predicted1 = model1.predict(x, batch_size=512)
    predicted2 = model2.predict(x, batch_size=512)

    # calc result
    for img_path, pred1, pred2 in zip(image_path_list, predicted1, predicted2):
        if np.argmax(pred1) == np.argmax(pred2):
            if pred1.max() > high_line and pred2.max() > high_line:
                give_count[i] += 1
                if np.argmax(pred1) != i:
                    wrong_count[i] += 1
        #         shutil.move(nolabel_dir+'/'+photo_name+'/'+img_path, save_img_dir+'/'+label_names[np.argmax(pred1)]+'/')
        #     else:
        #         os.remove(nolabel_dir+'/'+photo_name+'/'+img_path)
        # else:
        #     os.remove(nolabel_dir+'/'+photo_name+'/'+img_path)
print(give_count)
print(wrong_count)
