import os
import csv
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

import tensorflow as tf

import keras
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

class Dataset():

    def __init__(self, image_shape, test_dir):
        self.image_shape = image_shape
        self.test_dir = test_dir

    def get_batch(self):
        x_test, y_test = self.load_data(self.test_dir)

        return x_test, y_test

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
                set_label = [1,0,0]
            elif label_n == 'japanese_cypress':
                set_label = [0,1,0]
            # elif label_n == 'other_tree':
            #     set_label = [0,0,0]
            elif label_n == 'not_tree':
                set_label = [0,0,1]
            else:
                print('!!!set_label error!!! check datasets directory')
                exit()
                
            dir_path = data_dir+label_n
            image_dir = os.listdir(dir_path)
            image_dir = sorted(image_dir, key=str.lower)
            image_num = len(image_dir)
            print(label_n + ' image num : ',image_num)
            
            read_num = 0    
            for image_n in image_dir:
                image_path_list.append(image_n)
                image = img_to_array(load_img(dir_path+'/'+image_n, target_size=(self.image_shape[0],self.image_shape[1])))
                norm_image = image / 255.0
                X.append(norm_image)

                Y.append(set_label)

                read_num += 1
                
                if read_num >= 100000:
                    break
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X, Y

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

save_dir = './../../../Data/reduce_othertree_Report/'
if os.path.exists(save_dir):
    print('save_dir is exists')
else:
    os.mkdir(save_dir)

model_path = './../../../Data/reduce_othertree_Models/model_2250_0.92.h5'
test_dir = './../../../Data/trimed_img/test/'

# load model
model = load_model(model_path)
image_shape = (256, 256, 3)

# load dataset
image_path_list = []
dataset = Dataset(image_shape, test_dir)
x_test, y_test = dataset.get_batch()

# predict
predicted = model.predict_classes(x_test, batch_size=512)

# calc result
true_classes = np.argmax(y_test,axis=1)
label_names = ['japanese_cedar', 'japanese_cypress', 'not_tree']

report_df = pd.DataFrame(classification_report(true_classes, predicted, target_names=label_names, output_dict=True)).transpose()
confusion_matrix_df = pd.DataFrame(confusion_matrix(true_classes, predicted), columns=label_names, index=label_names)
print(report_df)
print(confusion_matrix_df)

wrong_index_list = np.where(predicted != true_classes)
wrong_image_path = [image_path_list[wrong_index] for wrong_index in wrong_index_list[0]]
wrong_true_list =  [true_classes[wrong_index] for wrong_index in wrong_index_list[0]]
wrong_predicted_list = [predicted[wrong_index] for wrong_index in wrong_index_list[0]]

# save report_df as csv
with open(save_dir+'wrong_img.csv','w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'true', 'pred'])
    for w_path, w_true, w_pred in zip(wrong_image_path, wrong_true_list, wrong_predicted_list):
        writer.writerow([w_path, w_true, w_pred])

report_save_path = save_dir+'predict_report.csv'
confusion_matrix_save_path = save_dir+'confusion_matrix.csv'
report_df.to_csv(report_save_path)
confusion_matrix_df.to_csv(confusion_matrix_save_path)