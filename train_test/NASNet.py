import os
import numpy as np

from keras.preprocessing.image import img_to_array, load_img

from keras.applications.nasnet import NASNetLarge, NASNetMobile, preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model

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

def get_NASNetMobile(classes):
    base_model = NASNetMobile(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def get_NASNetLarge(classes):
    base_model = NASNetLarge(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
