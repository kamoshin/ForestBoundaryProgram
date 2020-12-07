import os
import shutil
import csv
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

import tensorflow as tf
import keras

from keras.models import load_model

#from AlexNet import load_data
#from NASNet import load_data
from DenseNet import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

#save_dir = './../../Experiments/Report/AlexNet/'
#save_dir = './../../Experiments/Report/NASNet/'
save_dir = './../../Experiments/Report/DenseNet256/'
shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)

size = 224
#model_path = './../../Experiments/AlexNet/model_best.h5'
#model_path = './../../Experiments/NASNet/end_model.h5'
model_path = './../../Experiments/DenseNet256/model_best.h5'
test_dir = './../../Dataset/test/'

# load model
model = load_model(model_path)
image_shape = (size, size, 3)

# load dataset
image_path_list = []
x_test, y_test = load_data(test_dir, image_shape)

# predict
predicted = model.predict(x_test)
predicted_classes = np.argmax(predicted, axis=1)

# calc result
true_classes = np.argmax(y_test,axis=1)
label_names = ['japanese_cedar', 'japanese_cypress', 'other_tree', 'not_tree']

report_df = pd.DataFrame(classification_report(true_classes, predicted_classes, target_names=label_names, output_dict=True)).transpose()
confusion_matrix_df = pd.DataFrame(confusion_matrix(true_classes, predicted_classes), columns=label_names, index=label_names)
print(report_df)
print(confusion_matrix_df)

# save report_df as csv
with open(save_dir+'pred_results.csv','w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'pred_0', 'pred_1', 'pred_2', 'pred_3', 'true', 'pred_classes'])
    for path, pred, true, pred_class in zip(image_path_list, predicted, true_classes, predicted_classes):
        writer.writerow([path, pred[0], pred[1], pred[2], pred[3], true, pred_class])

report_save_path = save_dir+'predict_report.csv'
confusion_matrix_save_path = save_dir+'confusion_matrix.csv'
report_df.to_csv(report_save_path)
confusion_matrix_df.to_csv(confusion_matrix_save_path)
