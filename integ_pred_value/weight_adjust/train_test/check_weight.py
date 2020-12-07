import os
import numpy as np
import sys

import keras
from keras.models import load_model
from cv2 import cv2

# from future import print_function
from keras.models import load_model
import numpy as np

np.set_printoptions(linewidth=1000000)

model_path = './../../../../Data/Models/weight_adjust_other_tree/model_10000_0.86.h5'

print('Loading model and parameters...')
model = load_model(model_path)
model.summary()
print('Print the parameters...')

predicted_pred_value = model.predict(np.array([[0.0, 0.0, 0.0, 0.0]]))
print(predicted_pred_value)

layer = model.layers[0]     #summaryよりInput->[0], Dense->[1]なのでmodel.layers[1]
w = layer.get_weights()[0]
w = np.array(w)
print('**Parameters shape**')
print('w.shape', w.shape)

print('w = ', w)
