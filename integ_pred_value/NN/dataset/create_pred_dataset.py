import os
import numpy as np
import sys

np.set_printoptions(linewidth=100000)

lerge_size = 256
middle_size = 128
small_size = 64
most_small_size =32
slide = 32

#read images dir
image_list = ['bamba', 'kurikara', 'matii', 'takemata', 'togi', 'urii']
# image_list = ['ushitsu', 'yuusenji']
pred_value_dir = './../../../../Data/pred_value/DenseNet'
# pred_value_alex_dir = './../../../../Data/pred_value/AlexNet'
save_dir = './../../../../Data/pred_value/weight_adjust/'
colormap_dir = './../../../../../OrthoData/train/colormap/'

for image_n in image_list:
    X = []
    Y = [] 
    print(image_n)
    read_num = 0
    print('lerge_size loading...')
    learge_pred = np.load(pred_value_dir+str(lerge_size)+'/'+str(image_n)+'_'+str(lerge_size)+'_predicted_keeper.npy')
    print('middle_size loading...')
    middle_pred = np.load(pred_value_dir+str(middle_size)+'/'+str(image_n)+'_'+str(middle_size)+'_predicted_keeper.npy')
    print('small_size loading...')
    small_pred = np.load(pred_value_dir+str(small_size)+'/'+str(image_n)+'_'+str(small_size)+'_predicted_keeper.npy')
    print('most_small_size loading...')
    most_small_pred = np.load(pred_value_dir+str(most_small_size)+'/'+str(image_n)+'_'+str(most_small_size)+'_predicted_keeper.npy')
    print('colormap loading...')
    colormap_label = np.load(colormap_dir+str(image_n)+'_colormap_label.npy')
    height, width, channels = colormap_label.shape[:3] 

    for r_y in range(0, height, slide):
        if r_y + slide >= height:
            break
        for r_x in range(0, width, slide):
            sys.stdout.write("\r y : %d" % r_y)
            sys.stdout.flush()
            if r_x + slide >= width:
                break
            if not all(colormap_label[r_y][r_x] == [0, 0, 0, 0]):
                pred_value_set = []
                pred_value_set.extend(learge_pred[r_y][r_x])
                pred_value_set.extend(middle_pred[r_y][r_x])
                pred_value_set.extend(small_pred[r_y][r_x])
                pred_value_set.extend(most_small_pred[r_y][r_x])
                print(pred_value_set)
                if not any(np.isnan(pred_value_set)):
                    X.append(pred_value_set)
                    Y.append(colormap_label[r_y][r_x])
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save(save_dir+'train/X/'+'X_'+str(image_n)+str(lerge_size)+'_'+str(middle_size)+'_'+str(small_size)+'_'+str(most_small_size), X)
    np.save(save_dir+'train/Y/'+'Y_'+str(image_n)+str(lerge_size)+'_'+str(middle_size)+'_'+str(small_size)+'_'+str(most_small_size), Y)
print('end')
