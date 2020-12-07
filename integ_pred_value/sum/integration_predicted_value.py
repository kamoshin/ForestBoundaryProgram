import numpy as np
import os
from cv2 import cv2

def makeDir(dir_path, mode='keep'):
    if mode == 'remove':
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    elif mode == 'keep':
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

image_list = ['ushitsu', 'yuusenji']
# report_dir = './../../Data/Report/DenseNet/64+128+256/64[1,1,1,5]+128+256/'
report_dir = './../../../Data/Report/DenseNet/32/'
# weight = '128[1,1,1,10]+256/'
# weight = [1,1,1,2]
weight_path = '32'
report_dir = report_dir + weight_path + '/'
pred_dir = './../../../Data/pred_value/'

makeDir(report_dir)

color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128],
                [0, 0, 0]
                ]

for image_n in image_list:
    pred_keeper32 = np.nan_to_num(np.load(pred_dir+'DenseNet32/'+image_n+'_32_predicted_keeper.npy'))
    # pred_keeper64 = np.nan_to_num(np.load(pred_dir+'DenseNet64/'+image_n+'_64_predicted_keeper.npy'))
    # pred_keeper128 = np.nan_to_num(np.load(pred_dir+'DenseNet128/'+image_n+'_128_predicted_keeper.npy'))
    # pred_keeper256 = np.nan_to_num(np.load(pred_dir+'DenseNet256/'+image_n+'_256_predicted_keeper.npy'))
    height, width, channels = pred_keeper32.shape[:3]
    # print(pred_keeper32.shape)
    colormap = np.zeros((height, width, 3))
    # pred_keeper32 = pred_keeper32 * weight
    # integ_pred_keeper = pred_keeper32 + pred_keeper64 + pred_keeper128 + pred_keeper256
    integ_pred_keeper = pred_keeper32
    bg_map = cv2.inRange(pred_keeper32, (0,0,0,0), (0,0,0,0))
    for n in range(len(color_dict) - 1):
        colormap[np.where(np.argmax(integ_pred_keeper, axis=2) == n)] = color_dict[n]
    colormap[np.where(bg_map == 255)] = color_dict[len(color_dict)-1]
    print('\nwrite colormap image -> png')
    image_name = image_n.replace('.jpg','')
    cv2.imwrite(report_dir + image_name + '_Dense'+weight_path+'_pred_colormap.png', colormap)

#     print("time : {0:.1f}[sec]".format(time.time()-img_s_time))

# print("time : {0:.1f}[sec]".format(time.time()-s_time))
