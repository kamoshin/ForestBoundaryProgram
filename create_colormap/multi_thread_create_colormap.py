import numpy as np
from cv2 import cv2

import threading
import time
import sys

pred_value_path = './../../Data/pred_value/DenseNet128/ushitsu_128_predicted_keeper.npy'
pred_value_map = np.load(pred_value_path)
height, width = pred_value_map.shape[:2]
colormap = np.zeros((height, width, 3))
color_dict = [
                [0, 0, 255],
                [0, 128, 0],
                [0, 255, 255],
                [128, 128, 128]
                ]
half_height = int(height//5)
print(half_height)


def proc1():
    for y1 in range(0, half_height):
        for x1 in range(width):
            sys.stdout.write("\r y : %d" % y1)
            sys.stdout.flush()
            if not any(np.isnan(pred_value_map[y1][x1])):
                colormap[y1][x1] = color_dict[np.argmax(pred_value_map[y1][x1])]

def proc2():
    for y2 in range(half_height+1, half_height*2):
        for x2 in range(width):
            if not any(np.isnan(pred_value_map[y2][x2])):
                colormap[y2][x2] = color_dict[np.argmax(pred_value_map[y2][x2])]

def proc3():
    for y3 in range(half_height*2+1, half_height*3):
        for x3 in range(width):
            if not any(np.isnan(pred_value_map[y3][x3])):
                colormap[y3][x3] = color_dict[np.argmax(pred_value_map[y3][x3])]

def proc4():
    for y4 in range(half_height*3+1, half_height*4):
        for x4 in range(width):
            if not any(np.isnan(pred_value_map[y4][x4])):
                colormap[y4][x4] = color_dict[np.argmax(pred_value_map[y4][x4])]

def proc5():
    for y5 in range(half_height*4+1, height):
        for x5 in range(width):
            if not any(np.isnan(pred_value_map[y5][x5])):
                colormap[y5][x5] = color_dict[np.argmax(pred_value_map[y5][x5])]


if __name__ == "__main__":
    start_time = time.time()
    print(colormap.shape)

    th1 = threading.Thread(target=proc1)
    th2 = threading.Thread(target=proc2)
    th3 = threading.Thread(target=proc3)
    th4 = threading.Thread(target=proc4)
    th5 = threading.Thread(target=proc5)

    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    

    th1.join()
    th2.join()
    th3.join()
    th4.join()
    th5.join()

cv2.imwrite('./../../Data/Report/DenseNet128/predicted_colormap.png', colormap)
elapsed_time = time.time() - start_time
print(" ")
print(elapsed_time)