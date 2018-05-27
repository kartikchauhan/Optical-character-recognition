import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.misc
import os

img = cv2.imread('digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

rows = len(cells)
cols = len(cells[0])
k = 0
image_num = 0
initial_dir = 'training-images'

# creating initial directories
if not os.path.exists('training-images'):
    os.makedirs('training-images')
    print('created directory training-images')
    os.makedirs('test-images')
    print('created directory test-images')
else:
    print('Folders already exist')

initial_dir = 'test-images'
inside_dir = 0
image_num = 0

# creating inside directories and saving images as JPEG
for i in range(0, rows):
    for j in range(0, cols):
        if(image_num % 250 == 0):
            if(initial_dir == 'training-images'):
                initial_dir = 'test-images'
            else:
                initial_dir = 'training-images'

            if not os.path.exists(initial_dir + '/' + str(k)):
                os.makedirs(initial_dir + '/' + str(k))
                print('created directory ' + initial_dir + '/' + str(k))
            else:
                print('Folder already exists')

        scipy.misc.imsave(initial_dir + '/' + str(k) + '/' + str(image_num) + '.jpg', cells[i][j])
        image_num += 1
        if(image_num % 500 == 0):
            k += 1




            




