import Tkinter
import subprocess as sub
from skimage import io, img_as_float
import numpy as np
import os
import pickle
import glob
from PIL import Image
import time
import cv2
from matplotlib import pyplot as plt
import scipy.misc


training_set_image = np.empty(shape=[0, 784])
training_set_labels = np.ndarray((0, ), int)

test_set_image = np.empty(shape=[0, 784])
test_set_labels = np.ndarray((0, ), int)

def conversion(dir_name):
    for i in range(0, 10):
        str_i = str(i)

        write("Currently in directory %s/%s" % (dir_name, str_i))

        for filename in os.listdir(dir_name + str_i):
            write(filename)
            image = io.imread(dir_name + str_i + "/" + filename)
            image = img_as_float(image)
            files = os.path.splitext(filename)
            fullname = os.path.join(dir_name, str_i, files[0] + "." + "jpg")
            os.remove(os.path.join(dir_name, str_i, files[0] + "." + "png"))
            io.imsave(fullname, image)

def segment_images():
    img = cv2.imread('/home/killwithme/Desktop/projects/final_year_project/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    rows = len(cells)
    cols = len(cells[0])
    k = 0
    image_num = 0
    initial_dir = '/home/killwithme/Desktop/projects/final_year_project/training-images'

    # creating initial directories
    if not os.path.exists('/home/killwithme/Desktop/projects/final_year_project/training-images'):
        os.makedirs('/home/killwithme/Desktop/projects/final_year_project/training-images')
        write('created directory training-images')
        os.makedirs('/home/killwithme/Desktop/projects/final_year_project/test-images')
        write('created directory test-images')
    else:
        write('Folders already exist')

    initial_dir = '/home/killwithme/Desktop/projects/final_year_project/test-images'
    inside_dir = 0
    image_num = 0

    # creating inside directories and saving images as JPEG
    for i in range(0, rows):
        for j in range(0, cols):
            if(image_num % 250 == 0):
                if(initial_dir == '/home/killwithme/Desktop/projects/final_year_project/training-images'):
                    initial_dir = '/home/killwithme/Desktop/projects/final_year_project/test-images'
                else:
                    initial_dir = '/home/killwithme/Desktop/projects/final_year_project/training-images'

                if not os.path.exists(initial_dir + '/' + str(k)):
                    os.makedirs(initial_dir + '/' + str(k))
                    write('created directory ' + initial_dir + '/' + str(k))
                else:
                    write('Folder already exists')

            scipy.misc.imsave(initial_dir + '/' + str(k) + '/' + str(image_num) + '.jpg', cells[i][j])
            image_num += 1
            if(image_num % 500 == 0):
                k += 1
                
def convert_grayscale_to_float64():

    write("Starting conversion from Grayscale to float64")

    dirName = "test-images/"
    conversion(dirName)
    dirName = "training-images/"
    conversion(dirName)

    # for i in range(1, 10):
    #     str_i = str(i)
    #     for filename in os.listdir(dir_name + str_i):
    #         write(filename)
    #         image = io.imread(dir_name + str_i + "/" + filename)
    #         image = img_as_float(image)
    #         files = os.path.splitext(filename)
    #         fullname = os.path.join(dir_name, str_i, files[0] + "." + "jpg")
    #         os.remove(os.path.join(dir_name, str_i, files[0] + "." + "png"))
    #         io.imsave(fullname, image)


    write("Images converted successfully")

def create_set(setValue):   #setValue = 1 => training_set, setValue = 2 => test_set
    
    if(setValue == 1):
        targetFolder = "training-images"
    else:
        targetFolder = "test-images"

    numFolders = 10

    for index in range(0, numFolders):

        label = index

        for file in glob.glob("./" + targetFolder + "/" + str(index) + "/*.jpg"):
            image = io.imread(file)
            image = img_as_float(image)
            li = []
            for x in range(0, 28):
                for y in range(0, 28):
                    li.append(image[x][y])

            if(setValue == 1):
                global training_set_image
                global training_set_labels
                training_set_image = np.append(training_set_image, [li], axis=0)
                training_set_labels = np.append(training_set_labels, [label ], axis=0)
            else:
                global test_set_image
                global test_set_labels
                test_set_image = np.append(test_set_image, [li], axis=0)
                test_set_labels = np.append(test_set_labels, [label ], axis=0)                

        index = index + 1


def create_training_set():
    create_set(1)

def create_test_set():
    create_set(2)

def pickleData():
    create_training_set()
    create_test_set()

    set_list = [(training_set_image, training_set_labels), (test_set_image, test_set_labels)]

    PIK = "/home/killwithme/Desktop/projects/final_year_project/custom-data-pickle.dat"

    with open(PIK, "ab") as fileOpen:
        pickle.dump(set_list, fileOpen)

    write("Pickle file created")

def write(string):
    global text_box
    text_box.config(state=Tkinter.NORMAL)
    text_box.insert("end", string + "\n")
    text_box.see("end")
    text_box.config(state=Tkinter.DISABLED)


def normalise():
    time.sleep(2)
    write("Images normalised successfully")

def close_window(): 
    root.destroy()

def center_window(width=300, height=200):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))



root = Tkinter.Tk()
root.title('Optical Character Recognition Preprocessing')
# WINDOW_SIZE = "600x400"
# root.geometry(WINDOW_SIZE)
center_window(800, 600)

text_box = Tkinter.Text(root, state=Tkinter.DISABLED)
text_box.grid(row=0, column=1, columnspan=8)

button_1 = Tkinter.Button(root, text="Segment Images", command=segment_images)
button_1.grid(row=1, column=1)

button_2 = Tkinter.Button(root, text="Resize Images", command=lambda: sub.call('bash ./Desktop/projects/final_year_project/resize-script.sh', shell=True))
# button_1 = Tkinter.Button(root, text="Resize Images", command=resizeImages)
button_2.grid(row=1, column=2)

button_3 = Tkinter.Button(root, text="Convert Grayscale to float64", command=convert_grayscale_to_float64)
button_3.grid(row=1, column=3)

button_4 = Tkinter.Button(root, text="Normalize Images", command=normalise)
button_4.grid(row=1, column=4)

button_5 = Tkinter.Button(root, text="Create Pickle File", command=pickleData)
button_5.grid(row=1, column=5)

button_6 = Tkinter.Button(root, text="Exit", command=close_window)
button_6.grid(row=1, column=6)

root.mainloop()