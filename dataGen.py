import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
import glob


def rotate90(data_path):
    for file in glob.glob(data_path + "/*"):
        file_name = file.split("/")[-1]
        img = cv2.imread(file, 0)
        img1 = np.rot90(img, k=1)
        img2 = np.rot90(img, k=3)
        cv2.imwrite("%s.rot1_%s" % (data_path, file_name), img1)
        cv2.imwrite("%s.rot2_%s" % (data_path, file_name), img2)


if __name__ == '__main__':

    data_path = "train/"
    des_path = "gen/"
    target_size = 512
    batch_size = 16

    # offline prepare
    class1 = 'HAND'
    rot90 = False
    if rot90:
        rotate90(os.path.join(data_path, class1))

    des = os.path.join(des_path, class1)
    if not os.path.exists(des):
        os.mkdir(des)

    datagen = ImageDataGenerator(rotation_range=0., 
                                 width_shift_range=0., 
                                 height_shift_range=0., 
                                 brightness_range=None, 
                                 zoom_range=0., 
                                 fill_mode='constant', 
                                 cval=0., 
                                 horizontal_flip=False, 
                                 vertical_flip=False, 
                                 rescale=None)

    generator = datagen.flow_from_directory(
        directory=data_path, classes=[class1],
        target_size=(target_size, target_size), 
        color_mode='grayscale', 
        class_mode='categorical', 
        batch_size=batch_size, 
        save_to_dir=des, save_prefix=class1)


    for i, batch in enumerate(generator):
        print(i)
        if i > 10:         # (i+2)*min(filenum, batch)
            break










