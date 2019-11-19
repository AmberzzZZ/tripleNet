import os
import glob
import shutil
import random


train_pt = "data/train/"
val_pt = "data/val"
test_pt = "data/test"

if not os.path.exists(val_pt):
    os.mkdir(val_pt)
    os.mkdir(test_pt)


for cls_folder in os.listdir(train_pt):
    if cls_folder[0] == '.':       # .DS_Store
        continue
    print("cls folder: ", cls_folder)
    file_lst = glob.glob(os.path.join(train_pt, cls_folder) + "/*png")
    random.shuffle(file_lst)
    length = len(file_lst)

    # for val
    des = os.path.join(val_pt, cls_folder)
    if not os.path.exists(des):
        os.mkdir(des)
    for file in file_lst[:length//10]:
        shutil.move(file, des)

    # for test
    des = os.path.join(test_pt, cls_folder)
    if not os.path.exists(des):
        os.mkdir(des)
    for file in file_lst[length//10:length//5]:
        shutil.move(file, des)

