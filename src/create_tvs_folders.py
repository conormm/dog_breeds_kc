# seperates training data into sample, train and validation folders
# creates torch friendly folder structure, e.g:
#root/dog/xxx.png
#root/dog/xxy.png
#root/dog/xxz.png

#root/cat/123.png
#root/cat/nsdf3.png
#root/cat/asd932_.png

import numpy as np
import os
from os.path import join
import pandas as pd
import pathlib
import shutil
from src.utils_cm import mk_label_folders, img2folder, copy_images_to_new

sampledir = "data/sample"
traindir = "data/train"
valdir = "data/val"

train_files = os.listdir(traindir)
labels = pd.read_csv("data/labels.csv")
labels["img_name"] = labels["id"] + ".jpg"
lab_img = (labels["breed"] + "/" + labels["img_name"]).tolist()

# create breed folders in train
mk_label_folders(traindir, classes=labels.breed)
mk_label_folders(sampledir, classes=labels.breed)
mk_label_folders(valdir, classes=labels.breed)

# move training files into train folders
img2folder(traindir, labels)
# move sample files to sample folders

sample_imgs = sample_images(labels)
sample_imgs_cat = (sample_imgs["breed"] + "/" + sample_imgs["img_name"]).tolist()
for img in sample_imgs_cat:
    if not os.path.exists(join(valdir, img)):
        shutil.move(join(traindir, img), join(valdir, img))  

# test
county = 0
for folder in os.listdir(sampledir):
    try:
        county += len(os.listdir(join(sampledir, folder)))
    except:
        pass

# make validation set
val_imgs_df = sample_images(labels, n=2000)
val_imgs = (val_imgs_df["breed"] + "/" + val_imgs_df["img_name"]).tolist()
for img in val_imgs:
    if not os.path.exists(join(valdir, img)):
        shutil.move(join(traindir, img), join(valdir, img))  


