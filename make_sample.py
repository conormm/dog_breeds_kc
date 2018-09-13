import numpy as np
import os
from os.path import join
import pandas as pd
import pathlib
import shutil

sampledir = "data/sample"
traindir = "data/train"
valdir = "data/val"

train_files = os.listdir(traindir)
labels = pd.read_csv("data/labels.csv")
labels["img_name"] = labels["id"] + ".jpg"

def mk_label_folders(idir, img_lbl_df, img="img_name", label="breed"):

    lbls = set(img_lbl_df[label])
    for l in lbls:
        if not os.path.isdir(join(idir, l)):
            os.mkdir(join(idir, l))
            print(f"Making {l} folder in {idir}")
    print("Done!")

def sample_images(images, n=1000):

    ix = np.random.choice(np.arange(len(images)), size=n, replace=False)
    sample = labels.loc[ix, ["img_name", "breed"]]
    assert len(sample) == n
    return sample


def img2folder(idir, img_lbl_df, label="breed", img="img_name"):
    """Creates a folder directory to match the torchvision ImageFolder 
    required structure:

    :param idir: directory to the image folder
    :img_lbl_df: a pd.DataFrame with image names and labels

    """
    lbls = set(img_lbl_df[label])

    for img, label in tuple(zip(img_lbl_df[img], img_lbl_df[label])):
        for folder in lbls:
            if label == folder:
                try:
                    shutil.move(join(idir, img), join(idir, folder, img))
                except FileNotFoundError:
                    print(f"{img} not found.")
                    pass

    print("Done")

def copy_images_to_new(lab_img, from_dir, to_dir):
    """Copies images files from one loc to another.
    :param lab_img: e.g. /collie/img1.jpg (label/image)
    :param from_dir: root dir of file's current location (e.g. train)
    :param to_dir: root dir of where you want to move the file
    """
    
    for img in lab_img
        if not os.path.exists(join(todir, img)):
            shutil.copyfile(join(fromdir, img), join(todir, img))    
    print("Done")

def lazy_list_files(path):
    f = iter(os.listdir(path))
    return f


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

# create breed folders in train
mk_label_folders(traindir, img_lbl_df=labels)
mk_label_folders(sampledir, img_lbl_df=labels)
mk_label_folders(valdir, img_lbl_df=labels)
# move training files into train folders
img2folder(traindir, labels)
# move sample files to sample folders

sample_imgs = sample_images(labels)
sample_imgs_cat = (sample_imgs["breed"] + "/" + sample_imgs["img_name"]).tolist()
sample_imgs.to_csv(join("data", "sample_labels.csv"))

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

val_imgs_df.to_csv(join("data", "validation_labels.csv"))        


