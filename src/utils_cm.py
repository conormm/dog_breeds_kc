import numpy as np
import os
from os.path import join
import pandas as pd
import pathlib
import shutil

def mk_label_folders(idir, classes):
    """Makes subfolders in folder idir using the unique names of the model classes
    """
    lbls = set(classes)
    for l in lbls:
        if not os.path.isdir(join(idir, l)):
            os.mkdir(join(idir, l))
            print(f"Making {l} folder in {idir}")
    print("Done!")

def sample_images(images, n=1000):
    """Randomly samples n images from the training files.
    """
    ix = np.random.choice(np.arange(len(images)), size=n, replace=False)
    sample = labels.loc[ix, ["img_name", "breed"]]
    assert len(sample) == n
    return sample

def img2folder(idir, img_lbl_df, label="breed", img="img_name"):
    """Creates a folder directory to match the torchvision ImageFolder 
    required structure.

    :param idir: directory to the image folder
    :img_lbl_df: a pd.DataFrame with image names and labels

    """
    lbls = set(img_lbl_df[label])

    for img, label in zip(img_lbl_df[img], img_lbl_df[label]):
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
    
    for img in lab_img:
        if not os.path.exists(join(todir, img)):
            shutil.copyfile(join(fromdir, img), join(todir, img))    
    print("Done")

def lazy_list_files(path):
    f = iter(os.listdir(path))
    return f

class ModelParameters:
    
    @staticmethod
    def get_trainable(params):
        return (p for p in params if p.requires_grad)
        
    @staticmethod    
    def get_frozen(params):
        return (p for p in params if not p.requires_grad)
    
    @staticmethod
    def freeze_all(params):
        for p in params:
            p.requires_grad = False
    
    @staticmethod
    def all_trainable(params):
        return all(p.requires_grad for p in params)
    
    @staticmethod
    def all_frozen(params):
        return all(not p.requires_grad for p in params)


