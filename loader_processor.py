import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.datasets.utils import check_integrity
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt

IMG_SIZE = 250
normmean = [0.485, 0.456, 0.406]
normstd = [0.229, 0.224, 0.225]

images_dir = "data/sample"

img_f = ImageFolder(images_dir)
n_classes = len(img_f.classes)
ds = DataLoader(img_f, batch_size=28, shuffle=True)

trans = transforms.Compose([
    transforms.Resize(IMG_SIZE)
    transforms.RandomCrop(224),
    transforms.ColorJitter(.3, .3, .3),
    transforms.RandomHorizontalFlip(p=.3), 
    transforms.ToTensor(), 
    transforms.Normalize(normmean, normstd),
    
])

VGG16 = models.vgg16(pretrained=True)

VGG16.finallayer = nn.Linear(1000, n_classes)


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





def get_frozen(params):


p = get_trainable(VGG16.parameters())
next(p)