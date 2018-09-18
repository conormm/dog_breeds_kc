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
from src.utils_cm import ModelParameters

images_dir = "data/sample"
NUM_EPOCHS = 3
IMG_SIZE = 250
# these are standard pytorch values for image normalization
normmean = [0.485, 0.456, 0.406]
normstd = [0.229, 0.224, 0.225]


def fine_tuning_model(model, n_classes=120):
    ModelParameters.freeze_all(model.parameters())
    assert ModelParameters.all_frozen(model.parameters())
    model.ft_layer = nn.Linear(1000, n_classes)
    assert model.ft_layer.weight.requires_grad
    return model

train_trans = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(224),
    transforms.ColorJitter(.3, .3, .3),
    transforms.RandomHorizontalFlip(p=.3), 
    transforms.ToTensor(), 
    transforms.g
    transforms.Normalize(normmean, normstd)
    ])

val_trains = transforms.Compose([
    transforms.Resize(IMG_SIZE), 
    transforms.CenterCrop(), 
    transforms.ToTensor(), 
    transforms.Normalize()
    ])

img_f = ImageFolder(images_dir, transform=train_trans)
n_classes = len(img_f.classes)
ds = DataLoader(img_f, batch_size=28, shuffle=True)

VGG16 = models.vgg16(pretrained=True)

optim = torch.optim.Adam(
    ModelParameters.get_trainable(VGG16.parameters()), 
    lr=0.001
    )
criterion = nn.CrossEntropyLoss()

VGG16.train()

for epoch in range(NUM_EPOCHS):
    for ix, (X, y) in enumerate(ds):

        optim.zero_grad()
        X.requires_grad = True

        preds = VGG16(X)
        loss = criterion(preds, y)

        loss.backward()
        optim.step()

        print(f"Loss: {loss.item()}")
        
preds.shape

X.shape
y.shape
