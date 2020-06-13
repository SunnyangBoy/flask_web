import torch
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
import os 
import numpy as np
import configs
from PIL import Image
import random

class DetectAngleDataset(Dataset):

    def __init__(self, img_rootdir, images, transforms=None):
        self.img_rootdir = img_rootdir
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert idx <= self.__len__(), "index out of range"
        img_name = self.images[idx]['name']
        mode = self.images[idx]['mode']
        if mode == '1':
            img_mode = '模版一'
        elif mode == '2':
            img_mode = '模版二'
        elif mode == '3':
            img_mode = '模版三'
        img_dir = os.path.join(configs.img_rootdir, img_mode, 'Image', img_name)
        img = Image.open(img_dir)
        img = img.convert("L")
        img = img.resize((224, 224))    #暂时未定
        if self.transforms != None:
            if random.randint(1, 3) != 2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = self.transforms(img)
        img = np.array(img)
        img = img/255
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img, 0)   #(batch, h, w, channel)
        img_class = self.images[idx]['class']
        return (img, img_class)

class RandomEnhance():
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
    def __call__(self, img):
        sub_list = random.sample(self.transforms, 3)
        for t in sub_list:
            img = t(img)
        return img

def img_enhancer(x):
    x = transforms.Resize(size=(224, 224))(x)
    tfs_list = [    
        transforms.RandomCrop(size=(224, 224), padding=10, padding_mode='edge'),
        transforms.Pad((random.randint(2, 30), random.randint(2, 30)), padding_mode="edge"),
        transforms.Pad((random.randint(10, 30), 0), padding_mode="edge"),
        transforms.Pad(0, (random.randint(10, 30)), padding_mode="edge"),
        transforms.RandomRotation(3, expand=False),
        transforms.ColorJitter(brightness=0.7),
        transforms.ColorJitter(contrast=0.7),
        #img_aug.RandomAddToHueAndSaturation(),
        # img_aug.RandomElastic(),
        #img_aug.RandomMotionBlur(),
        # img_aug.RandomPerspective(),
    ]
    x = RandomEnhance(tfs_list)(x)
    x = transforms.Resize(size=(224, 224))(x)
    return x