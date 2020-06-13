import random 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import shutil 
import os 
import cv2 
from PIL import Image
import numpy as np


def img_nomalize(img, type):
    if type == 'word':
        size = (32, 450)
    elif type == 'num':
        size = (32, 150)
    elif type == 'handword':
        size = (32, 450)
    elif type == 'handnum':
        size = (32, 250)
    elif type == 'char':
        size = (32, 32)
    elif type == 'seal':
        size = (32, 100)
    else:
        return None

    img = img_padder(img, size)
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    img = Image.fromarray(img)
    toTensor = transforms.Compose([transforms.ToTensor()])
    img = toTensor(img)
    return img



def scale(img):
    w, h = img.size
    image = np.array(img)
    scaleh = np.random.randint(int(h*0.4), int(h*0.8))
    scalew = np.random.randint(int(w*0.4), int(w*0.8))
    image = cv2.resize(image, (scalew, scaleh))
    image = Image.fromarray(image)
    return image


def gaussi(img):
    image = np.array(img)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = Image.fromarray(image)
    return image


def shadow(img):
    w, h = img.size
    startx = random.randint(0, w)
    width = random.randint(w//5, w//2)
    gray = random.randint(50, 150)
    image = np.array(img)
    if startx + width > w:
        endx = w
    else:
        endx = startx + width
    for x in range(startx, endx):
        for y in range(h):
            value = image[y][x]
            if value - gray > 0:
                image[y][x] = value - gray
            else:
                image[y][x] = 0
    image = Image.fromarray(image)
    return image


def lines(img):
    w, h = img.size
    img = np.array(img)
    # vertical
    for i in range(10):
        startx = random.randrange(0, w, 20)
        width = random.randint(2, 4)
        gray = random.randint(40, 80)
        if startx + width > w:
            endx = w
        else:
            endx = startx + width
        for x in range(startx, endx):
            for y in range(h):
                value = img[y][x]
                if value - gray > 0:
                    img[y][x] = value - gray
                else:
                    img[y][x] = 0
    # horizontal
    for i in range(5):
        starty = random.randrange(0, w, 10)
        height = random.randint(2, 4)
        gray = random.randint(40, 80)
        if starty + height > h:
            endy = h
        else:
            endy = starty + height
        for y in range(starty, endy):
            for x in range(w):
                value = img[y][x]
                if value - gray > 0:
                    img[y][x] = value - gray
                else:
                    img[y][x] = 0
    image = Image.fromarray(img)
    return image


class RandomEnhance():
    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms
    def __call__(self, img):
        sub_list = random.sample(self.transforms, 2)
        for t in sub_list:
            img = t(img)
        return img


def img_enhancer(x, size):

    if random.randint(0, 1):
        x = lines(x)
    if random.randint(0, 1):
        x = gaussi(x)
    if random.randint(0, 1):
        x = scale(x)

    w, h = x.size
    ratio = size[0] / h
    new_w = int(ratio * w)
    x = transforms.Resize(size=(size[0], min(new_w, size[1])))(x)

    pad_length = (size[1] - x.size[0])
    left_pad_length = pad_length // 2
    right_pad_length = pad_length - left_pad_length
    x = transforms.Pad((left_pad_length, 0, right_pad_length, 0), fill=(230,), padding_mode="constant")(x)
    tfs_list = [
        transforms.RandomCrop(size=size, padding=(3, 3, 3, 3), padding_mode='edge'),
        transforms.RandomRotation(3, expand=False),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
    ]
    x = RandomEnhance(tfs_list)(x)
    x = transforms.Resize(size=size)(x)
    return x


def img_padder(x, size):
    w, h = x.size
    ratio = size[0] / h

    new_w = int(ratio * w)
    x = transforms.Resize(size=(size[0], min(new_w, size[1])))(x)
    pad_length = (size[1] - x.size[0])
    left_pad_length = pad_length // 2
    right_pad_length = pad_length - left_pad_length
    x = transforms.Pad((left_pad_length, 0, right_pad_length, 0), fill=(230,), padding_mode="constant")(x)

    x = transforms.Resize(size=size)(x)
    return x


class BaseDataset(Dataset):
    def __init__(self, img_dir, img_infos=None, transform=None, _type=None):
        '''
        img_infos 为n*2的numpy数组，其中第一列为图片的名字， 第二列为label
        '''
        
        self.img_dir = os.path.join(img_dir, 'Image') 
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)
        self.img_infos = np.array(img_infos)
        self.type = _type
        if _type   == 'print_word':
            self.size = (32,450)
        elif _type == 'print_num':
            self.size = (32, 150)
        elif _type == 'hand_word':
            self.size = (32, 450)
        elif _type == 'hand_num':
            self.size = (32, 250)
        elif _type == 'symbol':
            self.size = (32, 32)
        elif _type == 'seal':
            self.size = (32, 100)
        

    def __len__(self):
        if self.img_infos is not None:
            return self.img_infos.shape[0]
        else:
            return len(self.img_list)

    def __getitem__(self, index):
        if self.img_infos is not None:
            img_name, label, _ = self.img_infos[index]
        else:
            img_name = self.img_list[index]
            label = None
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img, size=self.size)

        #img.save('test/'+str(index)+"_"+label+".jpg")

        img = (np.array(img)/255.0-0.5)/0.5
        img = Image.fromarray(img)
        toTensor = transforms.Compose([transforms.ToTensor()])
        img = toTensor(img)
        return img, label, img_name



