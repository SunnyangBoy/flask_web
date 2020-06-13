import cv2
import numpy as np
import random
import pandas as pd
import os
import csv

def catimg(image1,image2):
    h1,w1,c1 = image1.shape
    h2,w2,c2 = image2.shape
    if c1 != c2:
        print("channels NOT match, cannot merge")
        return
    else:
        if h1 == h2:
            image3 = np.hstack([image1, image2])
        else:
            print('the height of two images are different')
    return image3

def cutimg(image, label):
    num = len(label)
    h, w, _ = image.shape
    cut = 0
    width = w // num
    imgs = []
    for i in range(num):
        if i == 0:
            img = image[:, cut:cut+width+1, :]
        elif i == num-1:
            img = image[:, cut-1:cut+width, :]
        else:
            img = image[:, cut-1:cut+width+1, :]
        #cv2.imwrite('cut_' + str(i) + '.jpg', img)
        cut += width
        imgs.append(img)

    state = np.random.get_state()
    np.random.shuffle(imgs)
    np.random.set_state(state)
    label = list(label)
    np.random.shuffle(label)

    imgbase = imgs[0]
    newlabel = label[0]
    for i in range(num-1):
        i += 1
        imgbase = catimg(imgbase, imgs[i])
        newlabel += label[i]
    #cv2.imwrite('cut_base.jpg', imgbase)
    return imgbase, newlabel

def skip(label):
    chars = '()-.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for char in label:
        if char in chars:
            return True
    return False

if __name__ == "__main__":

    csvFile = open("/home/chen-ubuntu/Desktop/checks_dataset/stamp_merge/Label.csv", "w")
    writer = csv.writer(csvFile)
    writer.writerow(["file_name", "label", "type"])

    img_infos = pd.read_csv('/home/chen-ubuntu/Desktop/checks_dataset/crop_merge/Label.csv', dtype='str')
    type = '印章汉字'
    features = ['file_name', 'label', 'type']
    img_infos = img_infos[features]
    seal_infos = img_infos[img_infos.type == type]
    seal_infos = np.array(seal_infos)
    for seal in seal_infos:
        label = seal[1]
        if label == '财务专用章' or skip(label):
            continue
        img_name = seal[0]
        img_path = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/crop_merge/Image', img_name)
        image = cv2.imread(img_path)
        print(img_name)
        #cv2.imwrite(os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/stamp_merge/Image/', img_name), image)
        for i in range(2):
            new_img, new_label = cutimg(image, label)
            new_name = img_name[:-4] + '_%02d' % i + '.jpg'
            writer.writerow([new_name, new_label, type])
            new_dir = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/stamp_merge/Image', new_name)
            cv2.imwrite(new_dir, new_img)

    csvFile.close()

