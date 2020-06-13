import os
import configs
import random
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

trainImgPath = '/Users/zhuzhenyang/Downloads/train/模版一/Image'
trainLabelPath = '/Users/zhuzhenyang/Downloads/train/模版一/Label'

#if os.path.exists('./flipLabel.txt'):
#    os.remove('./flipLabel.txt')

fliped_num = 0
unfliped_num = 0

#for root, dirs, files in os.walk(trainLabelPath):
#    with open('./flipLabel.txt', 'a') as lb:
#        for file in files[200:300]:
#            imagename = file[0: -4]+'.jpg'
#            imagepath = os.path.join(trainImgPath, imagename)
#            lb.write('1;' + imagename + ';')
#            image = cv2.imread(imagepath)
#            plt.imshow(image)
#            plt.show()
#            key = input(imagename + ': ')
#            if key == '1':
#                unfliped_num += 1
#                lb.write('1\n')
#            else:
#                fliped_num += 1
#                lb.write('0\n')
#        print('fliped_num :', fliped_num)
#        print('unfliped_num  :', unfliped_num)
#        lb.close()


images = []
with open(configs.label_dir, 'r') as labels:
    f = 0
    unf = 0
    for lb in labels.readlines():
        annotation = {}
        lblist = lb.split(';')
        annotation['mode'] = lblist[0]
        annotation['name'] = lblist[1]
        annotation['class'] = int(lblist[2][0])
        images.append(annotation)
        print(annotation['mode'], annotation['name'], annotation['class'])
        if lblist[2][0] == '0':
            f += 1
        else:
            unf += 1
    print('总数：', len(images))
    print('正面：', unf)
    print("反面：", f)


maxW = 0
maxH = 0
minW = 10000
minH = 10000
W = []
H = []
for i in range(617):
    #i = random.randint(0, 617)
    img_name = images[i]['name']
    mode = images[i]['mode']
    if mode == '1':
        img_mode = '模版一'
    elif mode == '2':
        img_mode = '模版二'
    elif mode == '3':
        img_mode = '模版三'
    img_dir = os.path.join(configs.img_rootdir, img_mode, 'Image', img_name)
    img = Image.open(img_dir)
    #plt.imshow(img)
    #plt.show()
    img = np.array(img)
    h, w = img.shape[:2]
    W.append(w)
    H.append(h)
    if w > maxW:
        maxW = w
    if w < minW:
        minW = w
    if h > maxH:
        maxH = h
    if h < minH:
        minH = h
    #print('class: ', images[i]['class'])
print('w:', W)
print('lenW', len(W))
print('h', H)
print('lenH', len(H))
print('maxW', maxW)
print('minW', minW)
print('maxH', maxH)
print('minH', minH)


#i = random.randint(0, 617)
#img_name = images[i]['name']
#mode = images[i]['mode']
#if mode == '1':
#    img_mode = '模版一'
#elif mode == '2':
#    img_mode = '模版二'
#elif mode == '3':
#    img_mode = '模版三'
#img_dir = os.path.join(configs.img_rootdir, img_mode, 'Image', img_name)
#img = Image.open(img_dir)
#plt.imshow(img)
#plt.show()
#print('class: ', images[i]['class'])
#img = np.array(img)
#img = img/255
#img = torch.from_numpy(img).float()
#print(img.shape)    #(h, w, chanel)
#img = torch.unsqueeze(img, 0)   #(batch, h, w, chanel)
#print(img.shape)
#

