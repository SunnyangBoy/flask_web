import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

trainImgPath = '/Users/zhuzhenyang/Downloads/train/one/Image'
trainLabelPath = '/Users/zhuzhenyang/Downloads/train/one/Label'

for root, dirs, files in os.walk(trainLabelPath):
    for file in files[100 : 101]:
        file_path = os.path.join(root, file)
        image_name = file[0: -4]+'.jpg'
        print(image_name)
        image_path = os.path.join(trainImgPath, image_name)
        print(image_path)
        image = cv2.imread(image_path)
        w, h = image.shape[:2]
        print(w, h)
        with open(file_path, 'r') as lines:
            lines = lines.readlines()
            for line in lines[:-3]:
                box = []
                sites = line.split(';')
                xlist = []
                for site in sites[1: 9: 2]:
                    xlist.append(site)
                ylist = []
                for site in sites[2: 9: 2]:
                    ylist.append(site)
                for i in range(4):
                    box.append([int(xlist[i]), int(ylist[i])])
                print(box)
                box = np.array(box)
                image = cv2.polylines(image, [box], True, (0, 255, 0))
        print('--------------------------')
        #plt.imshow(image)
        #plt.show()
        cv2.imshow('Draw', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




