import os
import numpy as np
import math
import csv
import pandas as pd
from PIL import Image

if __name__ == '__main__':

    ImgPath = '/Users/zhuzhenyang/Desktop/img/'

    writcsv = open("/Users/zhuzhenyang/Documents/new_label.csv", "w")
    writer = csv.writer(writcsv)
    writer.writerow(["file_name", "label", "type"])

    img_infos = pd.read_csv('/Users/zhuzhenyang/Documents/Label.csv', dtype='str')

    list = list(img_infos.values)
    cnt = 0
    for item in list:
        img_name = item[0]
        print(img_name)
        img_path = os.path.join(ImgPath, img_name)
        if os.path.exists(img_path):
            writer.writerow(item)
            cnt += 1

    print('label cnt: ', cnt)

    print('img cnt', len(os.listdir(ImgPath)))
