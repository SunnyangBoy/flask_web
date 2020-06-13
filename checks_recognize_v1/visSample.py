import cv2
import numpy as np

file_path = '/Users/zhuzhenyang/Downloads/train/one/Label/receipt_img_tm1_01413.txt'
image_path = '/Users/zhuzhenyang/Downloads/train/one/Image/receipt_img_tm1_01413.jpg'
print(image_path)
image = cv2.imread(image_path)
w, h = image.shape[:2]
print(w, h)
with open(file_path, 'r') as lines:
    for line in lines.readlines()[:-3]:
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
cv2.imwrite('./test.jpg', image)
#cv2.imshow('Draw', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




