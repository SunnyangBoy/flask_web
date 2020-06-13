import os
import torch
import configs
from model import DetectAngleModel
import numpy as np
from PIL import Image
import random
import math
from matplotlib import pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

createImg_root = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Image/'
createLab_root = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Label/'

def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def rotate_img(img_dir, img_savepath, file_path, lab_savepath, flag):
    image = Image.open(img_dir)
    if flag:
        image = image.rotate(180, Image.BILINEAR)
    image.save(img_savepath)


    with open(lab_savepath, 'w') as writer:
        with open(file_path, 'r') as lines:
            lines = lines.readlines()
            for l, line in enumerate(lines):
                line = line.split(';')
                vertice = [int(vt) for vt in line[1:-1]]
                vertice = np.array(vertice)
                if flag:
                    center_x = (image.width - 1) / 2
                    center_y = (image.height - 1) / 2
                    new_vertice = np.zeros(vertice.shape)
                    new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
                    vertice = new_vertice
                new_line = []
                new_line.append(line[0])
                for v in vertice:
                    new_line.append(str(int(v)))
                new_line.append(line[-1])
                new_line = ';'.join(new_line)
                writer.write(new_line)
        writer.close()

if __name__ == '__main__':

    model = DetectAngleModel()
    model.load_state_dict(torch.load('/home/chen-ubuntu/Desktop/checks_dataset/pths/rotate.pth'))
    model.to(device)
    model.eval()

    img_mode = 'three'
    img_label_dir = os.path.join(configs.img_rootdir, img_mode, 'Image')#'Label')

    with open('/home/chen-ubuntu/Desktop/checks_dataset/res_det/test_rotate.txt', 'a') as writer:
        for root, dirs, files in os.walk(img_label_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                img_name = file#[0: -4]+'.jpg'
                print(img_name)
                img_dir = file_path#os.path.join(configs.img_rootdir, img_mode, 'Image', img_name)
                img = Image.open(img_dir).convert('L')

                width = img.width
                height = img.height

                img = img.resize((224, 224))
                #plt.imshow(img)
                #plt.show()
                img = np.array(img)
                img = img / 255
                img = torch.from_numpy(img).float()
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                img = img.to(device)
                output = model(img)
                flag = True
                if output[0][0] < output[0][1]:
                    print('image_pred = 1')
                    writer.write(img_name + ';')
                    writer.write('1')
                    writer.write(str(width) + ',' + str(height))
                    writer.write('\n')
                    flag = False
                else:
                    print('image_pred = 0')
                    writer.write(img_name + ';')
                    writer.write('0')
                    writer.write(str(width) + ',' + str(height))
                    writer.write('\n')

                '''
                img_savepath = os.path.join(createImg_root, img_name)
                lab_savepath = os.path.join(createLab_root, file)
                rotate_img(img_dir, img_savepath, file_path, lab_savepath, flag)
                '''