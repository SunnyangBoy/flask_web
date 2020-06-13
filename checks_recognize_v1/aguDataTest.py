from PIL import Image
import numpy as np
import math
import cv2
from shapely.geometry import Polygon
import random
from torchvision import transforms
from matplotlib import pyplot as plt


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices

def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def rotate_vertices(vertices, theta, anchor=None):
	'''rotate vertices around anchor
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
		theta   : angle in radian measure
		anchor  : fixed position during rotation
	Output:
		rotated vertices <numpy.ndarray, (8,)>
	'''
	v = vertices.reshape((4,2)).T
	if anchor is None:
		anchor = v[:,:1]
	rotate_mat = get_rotate_mat(theta)
	res = np.dot(rotate_mat, v - anchor)
	return (res + anchor).T.reshape(-1)


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
    	img         : PIL Image
    	vertices    : vertices of text regions <numpy.ndarray, (n,8)>
    	angle_range : rotate range
    Output:
    	img         : rotated PIL Image
    	new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
    	new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices

def is_cross_text(start_loc, length, vertices):
	'''check if the crop image crosses text regions
	Input:
		start_loc: left-top position
		length   : length of crop image
		vertices : vertices of text regions <numpy.ndarray, (n,8)>
	Output:
		True if crop image crosses text region
	'''
	if vertices.size == 0:
		return False
	start_w, start_h = start_loc
	a = np.array([start_w, start_h, start_w + length, start_h, \
          start_w + length, start_h + length, start_w, start_h + length]).reshape((4,2))
	p1 = Polygon(a).convex_hull
	for vertice in vertices:
		p2 = Polygon(vertice.reshape((4,2))).convex_hull
		inter = p1.intersection(p2).area
		if 0.01 <= inter / p2.area <= 0.99:
			return True
	return False


def crop_img(img, vertices):
    start_x = random.randint(0, 10)
    start_y = random.randint(0, 10)

    img = np.array(img)

    h, w = img.shape[:2]
    vertices[:, [0, 2, 4, 6]] -= start_x
    vertices[:, [1, 3, 5, 7]] -= start_y
    img[:h-start_y, :w-start_x] = img[start_y:, start_x:]
    img = Image.fromarray(img)
    return img, vertices

def visual(image, vertices):
    image = np.array(image)
    for vertice in vertices:
        box = []
        xlist = []
        for site in vertice[0: 9: 2]:
            xlist.append(int(site))
        print('xlist ', xlist)
        ylist = []
        for site in vertice[1: 9: 2]:
            ylist.append(int(site))
        print('ylist ', ylist)
        for i in range(4):
            box.append([int(xlist[i]), int(ylist[i])])
        print('box', box)
        box = np.array(box)
        image = cv2.polylines(image, [box], True, (0, 255, 0))
    print('--------------------------')
    cv2.imshow('Draw', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    file_path = '/Users/zhuzhenyang/Downloads/train/three/Label/receipt_img_tm3_00335.txt'
    image_path = '/Users/zhuzhenyang/Downloads/train/three/Image/receipt_img_tm3_00335.jpg'
    #mage = cv2.imread(image_path)
    img = Image.open(image_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        vertices = []
        for line in lines:
            line = line.split(';')
            vertice = [int(vt) for vt in line[1:-1]]
            vertices.append(vertice)

        vertices = np.array(vertices)
        #print(vertices)
        visual(img, vertices)

        img, vertices = adjust_height(img, vertices)#数据增强
        img, vertices = rotate_img(img, vertices)#数据增强
        w, h = img.size
        img = img.resize((512, 512))
        ratio_w, ratio_h = 512 / w, 512 / h
        vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h
        img, vertices = crop_img(img, vertices)  # 数据增强

        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3),
            transforms.ColorJitter(contrast=0.3),
        ])

        visual(transform(img), vertices)


