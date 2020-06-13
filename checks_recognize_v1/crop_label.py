import os
import numpy as np
import math
import csv
from PIL import Image

def get_boundary(vertices):
	'''get the tight boundary around given vertices
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		the boundary
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	x_min = min(x1, x2, x3, x4)
	x_max = max(x1, x2, x3, x4)
	y_min = min(y1, y2, y3, y4)
	y_max = max(y1, y2, y3, y4)
	return x_min, x_max, y_min, y_max


def cal_distance(x1, y1, x2, y2):
	'''calculate the Euclidean distance'''
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def cal_error(vertices):
	'''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
	calculate the difference between the vertices orientation and default orientation
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		err     : difference measure
	'''
	x_min, x_max, y_min, y_max = get_boundary(vertices)
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
	return err


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


def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi

def rotate_img(img, vertice, angle):
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
	img = img.rotate(angle, Image.BILINEAR)
	new_vertice = np.zeros(vertice.shape)
	new_vertice[:] = rotate_vertices(vertice, - angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	return img, new_vertice

# 原数据集，valid_rotated_mode
trainImgPath = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Image'
trainLabelPath = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Label'

# 生成数据集，valid_croped_mode, 需要自己手动创建文件夹valid_croped_mode3和并在其下面创建Image文件夹
createImg_root = '/home/chen-ubuntu/Desktop/checks_dataset/valid_crop_mode3/Image'

csvFile = open("/home/chen-ubuntu/Desktop/checks_dataset/valid_crop_mode3/Label.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(["file_name", "label", "type"])

# 模板三使用
typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '手写数字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']
# 模板二使用
#typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '字符', '印刷汉字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']
# 模板一使用
#typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']


for root, dirs, files in os.walk(trainLabelPath):
    for file in files:
        file_path = os.path.join(root, file)
        image_name = file[0: -4]+'.jpg'
        image_path = os.path.join(trainImgPath, image_name)
        #print(image_path)
        #file_path = '/Users/zhuzhenyang/Downloads/train/模版一/Label/receipt_img_tm1_00003.txt'
        #image_path = '/Users/zhuzhenyang/Downloads/train/模版一/Image/receipt_img_tm1_00003.jpg'
        #file = 'receipt_img_tm1_00003.jpg'
        image = Image.open(image_path)
        #if False:
        #    image = image.rotate(180, Image.BILINEAR)
        #plt.imshow(image)
        #plt.show()
        with open(file_path, 'r') as lines:
            lines = lines.readlines()
            for l, line in enumerate(lines):
                line = line.split(';')
                img_newname = file[0:-4] + '_' + '%02d' % l + '.jpg'
                print(img_newname)
                label = line[-1][:-1]
                type = typelist[l]
                writer.writerow([img_newname, label, type])
                #print('text: ', line[-1])

                vertice = [int(vt) for vt in line[1:-1]]
                vertice = np.array(vertice)
                #if False:
                #    center_x = (image.width - 1) / 2
                #    center_y = (image.height - 1) / 2
                #    new_vertice = np.zeros(vertice.shape)
                #    new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
                #    vertice = new_vertice
                #print('origin vertices ', vertice)

                theta = find_min_rect_angle(vertice)
                #print('theta: ', theta)
                img, vertice = rotate_img(image, vertice,  - theta / math.pi * 180)
                x_min, x_max, y_min, y_max = get_boundary(vertice)
                #plt.imshow(img)
                #plt.show()
                #print(int(x_min), int(x_max), int(y_min), int(y_max))
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                img_savepath = os.path.join(createImg_root, img_newname)
                img.save(img_savepath)
                #plt.imshow(img)
                #plt.show()

    csvFile.close()