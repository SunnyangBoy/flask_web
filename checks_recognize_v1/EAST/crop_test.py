import math
import detect
import torch
from PIL import Image
from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np
from shapely.geometry import Polygon, MultiPoint


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


def rotate_allimg(img, vertices, angle):
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
	new_vertices = np.zeros(vertices.shape)
	for i, vertice in enumerate(vertices):
		new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	return img, new_vertices


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def sort_centers(list, aix):
    list = sorted(list.items(), key=lambda item: int(item[0].split(',')[aix]))
    return list

def sort_xcenters(list, aix):
    list = sorted(list, key=lambda item: int(item[0].split(',')[aix]))
    return list

if __name__ == '__main__':

    model_path = '/home/chen-ubuntu/Desktop/checks_dataset/pths/model3_epoch_14.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    trainImgPath = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Image'
    trainLabelPath = '/home/chen-ubuntu/Desktop/checks_dataset/valid_rotated_mode3/Label'

    #typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']
    #typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '字符', '印刷汉字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']
    typelist = ['手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写汉字', '手写数字', '手写数字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷汉字', '印刷数字',
                '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印刷数字', '印章汉字', '印章汉字', '印章汉字']

    wrong = 0
    all_cnt = 0
    if True:
        for root, dirs, files in os.walk(trainLabelPath):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                image_name = file[0: -4] + '.jpg'
                image_path = os.path.join(trainImgPath, image_name)

                image = Image.open(image_path)

                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    valid_vertices = []
                    for line in lines[:-3]:
                        line = line.split(';')
                        valid_vertice = [int(vt) for vt in line[1:-1]]
                        valid_vertice = np.array(valid_vertice)
                        valid_vertices.append(valid_vertice)

                img = image.convert("RGB")
                w, h = img.size
                ratio_w = 512 / w
                ratio_h = 512 / h
                img_tmp = img.resize((512, 512))
                boxes = detect.detect(img_tmp, model, device)
                boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

                orig_vertices = []
                theta = 0
                for box in boxes:
                    box = np.array(box[:8])
                    orig_vertices.append(box)
                    theta += find_min_rect_angle(box)

                ### print wrong detection
                if not len(boxes) == 18:
                    print('wrong detection: ', image_name, ' boxes: ', len(boxes))

                orig_vertices = np.array(orig_vertices)
                theta /= len(boxes)
                #center_x = (image.width - 1) / 2
                #center_y = (image.height - 1) / 2
                #new_vertices = np.zeros(orig_vertices.shape)
                #for i, vertice in enumerate(orig_vertices):
                #    new_vertices[i, :] = rotate_vertices(vertice, -theta, np.array([[center_x], [center_y]]))
                #vertices = new_vertices
                tmp_img, vertices = rotate_allimg(image, orig_vertices, - theta / math.pi * 180)

                dict_centers = {}
                for i, vertice in enumerate(vertices):
                    avg_x = int(averagenum(vertice[::2]))
                    avg_y = int(averagenum(vertice[1::2]))
                    dict_centers[str(avg_x) + ',' + str(avg_y)] = i

                #####
                #print(dict_centers)
                centers = sort_centers(dict_centers, 1)
                #####
                #print(centers)

                k = 0
                xcenters = []
                index = [3, 3, 1, 2, 6, 3]
                for i, j in enumerate(index):
                    xcenter = sort_xcenters(centers[k:k + j], 0)
                    if i == 4:
                        if int(xcenter[0][0].split(',')[1]) > int(xcenter[1][0].split(',')[1]):
                            tmp = xcenter[0]
                            xcenter[0] = xcenter[1]
                            xcenter[1] = tmp
                        if int(xcenter[2][0].split(',')[1]) > int(xcenter[3][0].split(',')[1]):
                            tmp = xcenter[2]
                            xcenter[2] = xcenter[3]
                            xcenter[3] = tmp
                        if int(xcenter[3][0].split(',')[1]) > int(xcenter[4][0].split(',')[1]):
                            tmp = xcenter[3]
                            xcenter[3] = xcenter[4]
                            xcenter[4] = tmp
                    k += j
                    xcenters.append(xcenter)

                ####
                # for xcenter in xcenters:
                #   print(xcenter)

                shape = []
                for i, xcenter in enumerate(xcenters):
                    if i == 0:
                        for j, center in enumerate(xcenter):
                            if j < 4:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '手写汉字'
                                shape.append(anno)
                            else:
                                break
                    if i == 1:
                        for j, center in enumerate(xcenter):
                            if j < 2:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷汉字'
                                shape.append(anno)
                            elif j == 2:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷数字'
                                shape.append(anno)
                            else:
                                break
                    if i == 2:
                        for j, center in enumerate(xcenter):
                            if j == 0:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '手写汉字'
                                shape.append(anno)
                            else:
                                break
                    if i == 3:
                        for j, center in enumerate(xcenter):
                            if j == 0:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '手写汉字'
                                shape.append(anno)
                            elif j == 1:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '手写数字'
                                shape.append(anno)
                            else:
                                break
                    if i == 4:
                        for j, center in enumerate(xcenter):
                            if j < 2:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷数字'
                                shape.append(anno)
                            elif j < 5:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷汉字'
                                shape.append(anno)
                            elif j == 5:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷数字'
                                shape.append(anno)
                            else:
                                break
                    if i == 5:
                        for j, center in enumerate(xcenter):
                            if j == 0:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '手写数字'
                                shape.append(anno)
                            elif j < 3:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印刷数字'
                                shape.append(anno)
                            else:
                                break

                for i, item in enumerate(shape):
                    all_cnt += 1
                    box = item['box']
                    box = np.array(box).reshape((4, 2))
                    p1 = Polygon(box).convex_hull
                    maxiou = 0
                    maxi = 0
                    for i, vertice in enumerate(valid_vertices):
                        vertice = np.array(vertice).reshape((4, 2))
                        p2 = Polygon(vertice).convex_hull
                        inter = p1.intersection(p2).area
                        union_area = p1.area + p2.area - inter
                        if union_area == 0:
                            iou = 0
                        else:
                            iou = float(inter) / union_area
                        if iou > maxiou:
                            maxiou = iou
                            maxi = i
                    imgclass = item['class']
                    if imgclass != typelist[maxi]:
                        wrong += 1
                        print('wrong....'+str(all_cnt))

        print('all_cnt: {}  wrong: {}   acc: {:.4f}'.format(all_cnt, wrong, (all_cnt-wrong)/all_cnt))