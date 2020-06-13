import math
import detect
import torch
from PIL import Image, ImageDraw
from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np

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

    model_path = '/home/chen-ubuntu/Desktop/checks_dataset/pths/model_epoch_lr4_bat_stamp_18.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    trainImgPath = '/home/chen-ubuntu/Desktop/checks_dataset/'
    create_label = '/home/chen-ubuntu/Desktop/checks_dataset/res_det/res_det_stamp.txt'
    # trainLabelPath = '/Volumes/朱振洋/rotated_mode3/Label'

    modelist = ['test_rotated_mode1', 'test_rotated_mode2', 'test_rotated_mode3']

    for mode in modelist:
        ImgPath = os.path.join(trainImgPath, mode, 'Image')
        with open(create_label, 'a') as lb:
            for root, dirs, files in os.walk(ImgPath):  # trainLabelPath):
                for file in sorted(files):
                    file_path = os.path.join(root, file)
                    image_name = file  # file[0: -4] + '.jpg'
                    #print(image_name)
                    image_path = file_path  # os.path.join(trainImgPath, image_name)

                    image = Image.open(image_path)

                    ####
                    #img_savepath = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/stamp', file[:-4] + '_origin' +  '.jpg')
                    #image.save(img_savepath)

                    #with open(file_path, 'r') as f:
                    #    lines = f.readlines()
                    #    orig_vertices = []
                    #    theta = 0
                    #    for line in lines:
                    #        line = line.split(';')
                    #        vertice = [int(vt) for vt in line[1:-1]]
                    #        vertice = np.array(vertice)
                    #        orig_vertices.append(vertice)
                    #        theta += find_min_rect_angle(vertice)
                    img = image.convert("RGB")
                    w, h = img.size
                    ratio_w = 512 / w
                    ratio_h = 512 / h
                    img_tmp = img.resize((512, 512))
                    boxes = detect.detect(img_tmp, model, device)
                    boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

                    #plot_img = detect.plot_boxes(image, boxes)
                    #plot_img.save(os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/stamp', file[:-4] + '_detected' + '.jpg'))
                    orig_vertices = []
                    theta = 0
                    if boxes is not None and boxes.size:
                        for box in boxes:
                            box = np.array(box[:8])
                            orig_vertices.append(box)
                            theta += find_min_rect_angle(box)

                        ### print wrong detection
                        if not len(boxes) == 3:
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
                        #img_savepath = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/stamp', file[:-4] + '_rotated' + '.jpg')
                        #tmp_img.save(img_savepath)

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

                        xcenters = []
                        for center in centers:
                            xcenters.append([center])

                        ####
                        #for xcenter in xcenters:
                         #   print(xcenter)

                        shape = []
                        for i, xcenter in enumerate(xcenters):
                            for center in xcenter:
                                anno = {}
                                anno['box'] = orig_vertices[int(center[1])]
                                anno['class'] = '印章汉字'
                                shape.append(anno)


                        for i, item in enumerate(shape):
                            box = item['box']
                            theta = find_min_rect_angle(box)
                            img, vertice = rotate_img(image, box, - theta / math.pi * 180)
                            x_min, x_max, y_min, y_max = get_boundary(vertice)
                            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                            img_newname = file[:-4] + '_' + '%02d' % i + '.jpg'
                            img_savepath = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/stamp', img_newname)
                            img.save(img_savepath)
                            #print(i, '  ', item['class'])

                            lb.write(img_newname + ';')
                            for site in box:
                                lb.write(str(int(site)) + ';')
                            lb.write(item['class'] + '\n')

                    else:
                        print('wrong detection: ', image_name, ' boxes: ', 0)

            lb.close()