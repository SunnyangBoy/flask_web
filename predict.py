import os
import sys
import math
import torch
import numpy as np
from PIL import Image
from checks_recognize_v1.EAST import detect
from checks_recognize_v1.CRNN import convert
from checks_recognize_v1.CRNN import dataset
from checks_recognize_v1.CRNN import alphabets
import checks_recognize_v1.EAST.crop_mode1 as crop_mode

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


alphabetdict = {'handword': alphabets.alphabet_handword, 'handnum': alphabets.alphabet_handnum,
                    'word': alphabets.alphabet_word, 'num': alphabets.alphabet_num, 'char': alphabets.alphabet_char,
                    'seal': alphabets.alphabet_word}


def rotate_det(img, rotate_model):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    img = torch.from_numpy(img).float()
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    output = rotate_model(img)
    if output[0][0] < output[0][1]:
        print('the image is not rotated')
        return 1
    else:
        print('the image is rotated')
        return 0


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def text_recognize(img, model, converter, mode):
    img = img.convert("L")
    img = dataset.img_nomalize(img, mode)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    preds = model(img)
    preds = preds.to('cpu')
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = torch.IntTensor([preds.size(0)])
    text = converter.decode(preds.data, preds_size.data, raw=False)
    return text


def seal_detect(img, model, ratio_w, ratio_h):
    boxes = detect.detect(img, model, device)
    boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

    orig_vertices = []
    theta = 0
    if boxes is not None and boxes.size:
        for box in boxes:
            box = np.array(box[:8])
            orig_vertices.append(box)
            theta += crop_mode.find_min_rect_angle(box)

        orig_vertices = np.array(orig_vertices)
        theta /= len(boxes)
        tmp_img, vertices = crop_mode.rotate_allimg(img, orig_vertices, - theta / math.pi * 180)

        dict_centers = {}
        for i, vertice in enumerate(vertices):
            avg_x = int(crop_mode.averagenum(vertice[::2]))
            avg_y = int(crop_mode.averagenum(vertice[1::2]))
            dict_centers[str(avg_x) + ',' + str(avg_y)] = i

        centers = crop_mode.sort_centers(dict_centers, 1)

        xcenters = []
        for center in centers:
            xcenters.append([center])

        shape = []
        for i, xcenter in enumerate(xcenters):
            for center in xcenter:
                anno = {}
                anno['box'] = orig_vertices[int(center[1])]
                anno['class'] = 'seal'
                shape.append(anno)

        return shape, boxes


def recognize(shape, rotate_img, rotate_flag, handword_model, handnum_model, word_model, num_model, char_model, seal_model):
    results = []
    models = {'handword': handword_model, 'handnum': handnum_model, 'num': num_model, 'word': word_model, 'char': char_model, 'seal': seal_model}
    for i, item in enumerate(shape):
        print('*' * 30)
        mode = item['class']
        print('mode：', mode)

        box = item['box']

        theta = crop_mode.find_min_rect_angle(box)
        tmp_img, vertice = crop_mode.rotate_img(rotate_img, box, - theta / math.pi * 180)
        x_min, x_max, y_min, y_max = crop_mode.get_boundary(vertice)
        crop_img = tmp_img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if rotate_flag == 0:
            width = img.width
            height = img.height
            center_x = (width - 1) / 2
            center_y = (height - 1) / 2
            new_vertice = np.zeros(box.shape)
            new_vertice[:] = rotate_vertices(box, - math.pi, np.array([[center_x], [center_y]]))
            box = new_vertice

        str_box = []
        for site in box:
            str_box.append(str(int(site)))
        print('box：', str_box)

        alphabet = alphabetdict[mode]
        converter = convert.strLabelConverter(alphabet)
        now_model = models[mode]

        result = text_recognize(crop_img, now_model, converter, mode)
        print('name：', item['name'])
        print('result：', result)
        results.append([item['name'], result])
    return results

# ***********************************************************************************************

def predict_mode1(img, img_name, rotate_model, seal_detect_model, detect_model,
                  handword_model, handnum_model, word_model, num_model, char_model, seal_model):
    rotate_flag = rotate_det(img, rotate_model)
    if rotate_flag == 0:
        rotate_img = img.rotate(180, Image.BILINEAR)
    else:
        rotate_img = img

    rotate_img = rotate_img.convert("RGB")
    w, h = rotate_img.size
    ratio_w = 512 / w
    ratio_h = 512 / h
    img_tmp = rotate_img.resize((512, 512))

    shape, seal_boxes = seal_detect(img_tmp, seal_detect_model, ratio_w, ratio_h)

    boxes = detect.detect(img_tmp, detect_model, device)
    boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

    plot_img = detect.plot_boxes(rotate_img, boxes)
    plot_img = detect.plot_boxes(plot_img, seal_boxes)
    plot_img.save(os.path.join('/home/flask_web/uploads', img_name[:-4] + "_result" + ".jpg"))
    print('detection result saved')

    orig_vertices = []
    theta = 0
    for box in boxes:
        box = np.array(box[:8])
        orig_vertices.append(box)
        theta += crop_mode.find_min_rect_angle(box)

    orig_vertices = np.array(orig_vertices)
    theta /= len(boxes)

    tmp_img, vertices = crop_mode.rotate_allimg(rotate_img, orig_vertices, - theta / math.pi * 180)

    dict_centers = {}
    for i, vertice in enumerate(vertices):
        avg_x = int(crop_mode.averagenum(vertice[::2]))
        avg_y = int(crop_mode.averagenum(vertice[1::2]))
        dict_centers[str(avg_x) + ',' + str(avg_y)] = i

    centers = crop_mode.sort_centers(dict_centers, 1)

    k = 0
    xcenters = []
    index = [6, 2, 2, 1, 2]
    for i, j in enumerate(index):
        xcenter = crop_mode.sort_xcenters(centers[k:k + j], 0)
        if i == 0:
            if int(xcenter[0][0].split(',')[1]) < int(xcenter[1][0].split(',')[1]):
                tmp = xcenter[0]
                xcenter[0] = xcenter[1]
                xcenter[1] = tmp
            if int(xcenter[5][0].split(',')[1]) < int(xcenter[4][0].split(',')[1]):
                tmp = xcenter[5]
                xcenter[5] = xcenter[4]
                xcenter[4] = tmp
        k += j
        xcenters.append(xcenter)

    for i, xcenter in enumerate(xcenters):
        if i == 0:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '收款人'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '年'
                    shape.append(anno)
                elif j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '月'
                    shape.append(anno)
                elif j == 3:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '日'
                    shape.append(anno)
                elif j == 4:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '出票人账号'
                    shape.append(anno)
                elif j == 5:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '付款行名称'
                    shape.append(anno)
                else:
                    break
        if i == 1:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '人民币（大写）'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handnum'
                    anno['name'] = '人民币（小写）'
                    shape.append(anno)
                else:
                    break
        if i == 2:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '行号'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '劳务费'
                    shape.append(anno)
                else:
                    break
        if i == 3:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '密码'
                    shape.append(anno)
                else:
                    break
        if i == 4:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '复核'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '记账'
                    shape.append(anno)
                else:
                    break

    results = recognize(shape, rotate_img, rotate_flag, handword_model, handnum_model, word_model, num_model, char_model, seal_model)
    return results


# ***********************************************************************************************

def predict_mode2(img, img_name, rotate_model, seal_detect_model, detect_model,
                    handword_model, handnum_model, word_model, num_model, char_model, seal_model):
    rotate_flag = rotate_det(img, rotate_model)
    if rotate_flag == 0:
        rotate_img = img.rotate(180, Image.BILINEAR)
    else:
        rotate_img = img

    rotate_img = rotate_img.convert("RGB")
    w, h = rotate_img.size
    ratio_w = 512 / w
    ratio_h = 512 / h
    img_tmp = rotate_img.resize((512, 512))

    shape, seal_boxes = seal_detect(img_tmp, seal_detect_model, ratio_w, ratio_h)

    boxes = detect.detect(img_tmp, detect_model)
    boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

    plot_img = detect.plot_boxes(rotate_img, boxes)
    plot_img = detect.plot_boxes(plot_img, seal_boxes)
    plot_img.save(os.path.join('/home/flask_web/uploads', img_name[:-4] + "_result" + ".jpg"))
    print('detection result saved')

    orig_vertices = []
    theta = 0
    for box in boxes:
        box = np.array(box[:8])
        orig_vertices.append(box)
        theta += crop_mode.find_min_rect_angle(box)

    orig_vertices = np.array(orig_vertices)
    theta /= len(boxes)

    tmp_img, vertices = crop_mode.rotate_allimg(rotate_img, orig_vertices, - theta / math.pi * 180)

    dict_centers = {}
    for i, vertice in enumerate(vertices):
        avg_x = int(crop_mode.averagenum(vertice[::2]))
        avg_y = int(crop_mode.averagenum(vertice[1::2]))
        dict_centers[str(avg_x) + ',' + str(avg_y)] = i

    centers = crop_mode.sort_centers(dict_centers, 1)

    k = 0
    xcenters = []
    index = [3, 2, 2, 1, 4]
    for i, j in enumerate(index):
        xcenter = crop_mode.sort_xcenters(centers[k:k + j], 0)
        k += j
        xcenters.append(xcenter)

    for i, xcenter in enumerate(xcenters):
        if i == 0:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '年'
                    shape.append(anno)
                if j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '月'
                    shape.append(anno)
                if j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '日'
                    shape.append(anno)
                else:
                    break
        if i == 1:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '人民币（大写）'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handnum'
                    anno['name'] = '人民币（小写）'
                    shape.append(anno)
                else:
                    break
        if i == 2:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'char'
                    anno['name'] = '使用现金'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '行号'
                    shape.append(anno)
                else:
                    break
        if i == 3:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '密押'
                    shape.append(anno)
                else:
                    break
        if i == 4:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '备注'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '经办'
                    shape.append(anno)
                elif j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '复核'
                    shape.append(anno)
                elif j == 3:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '出纳'
                    shape.append(anno)
                else:
                    break

    results = recognize(shape, rotate_img, rotate_flag, handword_model, handnum_model, word_model, num_model, char_model, seal_model)
    return results

# ***********************************************************************************************

def predict_mode3(img, img_name, rotate_model, seal_detect_model, detect_model,
                    handword_model, handnum_model, word_model, num_model, char_model, seal_model):
    rotate_flag = rotate_det(img, rotate_model)
    if rotate_flag == 0:
        rotate_img = img.rotate(180, Image.BILINEAR)
    else:
        rotate_img = img

    rotate_img = rotate_img.convert("RGB")
    w, h = rotate_img.size
    ratio_w = 512 / w
    ratio_h = 512 / h
    img_tmp = rotate_img.resize((512, 512))

    shape, seal_boxes = seal_detect(img_tmp, seal_detect_model, ratio_w, ratio_h)

    boxes = detect.detect(img_tmp, detect_model)
    boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

    plot_img = detect.plot_boxes(rotate_img, boxes)
    plot_img = detect.plot_boxes(plot_img, seal_boxes)
    plot_img.save(os.path.join('/home/flask_web/uploads', img_name[:-4] + "_result" + ".jpg"))
    print('detection result saved')

    orig_vertices = []
    theta = 0
    for box in boxes:
        box = np.array(box[:8])
        orig_vertices.append(box)
        theta += crop_mode.find_min_rect_angle(box)

    orig_vertices = np.array(orig_vertices)
    theta /= len(boxes)
    tmp_img, vertices = crop_mode.rotate_allimg(rotate_img, orig_vertices, - theta / math.pi * 180)

    dict_centers = {}
    for i, vertice in enumerate(vertices):
        avg_x = int(crop_mode.averagenum(vertice[::2]))
        avg_y = int(crop_mode.averagenum(vertice[1::2]))
        dict_centers[str(avg_x) + ',' + str(avg_y)] = i

    centers = crop_mode.sort_centers(dict_centers, 1)

    k = 0
    xcenters = []
    index = [3, 3, 1, 2, 6, 3]
    for i, j in enumerate(index):
        xcenter = crop_mode.sort_xcenters(centers[k:k + j], 0)
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

    for i, xcenter in enumerate(xcenters):
        if i == 0:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '年'
                    shape.append(anno)
                if j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '月'
                    shape.append(anno)
                if j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '日'
                    shape.append(anno)
                else:
                    break
        if i == 1:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '收款人'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '代理付款行'
                    shape.append(anno)
                elif j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '代理付款行号'
                    shape.append(anno)
                else:
                    break
        if i == 2:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '出票金额'
                    shape.append(anno)
                else:
                    break
        if i == 3:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handword'
                    anno['name'] = '实际结算金额（大写）'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handnum'
                    anno['name'] = '实际结算金额（小写）'
                    shape.append(anno)
                else:
                    break
        if i == 4:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '账号'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '密押'
                    shape.append(anno)
                elif j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '申请人'
                    shape.append(anno)
                elif j == 3:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '出票行'
                    shape.append(anno)
                elif j == 4:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'word'
                    anno['name'] = '备注'
                    shape.append(anno)
                elif j == 5:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '出票行号'
                    shape.append(anno)
                else:
                    break
        if i == 5:
            for j, center in enumerate(xcenter):
                if j == 0:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'handnum'
                    anno['name'] = '多余金额'
                    shape.append(anno)
                elif j == 1:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '复核'
                    shape.append(anno)
                elif j == 2:
                    anno = {}
                    anno['box'] = orig_vertices[int(center[1])]
                    anno['class'] = 'num'
                    anno['name'] = '记账'
                    shape.append(anno)
                else:
                    break

    results = recognize(shape, rotate_img, rotate_flag, handword_model, handnum_model, word_model, num_model, char_model,
              seal_model)
    return results
