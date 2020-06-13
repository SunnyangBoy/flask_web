import config
import torch
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
import dataset
from torch.autograd import Variable
from models import model
import torch.optim as optim
from tensorboardX import SummaryWriter
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
import alphabets
import convert
import math
from torch.nn import CTCLoss
import torch.optim as optim
import torch.nn.functional as F


def recog(img, model, converter, device, mode):
    model = model.to(device)
    img = dataset.img_nomalize(img, mode)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    preds = model(img)
    preds = preds.to('cpu')
    probs, preds = preds.max(2)
    text_preds = preds.transpose(1, 0).contiguous().view(-1)

    text_preds_size = torch.IntTensor([text_preds.size(0)])
    text = converter.decode(text_preds.data, text_preds_size.data, raw=False)
    return text, probs.detach().numpy()


def select_mode(mode):
    if mode == 'handword':
        return '手写汉字'

    elif mode == 'handnum':
        return '手写数字'

    elif mode == 'word':
        return '印刷汉字'

    elif mode == 'num':
        return '印刷数字'

    elif mode == 'char':
        return '字符'

    else:
        return '印章汉字'


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate(line, img_name):
    img_name = img_name[:-7] + '.jpg'
    flag = int(rotate_dict[img_name][0])
    vertice = [int(vt) for vt in line[1:-1]]
    vertice = np.array(vertice)
    if flag == 0:
        # print('origin: ', vertice)
        width = int(rotate_dict[img_name][1:].split(',')[0])
        height = int(rotate_dict[img_name][1:].split(',')[1])
        center_x = (width - 1) / 2
        center_y = (height - 1) / 2
        new_vertice = np.zeros(vertice.shape)
        new_vertice[:] = rotate_vertices(vertice, - math.pi, np.array([[center_x], [center_y]]))
        vertice = new_vertice
        # print('rotate: ', vertice)
    vertice = [str(int(vt)) for vt in vertice]
    return vertice


rotate_dict = {}

if __name__ == '__main__':

    with open('/home/chen-ubuntu/Desktop/checks_dataset/res_det/test_rotate.txt', 'r') as reader:
        rotatelines = reader.readlines()
        for rotateline in rotatelines:
            rotateline = rotateline.split(';')
            rotate_dict[rotateline[0]] = rotateline[1][:-1]

    probability_path = '/home/chen-ubuntu/Desktop/checks_dataset/res_det/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    alphabetdict = {'handword': alphabets.alphabet_handword, 'handnum': alphabets.alphabet_handnum,
                    'word': alphabets.alphabet_word, 'num': alphabets.alphabet_num, 'char': alphabets.alphabet_char,
                    'seal': alphabets.alphabet_word}

    modeldict = {'handword': 'hand_word_epoch68_acc0.997709.pth', 'handnum': 'hand_num_epoch278_acc0.995020.pth',
                 'word': 'word_lr3_bat128_expaug_epoch34_acc0.951470.pth',
                 'num': 'print_num_lr3_bat192_expaug_epoch22_acc0.990815.pth', 'char': 'symbol_epoch88_acc1.000000.pth',
                 'seal': 'seal2_lr3_bat512_expaug_epoch36_acc0.967520.pth'}

    for i, mode in enumerate(['word', 'seal']):

        print('mode :', mode)
        alphabet = alphabetdict[mode]
        n_class = len(alphabet) + 1

        converter = convert.strLabelConverter(alphabet)
        now_model = model.CRNN(class_num=n_class, backbone='resnet', pretrain=False)

        state_dict = torch.load(os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/pths/', modeldict[mode]))
        now_model.load_state_dict(state_dict=state_dict)

        now_model.to(device)
        now_model.eval()

        chn_mode = select_mode(mode)
        print('chn_mode ', chn_mode)

        testlines = []

        test_mode = []

        if i == 0:
            test_mode = ['mode1', 'mode2', 'mode3']
        elif i == 1:
            test_mode = ['stamp']

        for txtmode in test_mode:
            txt_dir = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/', 'res_det_' + txtmode + '.txt')
            with open(txt_dir, 'r') as lb:
                lines = lb.readlines()
                for line in lines[:10]:
                    line = line.split(';')
                    if line[-1][:-1] == chn_mode:
                        anno = {}
                        anno['img_name'] = line[0]
                        rotate_box = rotate(line, line[0])
                        anno['box'] = rotate_box
                        anno['mode'] = txtmode
                        testlines.append(anno)
        '''
        for x in testlines[:30]:
            print('img_name', x['img_name'])
            print('box', x['box'])
            print('mode', x['mode'])
        '''

        probability = os.path.join(probability_path, mode+'_probability.txt')
        with open(probability, 'a') as writer:
            for l, imgline in enumerate(testlines):
                img_name = imgline['img_name']
                img_mode = imgline['mode']
                img_dir = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/res_det/', img_mode, img_name)

                img = Image.open(img_dir).convert("L")
                result, preds = recog(img, now_model, converter, device, mode)

                writer.write(img_name[:-7] + '.jpg' + ';')
                for t, word in enumerate(result):
                    writer.write(word + ';')
                    if t == len(result)-1:
                        writer.write(str(round(preds[t][0], 4))[1:-1])
                    else:
                        writer.write(str(round(preds[t][0], 4))[1:-1] + ';')
                writer.write('\n')

            writer.close()










