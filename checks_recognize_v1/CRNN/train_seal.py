import config
import torch
import random
import os
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
import dataset_seal
from torch.autograd import Variable
from models import model
import torch.optim as optim
from tensorboardX import SummaryWriter
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import alphabets
import convert
from torch.nn import CTCLoss
import torch.optim as optim

writer = SummaryWriter(log_dir='logs_seal2', comment='train')


def valid(model, criterion, converter, device, dataset):
    print('start evaluate')

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    model.eval()
    with torch.no_grad():
        all_cnt = 0
        n_correct = 0
        total_loss = 0
        total_cnt = 0

        for i, (img, label, img_paths) in enumerate(dataloader):

            total_cnt += 1
            img = img.to(device)
            batch_size = img.size(0)
            all_cnt += batch_size
            text, length = converter.encode(label)
            preds = model(img)

            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            preds = preds.to('cpu')
            text, length = text.to('cpu'), length.to('cpu')
            loss = criterion(preds, text, preds_size, length)
            total_loss += loss.item()

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

            list1 = [x for x in label]
            for pred, target, img_path in zip(sim_preds, list1, img_paths):
                if pred == target:
                    n_correct += 1
                else:
                    img_name = img_path[94:]
                    print(img_name + '  pred:' + pred)
                    image = Image.open(img_path)
                    image.save('test/' + target + "_" + pred + ".jpg")

        acc = n_correct / all_cnt
        loss = total_loss / total_cnt
        print("acc: {}, loss: {}".format(acc, loss))
        print('imgs count: {},  correct: {}'.format(all_cnt, n_correct))
    return acc, loss


def train(model, criterion, converter, device, pretrain=False):
    imgdir_list = []
    label_list = []

    rootdir = '/home/chen-ubuntu/Desktop/checks_dataset/new_train_stamp_crop/'

    for mode in ['new_crop', 'new_crop2', 'new_crop3']:
        imgfile_dir = os.path.join(rootdir, mode)
        imgs_files = sorted(os.listdir(imgfile_dir))

        for img_file in imgs_files:
            imgs_dir = os.path.join(imgfile_dir, img_file)
            imgs = sorted(os.listdir(imgs_dir))
            for img in imgs:
                img_dir = os.path.join(imgs_dir, img)
                imgdir_list.append(img_dir)
                label_list.append(img[: -6])

    if len(imgdir_list) != len(label_list):
        print('dataset is wrong!')

    np.random.seed(10)
    state = np.random.get_state()
    np.random.shuffle(imgdir_list)
    np.random.set_state(state)
    np.random.shuffle(label_list)

    segment = len(imgdir_list) // 10
    train_imgdirs = imgdir_list[:segment * 9]
    train_labels = label_list[:segment * 9]
    val_imgdirs = imgdir_list[segment * 9:]
    val_labels = label_list[segment * 9:]

    print('trainset: ', len(train_imgdirs))
    print('validset: ', len(val_labels))

    trainset = dataset_seal.BaseDataset(train_imgdirs, train_labels, transform=dataset_seal.img_enhancer, _type='seal')
    validset = dataset_seal.BaseDataset(val_imgdirs, val_labels, transform=dataset_seal.img_padder, _type='seal')

    print('Device:', device)
    model = model.to(device)

    if pretrain:
        print("Using pretrained model")
        '''
        state_dict = torch.load("/home/chen-ubuntu/Desktop/checks_dataset/pths/crnn_pertrain.pth", map_location=device)

        cnn_modules = {}
        rnn_modules = {}
        for module in state_dict:
            if module.split('.')[1] == 'FeatureExtraction':
                key = module.replace("module.FeatureExtraction.", "")
                cnn_modules[key] = state_dict[module]
            elif module.split('.')[1] == 'SequenceModeling':
                key = module.replace("module.SequenceModeling.", "")
                rnn_modules[key] = state_dict[module]

        model.cnn.load_state_dict(cnn_modules)
        model.rnn.load_state_dict(rnn_modules)
        '''
    model.load_state_dict(torch.load('/home/chen-ubuntu/Desktop/checks_dataset/pths/seal_lr3_bat256_aug_epoch15_acc0.704862.pth'))

    dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)
    '''
    lr = 1e-3
    params = model.parameters()
    optimizer = optim.Adam(params, lr)
    optimizer.zero_grad()
    batch_cnt = 0
    for epoch in range(config.epochs):
        epoch_loss = 0
        model.train()
        train_acc = 0
        train_acc_cnt = 0
        
        for i, (img, label, _) in enumerate(dataloader):
            n_correct = 0
            batch_cnt += 1
            train_acc_cnt += 1
            img = img.to(device)
            text, length = converter.encode(label)
            preds = model(img)
            preds_size = torch.IntTensor([preds.size(0)] * img.size(0))
            preds = preds.to('cpu')
            loss = criterion(preds, text, preds_size, length)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

            list1 = [x for x in label]
            for pred, target in zip(sim_preds, list1):
                if pred == target:
                    n_correct += 1

            # loss.backward()
            # optimizer.step()
            # model.zero_grad()

            loss.backward()
            if (i + 1) % 4:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            train_acc += n_correct / len(list1)

            if (i + 1) % 4 == 0:
                print("epoch: {:<3d}, batch: {:<3d},  batch loss: {:4f}, epoch loss: {:4f}, acc: {}". \
                      format(epoch, i, loss.item(), epoch_loss, n_correct / len(list1)))
                writer.add_scalar('data/train_loss', loss.item(), batch_cnt)
                writer.add_scalar('data/train_acc', n_correct / len(list1), batch_cnt)
    '''
    #print('train_average_acc is: {:.3f}'.format(train_acc / train_acc_cnt))
    acc, valid_loss = valid(model, criterion, converter, device, validset)
    '''
        writer.add_scalar('data/valid_{}acc'.format('seal'), acc, batch_cnt)
        writer.add_scalar('data/valid_{}loss'.format('seal'), valid_loss, batch_cnt)

        if epoch % 3 == 0:
            torch.save(model.state_dict(),
                       '/home/chen-ubuntu/Desktop/checks_dataset/tmp_pths/seal2_lr3_bat512_expaug_epoch_{}_acc{:4f}.pth'.format(
                           epoch + 1, acc))

        if acc > 0.8:
            torch.save(model.state_dict(),
                       '/home/chen-ubuntu/Desktop/checks_dataset/tmp_pths/seal2_lr3_bat512_expaug_epoch{}_acc{:4f}.pth'.format(
                           epoch + 1, acc))
    '''

if __name__ == '__main__':

    alphabets = alphabets.alphabet_word
    n_class = len(alphabets) + 1

    torch.manual_seed(config.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    converter = convert.strLabelConverter(alphabets)
    criterion = CTCLoss()
    model = model.CRNN(class_num=n_class, backbone='resnet', pretrain=False)

    train(model, criterion, converter, device, pretrain=True)

