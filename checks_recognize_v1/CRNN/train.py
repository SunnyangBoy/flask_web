import config
import torch
import random
import os
import shutil
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
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import alphabets
import convert
from torch.nn import CTCLoss
import torch.optim as optim
writer = SummaryWriter(log_dir='logs_allseal', comment='train')

def valid(model, criterion,converter, device, datasets, dataset_names):
    print('start evaluate')

    accs = {}
    losses = {}
    dataloaders = {}
    batch_dict  = {'print_word': 48, 'hand_num': 52, 'print_num': 48, 'symbol': 88, 'hand_word': 64, 'seal': 64}
    for dataset_name in dataset_names:
        dataset = datasets.get(dataset_name)

        dataloader = DataLoader(dataset, batch_size=batch_dict.get(dataset_name),
                                 shuffle=True, num_workers=0, drop_last=False)
        dataloaders[dataset_name] = dataloader

    model.eval()
    with torch.no_grad():
        for dataset_name in dataset_names:
            all_cnt = 0
            n_correct = 0
            total_loss = 0
            total_cnt = 0
            dataloader = dataloaders.get(dataset_name)

            for img, label, img_names in dataloader:

                total_cnt += 1
                img = img.to(device)
                batch_size = img.size(0)
                all_cnt += batch_size
                text, length = converter.encode(label)
                preds = model(img)

                preds_size = torch.IntTensor([preds.size(0)]*batch_size)
                preds = preds.to('cpu')
                text, length = text.to('cpu'), length.to('cpu')
                loss = criterion(preds, text, preds_size, length)
                total_loss += loss.item()


                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

                list1 = [x for x in label]
                for pred, target, img_name in zip(sim_preds, list1, img_names):
                    if pred == target:
                        n_correct += 1
                    #else:
                        #print(img_name + '  label:' + target + '  pred:' + pred)
                        #img_path = os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/merge2/Image/', img_name)
                        #image = Image.open(img_path)
                        #image.save('test/' + img_name + "_" + target + "_" + pred + ".jpg")

            acc = n_correct/all_cnt
            loss = total_loss/total_cnt
            accs[dataset_name] = acc
            losses[dataset_name] = loss
            print("dataset: {},  acc: {}, loss: {}".format(dataset_name, acc, loss))
            print('imgs count: {},  correct: {}'.format(all_cnt, n_correct))
    return accs, losses


def train(model, criterion, converter, device, train_datasets, pretrain=False):#valid_datasets=None, pretrain=False):
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

    dataset_name = 'seal'
    batch_dict  = {'print_word': 32, 'hand_num': 48, 'print_num': 48, 'symbol': 64, 'hand_word': 64, 'seal': 64}
    dataset = train_datasets.get(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_dict.get(dataset_name), shuffle=True, num_workers=4, drop_last=False)

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
            preds_size = torch.IntTensor([preds.size(0)]*img.size(0))
            preds = preds.to('cpu')
            loss = criterion(preds, text, preds_size, length)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

            list1 = [x for x in label]
            for pred, target in zip(sim_preds, list1):
                if pred == target:
                    n_correct += 1

            #loss.backward()
            #optimizer.step()
            #model.zero_grad()

            loss.backward()
            if (i + 1) % 4:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            train_acc += n_correct/len(list1)

            if (i + 1) % 4 == 0:
                print("epoch: {:<3d}, dataset:{:<8}, batch: {:<3d},  batch loss: {:4f}, epoch loss: {:4f}, acc: {}".\
                    format(epoch, dataset_name, i, loss.item(), epoch_loss, n_correct/len(list1)))
                writer.add_scalar('data/train_loss', loss.item(), batch_cnt)
                writer.add_scalar('data/train_acc', n_correct/len(list1), batch_cnt)

        print('train_average_acc is: {:.3f}'.format(train_acc/train_acc_cnt))
        writer.add_scalar('data/valid_{}acc'.format(dataset_name), train_acc/train_acc_cnt, batch_cnt)
        '''
        dataset_names = [dataset_name]
        accs, valid_losses = valid(model, criterion, converter, device, valid_datasets, dataset_names)

        acc, valid_loss = accs.get(dataset_name), valid_losses.get(dataset_name)
        writer.add_scalar('data/valid_{}acc'.format(dataset_name), acc, batch_cnt)
        writer.add_scalar('data/valid_{}loss'.format(dataset_name), valid_loss, batch_cnt)
        '''
        if epoch % 3 == 0:
            torch.save(model.state_dict(), '/home/chen-ubuntu/Desktop/checks_dataset/tmp_pths/allseal_lr3_bat512_expaug_epoch_{}_acc{:4f}.pth'.format(epoch + 1, train_acc/train_acc_cnt))

        if train_acc/train_acc_cnt > 0.8:
            torch.save(model.state_dict(), '/home/chen-ubuntu/Desktop/checks_dataset/tmp_pths/allseal_lr3_bat512_expaug_epoch{}_acc{:4f}.pth'.format(epoch + 1, train_acc/train_acc_cnt))



if __name__ == '__main__':
    torch.manual_seed(config.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_label_path = os.path.join(config.train_data_dir, 'Label.csv')
    img_infos = pd.read_csv(train_label_path, dtype='str')

    '''
    划分训练-验证集
    
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=config.random_seed)

    # 将类型编码为数字, 便于进行分层采样
    type_set = set(list(img_infos['type'].values))
    type_dict = dict(
        zip(type_set, [i for i in range(len(type_set))])
        )
    
    img_infos['type_num'] = img_infos['type'].apply(lambda x: type_dict.get(x))

    kf_id = 10
    features = ['file_name', 'label', 'type']
    X = img_infos[features]
    y = img_infos[['type_num']]
    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        if i == kf_id:
            train_img_infos = img_infos.iloc[train_index][features]
            valid_img_infos = img_infos.iloc[valid_index][features]
            break
    print('train numbers: ', len(train_index))
    print('valid numbers: ', len(valid_index))
    
    '''
    features = ['file_name', 'label', 'type']
    train_img_infos = img_infos[features]

    train_infos_hand_word    = train_img_infos[train_img_infos.type=='手写汉字']
    train_infos_hand_num     = train_img_infos[train_img_infos.type=='手写数字']
    train_infos_print_word   = train_img_infos[train_img_infos.type=='印刷汉字']
    train_infos_print_num    = train_img_infos[train_img_infos.type=='印刷数字']
    train_infos_symbol       = train_img_infos[train_img_infos.type=='字符']
    train_infos_seal         = train_img_infos[train_img_infos.type=='印章汉字']
    '''
    valid_infos_hand_word    = valid_img_infos[valid_img_infos.type=='手写汉字']
    valid_infos_hand_num     = valid_img_infos[valid_img_infos.type=='手写数字']
    valid_infos_print_word   = valid_img_infos[valid_img_infos.type=='印刷汉字']
    valid_infos_print_num    = valid_img_infos[valid_img_infos.type=='印刷数字']
    valid_infos_symbol       = valid_img_infos[valid_img_infos.type=='字符']
    valid_infos_seal         = valid_img_infos[valid_img_infos.type=='印章汉字']
    '''
    '''
    ### delete the third row of seal
    train_infos_seal = np.array(train_infos_seal)
    tmp = []
    for seal in train_infos_seal:
        if seal[1] == '财务专用章':# or seal[1][-2:] == '公司':
            continue
        tmp.append(seal)
    train_infos_seal = np.array(tmp)

    valid_infos_seal = np.array(valid_infos_seal)
    tmp = []
    for validseal in valid_infos_seal:
        if not validseal[1] == '财务专用章':# or validseal[1][-2:] == '公司':
            continue
        tmp.append(validseal)
    valid_infos_seal = np.array(tmp)
    
    train_infos_print_word = np.array(train_infos_print_word)
    tmp = []
    for word in train_infos_print_word:
        if len(word[1]) < 5:
            continue
        tmp.append(word)
    train_infos_print_word = np.array(tmp)

    valid_infos_print_word = np.array(valid_infos_print_word)
    tmp = []
    for validword in valid_infos_print_word:
        if len(validword[1]) < 5:
            continue
        tmp.append(validword)
    valid_infos_print_word = np.array(tmp)
    '''
    data_dir = config.train_data_dir

    train_dataset_hand_word  = dataset.BaseDataset(data_dir, train_infos_hand_word,  transform=dataset.img_enhancer, _type='hand_word')
    train_dataset_hand_num   = dataset.BaseDataset(data_dir, train_infos_hand_num,   transform=dataset.img_enhancer, _type='hand_num')
    train_dataset_print_word = dataset.BaseDataset(data_dir, train_infos_print_word, transform=dataset.img_enhancer, _type='print_word')
    train_dataset_print_num  = dataset.BaseDataset(data_dir, train_infos_print_num,  transform=dataset.img_enhancer, _type='print_num')
    train_dataset_symbol     = dataset.BaseDataset(data_dir, train_infos_symbol,     transform=dataset.img_enhancer, _type='symbol')
    train_dataset_seal       = dataset.BaseDataset(data_dir, train_infos_seal,       transform=dataset.img_enhancer, _type='seal')
    '''
    valid_dataset_hand_word  = dataset.BaseDataset(data_dir, valid_infos_hand_word,  transform=dataset.img_padder, _type='hand_word')
    valid_dataset_hand_num   = dataset.BaseDataset(data_dir, valid_infos_hand_num,   transform=dataset.img_padder, _type='hand_num')
    valid_dataset_print_word = dataset.BaseDataset(data_dir, valid_infos_print_word, transform=dataset.img_padder, _type='print_word')
    valid_dataset_print_num  = dataset.BaseDataset(data_dir, valid_infos_print_num,  transform=dataset.img_padder, _type='print_num')
    valid_dataset_symbol     = dataset.BaseDataset(data_dir, valid_infos_symbol,     transform=dataset.img_padder, _type='symbol')
    valid_dataset_seal       = dataset.BaseDataset(data_dir, valid_infos_seal,       transform=dataset.img_padder, _type='seal')
    '''

    train_datasets = {
            'hand_word':  train_dataset_hand_word,
            'hand_num':   train_dataset_hand_num,
            'print_word': train_dataset_print_word,
            'print_num':  train_dataset_print_num,
            'symbol':     train_dataset_symbol,
            'seal':       train_dataset_seal,
            }
    '''
    valid_datasets = {
            'hand_word':  valid_dataset_hand_word, 
            'hand_num':   valid_dataset_hand_num, 
            'print_word': valid_dataset_print_word, 
            'print_num':  valid_dataset_print_num,
            'symbol':     valid_dataset_symbol,
            'seal':       valid_dataset_seal, 
            }
    '''
    alphabets = alphabets.alphabet_word
    n_class = len(alphabets)+1

    converter = convert.strLabelConverter(alphabets)
    criterion = CTCLoss()
    model = model.CRNN(class_num=n_class, backbone='resnet', pretrain=False)
    train(model, criterion, converter, device, train_datasets, pretrain=True)#valid_datasets, pretrain=True)

