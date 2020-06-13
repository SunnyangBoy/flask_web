import torch
from torch.utils import data
from dataset_stamp import custom_dataset
from tensorboardX import SummaryWriter
from shapely.geometry import Polygon, MultiPoint
from torchvision.transforms import ToPILImage
from dataset import get_rotate_mat
from model import EAST
from loss import Loss
from PIL import Image, ImageDraw
import os
import numpy as np
import lanms

#mpl.use('Agg')
#import matplotlib.pyplot as plt

def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.90, nms_thresh=0.2):
    '''get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def valid(valid_loader, model, criterion, device):

    model.eval()
    corret = 0
    batch_cnt = 0
    box_cnt = 0
    valid_loss = 0

    wrong_cnt = 0

    for i, (img, gt_score, gt_geo, ignored_map, vertices) in enumerate(valid_loader):

        ###
        image = ToPILImage()(img.squeeze(0))
        draw = ImageDraw.Draw(image)
        flag = False

        batch_cnt += 1
        img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
            device)
        with torch.no_grad():
            pred_score, pred_geo = model(img)
        loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        valid_loss += loss.item()

        boxes = get_boxes(pred_score.squeeze(0).cpu().numpy(), pred_geo.squeeze(0).cpu().numpy())

        vertices = np.array(vertices).squeeze(0)

        if len(boxes) != 3:
            wrong_cnt += 1

        for box in boxes:
            box = np.array(box[:8]).reshape((4, 2))
            p1 = Polygon(box).convex_hull
            box_cnt += 1
            maxiou = 0
            for vertice in vertices:
                vertice = np.array(vertice).reshape((4, 2))
                p2 = Polygon(vertice).convex_hull
                inter = p1.intersection(p2).area
                #union_poly = np.concatenate((box, vertice))
                #union_area = MultiPoint(union_poly).convex_hull.area
                union_area = p1.area + p2.area - inter
                if union_area == 0:
                    iou = 0
                else:
                    iou = float(inter) / union_area
                if iou > maxiou:
                    maxiou = iou
                    ###
                    maxvertice = vertice
            if maxiou > 0.7:
                #print('maxiou: ', maxiou)
                corret += 1
            ###
            if maxiou <= 0.5:
                flag = True
                print('maxiou:           ', maxiou)
                box = np.array(box[:8]).reshape(-1)
                vertice = np.array(maxvertice).reshape(-1)
                draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
                draw.polygon([vertice[0], vertice[1], vertice[2], vertice[3], vertice[4], vertice[5], vertice[6], vertice[7]], outline=(255, 0, 0))

        if flag:
            image.save(os.path.join('/home/chen-ubuntu/Desktop/checks_dataset/tmp/',
                                           str(i) + '_stamp_detected' + '.jpg'))

        print('*'*10)


    print('valid acc is [{}/{}],  box wrong {}'.format(corret, box_cnt, wrong_cnt))
    return valid_loss/batch_cnt, corret/box_cnt



def train(train_root_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
    trainset = custom_dataset(train_root_path)
    file_num = trainset.__len__()
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    criterion = Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(pretrained=False)
    model.load_state_dict(torch.load('/home/chen-ubuntu/Desktop/checks_dataset/pths/model_epoch_stamp_8.pth'))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

    for epoch in range(epoch_iter):
        model.train()
        epoch_loss = 0

        loss_plot = []
        bx = []
        '''
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()
            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(
                device)
            pred_score, pred_geo = model(img)
            loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

            epoch_loss += loss.item()
            loss.backward()
            if (i + 1) % 8 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(
                    'Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
                        epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time,
                        loss.item()))

            if (i + 1) % 30 == 0:
                loss_plot.append(loss.item())
                bx.append(i + epoch * int(file_num / batch_size))
            plt.plot(bx, loss_plot, label='loss_mean', linewidth=1, color='b', marker='o',
                     markerfacecolor='green', markersize=2)
            plt.savefig(os.path.abspath('./labeled2.jpg'))
        
        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / batch_size),
                                                                  time.time() - epoch_time))
        print(time.asctime(time.localtime(time.time())))
        
        print('=' * 50)'''
        if epoch % interval == 0:
            validloss, validacc = valid(train_loader, model, criterion, device)
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            #torch.save(state_dict, os.path.join(pths_path, 'model_epoch_lr4_bat_stamp_{}.pth'.format(epoch + 9)))


if __name__ == '__main__':
    train_root_path = os.path.abspath('/home/chen-ubuntu/Desktop/checks_dataset')
    pths_path = '/home/chen-ubuntu/Desktop/checks_dataset/pths/'
    batch_size = 8
    lr = 1e-4
    num_workers = 4
    epoch_iter = 1
    save_interval = 2
    train(train_root_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)