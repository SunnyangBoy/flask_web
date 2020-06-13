import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval):
	file_num = len(os.listdir(train_img_path))
	trainset = custom_dataset(train_img_path, train_gt_path)
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)

	criterion = Loss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(pretrained=False)
	model.load_state_dict(torch.load('/home/chen-ubuntu/Desktop/checks_dataset/pths/model_epoch_mode3_14.pth'))
	data_parallel = False


	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	optimizer.zero_grad()
	#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	for epoch in range(epoch_iter):	
		model.train()
		epoch_loss = 0
		epoch_time = time.time()

		loss_plot = []
		bx = []
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time()
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			pred_score, pred_geo = model(img)
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)

			epoch_loss += loss.item()
			loss.backward()
			if (i + 1) % 3:
				optimizer.step()
				optimizer.zero_grad()

			if (i + 1) % 100 == 0:
				print(
					'Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(
						epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time,
						loss.item()))

			if (i + 1) % 100 == 0:
				loss_plot.append(loss.item())
				bx.append(i + epoch * int(file_num / batch_size))
			plt.plot(bx, loss_plot, label='loss_mean', linewidth=1, color='b', marker='o',
					 markerfacecolor='green', markersize=2)
			plt.savefig(os.path.abspath('./labeled.jpg'))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		if (epoch + 1) % interval == 0:
			state_dict = model.module.state_dict() if data_parallel else model.state_dict()
			torch.save(state_dict, os.path.join(pths_path, 'model3_epoch_{}.pth'.format(epoch+1+14)))


if __name__ == '__main__':
	train_img_path = os.path.abspath('/home/chen-ubuntu/Desktop/checks_dataset/rotated_mode3/Image')
	train_gt_path  = os.path.abspath('/home/chen-ubuntu/Desktop/checks_dataset/rotated_mode3/Label')
	pths_path      = '/home/chen-ubuntu/Desktop/checks_dataset/pths/'
	batch_size     = 8
	lr             = 1e-4
	num_workers    = 4
	epoch_iter     = 600
	save_interval  = 2
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval)