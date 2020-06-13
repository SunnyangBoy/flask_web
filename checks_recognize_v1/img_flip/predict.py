import os
import torch
import configs
from model import DetectAngleModel
import numpy as np 
from PIL import Image
import random
from matplotlib import pyplot as plt
import time

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
device = 'cuda:0'

if __name__ == '__main__':
    with torch.no_grad():
        model = DetectAngleModel()
        model.load_state_dict(torch.load('./pths/rotate.pth'))
        # model = model.cuda().half()
        model.to(device)
        model.eval()

    for mode in random.sample(('1'), 1):
        time_used = 0
        print('mode  ', mode)
        if mode == '1':
            img_mode = 'one'
        elif mode == '2':
            img_mode = 'two'
        elif mode == '3':
            img_mode = 'three'
        img_label_dir = os.path.join(configs.img_rootdir, img_mode, 'Label')
        for root, dirs, files in os.walk(img_label_dir):
            i = 0
            for file in sorted(files)[:800]:
                img_name = file[0: -4]+'.jpg'
                img_dir = os.path.join(configs.img_rootdir, img_mode, 'Image', img_name)
                img = Image.open(img_dir).convert('L')
                img = img.resize((224, 224))
                #plt.imshow(img)
                #plt.show()
                img = np.array(img)
                img = img / 255.
                img = torch.from_numpy(img).float()
                # print(img.dtype)
                img = torch.unsqueeze(img, 0)
                img = torch.unsqueeze(img, 0)
                # img = img.cuda().half()
                # print(img.dtype)
                if i == 0:
                    img_ = img
                else:
                    img_ = torch.cat((img_, img), 0)
                i += 1
                if i % 800 == 0:
                    img = img_.to(device)
                    #print(img.shape)
                    start_time = time.time()
                    output = model(img)
                    time_used += time.time()-start_time
                    i = 0
                    # f.write(str(output)+'\n')
                    # if output[0][0] < output[0][1]:
                    #     print('image_pred = 1')
                    # else:
                    #     print('image_pred = 0')
                # f.close()
        print('time used ={}'.format(time_used))
