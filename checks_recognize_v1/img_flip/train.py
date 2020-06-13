import torch
from img_flip.dataset import DetectAngleDataset, img_enhancer
from torch.utils.data import DataLoader
from torch import optim
import configs
from model import DetectAngleModel

if __name__ == '__main__':

    print('---------------------运行环境准备-------------------------')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 50

    print('---------------------数据集准备-------------------------')
    images = []
    with open(configs.label_dir, 'r') as labels:
        for lb in labels.readlines():
            annotation = {}
            lblist = lb.split(';')
            annotation['mode'] = lblist[0]
            annotation['name'] = lblist[1]
            annotation['class'] = int(lblist[2][0])
            images.append(annotation)

    train_dataset = DetectAngleDataset(configs.img_rootdir, images)
    #test_dataset = DetectAngleDataset(config.test_img_dir, config.test_angle_label_path)
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    #test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print('---------------------网络准备-------------------------')
    model = DetectAngleModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for i, (img, img_class) in enumerate(dataloader):
            img = img.to(device)
            img_class = img_class.to(device)
            output = model(img)
            loss = criterion(output, img_class)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('epoch {} / {} ： loss: {:4f}'.format(epoch, epochs, epoch_loss))

        # print("Starting evaluate------------------------------------------")
        #model.eval()
        #right_cnt = 0
        #total_cnt = 0
        #for j, (img_name, img, angle) in enumerate(test_dataloader):
        #    img = img.to(device)
        #    angle = angle.to(device)
        #    output = model(img)
        #    result = np.argmax(output.to('cpu').detach().numpy(), axis=1)
        #    total_cnt += len(angle)
        #    for k in range(len(angle)):
        #        if angle[k] == result[k]:
        #            right_cnt += 1
        #        # print("right: ", img_name[k], " ", output[k])
        #        else:
        #            print("wrong: ", img_name[k], " ", torch.nn.functional.softmax(output[k]))
        #            print(angle[k].to('cpu'), " ", result[k])
        #print("precision: {} right_cnt: {}/{}".format(right_cnt / total_cnt, right_cnt, total_cnt))
        # print("Finish evaluate---------------------------------------------")
    torch.save(model.state_dict(), 'checkpoint.pth')