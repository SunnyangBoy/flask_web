import torch
import os
from checks_recognize_v1.CRNN import alphabets
from checks_recognize_v1.img_flip.model import DetectAngleModel
# from checks_recognize_v1.EAST.model import EAST
from checks_recognize_v1.CRNN.models import model as recog_model

model_root_path = '/home/flask_web/static/pths'
rotate_model_path = '/home/flask_web/static/pths/rotated.pth'
'''
detect_model_one = '/home/flask_web/static/pths/model_1_epoch_20.pth'
detect_model_two = '/home/flask_web/static/pths/model_2_epoch_20.pth'
detect_model_three = '/home/flask_web/static/pths/model_3_epoch_20.pth'
stamp_model_one = '/home/flask_web/static/pths/model_1_seal_20.pth'
stamp_model_two = '/home/flask_web/static/pths/model_2_seal_20.pth'
stamp_model_three = '/home/flask_web/static/pths/model_3_seal_20.pth'
'''

alphabetdict = {'handword': alphabets.alphabet_handword, 'handnum': alphabets.alphabet_handnum,
                    'word': alphabets.alphabet_word, 'num': alphabets.alphabet_num, 'char': alphabets.alphabet_char,
                    'seal': alphabets.alphabet_word}


modeldict = {'handword': 'hand_word_epoch68_acc0.997709.pth', 'handnum': 'hand_num_epoch278_acc0.995020.pth',
                 'word': 'word_lr3_bat128_expaug_epoch34_acc0.951470.pth',
                 'num': 'print_num_lr3_bat192_expaug_epoch22_acc0.990815.pth', 'char': 'symbol_epoch88_acc1.000000.pth',
                 'seal': 'seal_lr3_bat256_expaug_epoch64_acc0.860740.pth'}


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
rotate_model = DetectAngleModel()
rotate_model.load_state_dict(torch.load(rotate_model_path, map_location=torch.device('cpu')))
rotate_model.to(device)
rotate_model.eval()

'''
detect_model_mode1 = EAST(pretrained=False).to(device)
detect_model_mode1.load_state_dict(torch.load(detect_model_one, map_location=torch.device('cpu')))
detect_model_mode1.eval()


detect_model_mode2 = EAST(pretrained=False).to(device)
detect_model_mode2.load_state_dict(torch.load(detect_model_two, map_location=torch.device('cpu')))
detect_model_mode2.eval()


detect_model_mode3 = EAST(pretrained=False).to(device)
detect_model_mode3.load_state_dict(torch.load(detect_model_three, map_location=torch.device('cpu')))
detect_model_mode3.eval()


seal_model_mode1 = EAST(pretrained=False).to(device)
seal_model_mode1.load_state_dict(torch.load(stamp_model_one, map_location=torch.device('cpu')))
seal_model_mode1.eval()


seal_model_mode2 = EAST(pretrained=False).to(device)
seal_model_mode2.load_state_dict(torch.load(stamp_model_two, map_location=torch.device('cpu')))
seal_model_mode2.eval()


seal_model_mode3 = EAST(pretrained=False).to(device)
seal_model_mode3.load_state_dict(torch.load(stamp_model_three, map_location=torch.device('cpu')))
seal_model_mode3.eval()
'''

handword_model = recog_model.CRNN(class_num=len(alphabetdict['handword']) + 1, backbone='resnet', pretrain=False).to(device)
handword_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['handword']), map_location=torch.device('cpu')))
handword_model.eval()


handnum_model = recog_model.CRNN(class_num=len(alphabetdict['handnum']) + 1, backbone='resnet', pretrain=False).to(device)
handnum_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['handnum']), map_location=torch.device('cpu')))
handnum_model.eval()


word_model = recog_model.CRNN(class_num=len(alphabetdict['word']) + 1, backbone='resnet', pretrain=False).to(device)
word_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['word']), map_location=torch.device('cpu')))
word_model.eval()


num_model = recog_model.CRNN(class_num=len(alphabetdict['num']) + 1, backbone='resnet', pretrain=False).to(device)
num_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['num']), map_location=torch.device('cpu')))
num_model.eval()


char_model = recog_model.CRNN(class_num=len(alphabetdict['char']) + 1, backbone='resnet', pretrain=False).to(device)
char_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['char']), map_location=torch.device('cpu')))
char_model.eval()


seal_model = recog_model.CRNN(class_num=len(alphabetdict['seal']) + 1, backbone='resnet', pretrain=False).to(device)
seal_model.load_state_dict(torch.load(os.path.join(model_root_path, modeldict['seal']), map_location=torch.device('cpu')))
seal_model.eval()
