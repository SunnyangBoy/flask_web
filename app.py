# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li
    :license: MIT, see LICENSE for more details.
"""
import os
import sys
import uuid
import click
from flask import current_app
from flask import Flask, render_template, flash, redirect, url_for, request, send_from_directory, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_dropzone import Dropzone
from wtforms import SubmitField, StringField, PasswordField, BooleanField
from wtforms.validators import DataRequired, Length, ValidationError
# from load_models import rotate_model, detect_model_mode1, detect_model_mode2, detect_model_mode3, seal_model_mode1, \
#     seal_model_mode2, seal_model_mode3, handword_model, handnum_model, word_model, num_model, char_model, seal_model

from threading import Thread

# app config
app = Flask(__name__)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret string')


# SQLite URI compatible
WIN = sys.platform.startswith('win')
if WIN:
    prefix = 'sqlite:///'
else:
    prefix = 'sqlite:////'


# db confige
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', prefix + os.path.join(app.root_path, 'data.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Flask-Dropzone config
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image'
app.config['DROPZONE_MAX_FILE_SIZE'] = 3
app.config['DROPZONE_MAX_FILES'] = 30
app.config['UPLOAD_PATH'] = os.path.join(app.root_path, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = ['png', 'jpg', 'jpeg']
dropzone = Dropzone(app)


# handlers
@app.shell_context_processor
def make_shell_context():
    return dict(db=db, User=User, Image=Image)


@app.cli.command()
@click.option('--drop', is_flag=True, help='Create after drop.')
def initdb(drop):
    """Initialize the database."""
    if drop:
        db.drop_all()
    db.create_all()
    click.echo('Initialized database.')


# Forms
class DeleteImgForm(FlaskForm):
    submit = SubmitField('Delete')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(6, 128)])
    remember = BooleanField('Remember me')
    submit = SubmitField('Log in')


class SigninForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(6, 128)])
    submit = SubmitField('Sign in')


# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True)
    password = db.Column(db.String(20))
    images = db.relationship('Image')  # collection

    def __repr__(self):
        return '<Author %r>' % self.name


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img_name = db.Column(db.String(20), index=True)
    origin_name = db.Column(db.String(20))
    result_flag = db.Column(db.String(10))
    img_mode = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    # optional
    def __repr__(self):
        return '<Note %r>' % self.img_name


def let_in(username, password):
    if User.query.filter_by(name=username, password=password).first():
        return True
    return False


@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if let_in(username, password):
            flash('Welcome home, %s!' % username)
            user = User.query.filter_by(name=username, password=password).first()
            current_app.user_id = user.id
            return redirect(url_for('index'))
        flash('Your password or name is wrong !')
    return render_template('basic.html', form=form)


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if not let_in(username, password):
            user = User(name=username, password=password)
            db.session.add(user)
            db.session.commit()
            flash('Sign in Success!')
            # 重定向和渲染模板不一样，渲染模板的URL并没有改变
            return redirect(url_for('login'))
        flash('You have already signed in，Please log in !')
    return render_template('signin.html', form=form)


@app.route('/main')
def index():
    form = DeleteImgForm()
    images = User.query.get(current_app.user_id).images
    return render_template('index.html', images=images, form=form)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def random_filename(filename):
    ext = os.path.splitext(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


@app.route('/dropzone-upload', methods=['GET', 'POST'])
def dropzone_upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'This field is required.', 400
        f = request.files.get('file')

        if f and allowed_file(f.filename):
            filename = random_filename(f.filename)
            save_path = os.path.join(app.config['UPLOAD_PATH'], filename)
            f.save(save_path)
            img = Image(img_name=filename, origin_name=f.filename, result_flag='Wait...')
            db.session.add(img)
            img.user_id = current_app.user_id
            db.session.commit()

            print('*' * 50)
            print('start predict !')
            # do_predict(app, save_path, f.filename)
            thr = Thread(target=do_predict, args=[app, save_path, f.filename])
            thr.start()
            print('*' * 50)
        else:
            return 'Invalid file type.', 400
    return render_template('dropzone.html')


@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    form = DeleteImgForm()
    if form.validate_on_submit():
        image = Image.query.get(image_id)
        if image.result_flag == 'wait...':
            flash('Please wait to delete!!!')
        else:
            db.session.delete(image)
            db.session.commit()
            os.remove(os.path.join(app.config['UPLOAD_PATH'], image.img_name))
            os.remove(os.path.join(app.config['UPLOAD_PATH'], image.img_name[:-4] + '_result' + '.jpg'))
            os.remove(os.path.join(app.config['UPLOAD_PATH'], image.img_name[:-4] + '.txt'))
            flash('Your image and result are deleted.')
    else:
        abort(400)
    return redirect(url_for('index'))


@app.route('/show/<int:image_id>')
def show_img(image_id):
    image = Image.query.get(image_id)
    results = []
    if image.result_flag == 'Done!!!':
        with open(os.path.join(app.config['UPLOAD_PATH'], image.img_name[:-4] + '.txt'), 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                line = line.split(';')
                results.append([line[0], line[1]])
        return render_template('show_img.html', filename=image.img_name, results=results)
    else:
        return render_template('show_img.html', filename='None', results=results)


@app.route('/uploads/<path:filename>')
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route("/download/<path:filename>")
def downloader(filename):
    respodse = make_response(send_from_directory(app.config['UPLOAD_PATH'], filename, as_attachment=True))
    return respodse


import math
import torch
import numpy as np
from PIL import Image as PIL_Image
from checks_recognize_v1.EAST import detect
from checks_recognize_v1.CRNN import convert
from checks_recognize_v1.CRNN import dataset
from checks_recognize_v1.CRNN import alphabets
from checks_recognize_v1.EAST.model import EAST
import checks_recognize_v1.EAST.crop_mode1 as crop_mode
# from load_models import rotate_model, handword_model, handnum_model, word_model, num_model, char_model, seal_model
from checks_recognize_v1.img_flip.model import DetectAngleModel
from checks_recognize_v1.CRNN.models import model as recog_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

rotate_model_path = '/home/flask_web/static/pths/rotated.pth'
rotate_model = DetectAngleModel()
rotate_model.load_state_dict(torch.load(rotate_model_path, map_location=torch.device('cpu')))
rotate_model.to(device)
rotate_model.eval()


alphabetdict = {'handword': alphabets.alphabet_handword, 'handnum': alphabets.alphabet_handnum,
                    'word': alphabets.alphabet_word, 'num': alphabets.alphabet_num, 'char': alphabets.alphabet_char,
                    'seal': alphabets.alphabet_word}

modeldict = {'handword': 'hand_word_epoch68_acc0.997709.pth', 'handnum': 'hand_num_epoch278_acc0.995020.pth',
                 'word': 'word_lr3_bat128_expaug_epoch34_acc0.951470.pth',
                 'num': 'print_num_lr3_bat192_expaug_epoch22_acc0.990815.pth', 'char': 'symbol_epoch88_acc1.000000.pth',
                 'seal': 'seal_lr3_bat256_expaug_epoch64_acc0.860740.pth'}

detect_model_one = '/home/flask_web/static/pths/model_1_epoch_20.pth'
detect_model_two = '/home/flask_web/static/pths/model_2_epoch_20.pth'
detect_model_three = '/home/flask_web/static/pths/model_3_epoch_20.pth'
stamp_model_one = '/home/flask_web/static/pths/model_1_seal_20.pth'
stamp_model_two = '/home/flask_web/static/pths/model_2_seal_20.pth'
stamp_model_three = '/home/flask_web/static/pths/model_3_seal_20.pth'


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
                anno['name'] = '印章第' + str(i+1) + '行'
                shape.append(anno)

        return shape, boxes


def recognize(shape, rotate_img, rotate_flag):
    results = []
    '''
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
            width = rotate_img.width
            height = rotate_img.height
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
    '''
    for mode in ['handword', 'handnum', 'num', 'word', 'char', 'seal']:
        alphabet = alphabetdict[mode]
        n_class = len(alphabet) + 1

        converter = convert.strLabelConverter(alphabet)
        now_model = recog_model.CRNN(class_num=n_class, backbone='resnet', pretrain=False)
        state_dict = torch.load(os.path.join('/home/flask_web/static/pths', modeldict[mode]),
                                map_location=torch.device('cpu'))
        now_model.load_state_dict(state_dict=state_dict)
        now_model.to(device)
        now_model.eval()

        for i, item in enumerate(shape):
            if item['class'] == mode:
                print('*' * 30)
                print('mode：', mode)

                box = item['box']

                theta = crop_mode.find_min_rect_angle(box)
                tmp_img, vertice = crop_mode.rotate_img(rotate_img, box, - theta / math.pi * 180)
                x_min, x_max, y_min, y_max = crop_mode.get_boundary(vertice)
                crop_img = tmp_img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

                if rotate_flag == 0:
                    width = rotate_img.width
                    height = rotate_img.height
                    center_x = (width - 1) / 2
                    center_y = (height - 1) / 2
                    new_vertice = np.zeros(box.shape)
                    new_vertice[:] = rotate_vertices(box, - math.pi, np.array([[center_x], [center_y]]))
                    box = new_vertice

                str_box = []
                for site in box:
                    str_box.append(str(int(site)))
                print('box：', str_box)

                result = text_recognize(crop_img, now_model, converter, mode)
                print('name', item['name'])
                print('result', result)
                results.append([item['name'], result])
    return results


def do_predict(app, img_path, origin_name):
    img_name = img_path[-36:]
    db_image = Image.query.filter_by(img_name=img_name).first()
    img_mode = int(origin_name[14])
    print('image mode is : ', img_mode)
    img = PIL_Image.open(img_path).convert('L')

    if img_mode == 1:
        db_image.img_mode = 1
        # results = predict.predict_mode1(img, img_name, rotate_model, seal_model_mode1, detect_model_mode1,
        #                       handword_model, handnum_model, word_model, num_model, char_model, seal_model)
        rotate_flag = rotate_det(img, rotate_model)
        if rotate_flag == 0:
            rotate_img = img.rotate(180, PIL_Image.BILINEAR)
        else:
            rotate_img = img

        rotate_img = rotate_img.convert("RGB")
        w, h = rotate_img.size
        ratio_w = 512 / w
        ratio_h = 512 / h
        img_tmp = rotate_img.resize((512, 512))

        seal_model_mode1 = EAST(pretrained=False).to(device)
        seal_model_mode1.load_state_dict(torch.load(stamp_model_one, map_location=torch.device('cpu')))
        seal_model_mode1.eval()

        shape, seal_boxes = seal_detect(img_tmp, seal_model_mode1, ratio_w, ratio_h)

        detect_model_mode1 = EAST(pretrained=False).to(device)
        detect_model_mode1.load_state_dict(torch.load(detect_model_one, map_location=torch.device('cpu')))
        detect_model_mode1.eval()

        boxes = detect.detect(img_tmp, detect_model_mode1, device)
        boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

        plot_img = detect.plot_boxes(rotate_img, boxes)
        plot_img = detect.plot_boxes(plot_img, seal_boxes)
        plot_img.save(os.path.join(app.config['UPLOAD_PATH'], img_name[:-4] + "_result" + ".jpg"))
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
                        anno['name'] = '出票日期（年）'
                        shape.append(anno)
                    elif j == 2:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（月）'
                        shape.append(anno)
                    elif j == 3:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（日）'
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
                        anno['name'] = '用途'
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
        results = recognize(shape, rotate_img, rotate_flag)

    elif img_mode == 2:
        db_image.img_mode = 2
        # results = predict.predict_mode2(img, img_name, rotate_model, seal_model_mode2, detect_model_mode2,
        #                       handword_model, handnum_model, word_model, num_model, char_model, seal_model)
        rotate_flag = rotate_det(img, rotate_model)
        if rotate_flag == 0:
            rotate_img = img.rotate(180, PIL_Image.BILINEAR)
        else:
            rotate_img = img

        rotate_img = rotate_img.convert("RGB")
        w, h = rotate_img.size
        ratio_w = 512 / w
        ratio_h = 512 / h
        img_tmp = rotate_img.resize((512, 512))

        seal_model_mode2 = EAST(pretrained=False).to(device)
        seal_model_mode2.load_state_dict(torch.load(stamp_model_two, map_location=torch.device('cpu')))
        seal_model_mode2.eval()

        shape, seal_boxes = seal_detect(img_tmp, seal_model_mode2, ratio_w, ratio_h)

        detect_model_mode2 = EAST(pretrained=False).to(device)
        detect_model_mode2.load_state_dict(torch.load(detect_model_two, map_location=torch.device('cpu')))
        detect_model_mode2.eval()

        boxes = detect.detect(img_tmp, detect_model_mode2, device)
        boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

        plot_img = detect.plot_boxes(rotate_img, boxes)
        plot_img = detect.plot_boxes(plot_img, seal_boxes)
        plot_img.save(os.path.join(app.config['UPLOAD_PATH'], img_name[:-4] + "_result" + ".jpg"))
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
                        anno['name'] = '出票日期（年）'
                        shape.append(anno)
                    elif j == 1:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（月）'
                        shape.append(anno)
                    elif j == 2:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（日）'
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
        results = recognize(shape, rotate_img, rotate_flag)

    elif img_mode == 3:
        db_image.img_mode = 3
        # results = predict.predict_mode3(img, img_name, rotate_model, seal_model_mode3, detect_model_mode3,
        #                      handword_model, handnum_model, word_model, num_model, char_model, seal_model)
        rotate_flag = rotate_det(img, rotate_model)
        if rotate_flag == 0:
            rotate_img = img.rotate(180, PIL_Image.BILINEAR)
        else:
            rotate_img = img

        rotate_img = rotate_img.convert("RGB")
        w, h = rotate_img.size
        ratio_w = 512 / w
        ratio_h = 512 / h
        img_tmp = rotate_img.resize((512, 512))

        seal_model_mode3 = EAST(pretrained=False).to(device)
        seal_model_mode3.load_state_dict(torch.load(stamp_model_three, map_location=torch.device('cpu')))
        seal_model_mode3.eval()

        shape, seal_boxes = seal_detect(img_tmp, seal_model_mode3, ratio_w, ratio_h)

        detect_model_mode3 = EAST(pretrained=False).to(device)
        detect_model_mode3.load_state_dict(torch.load(detect_model_three, map_location=torch.device('cpu')))
        detect_model_mode3.eval()

        boxes = detect.detect(img_tmp, detect_model_mode3, device)
        boxes = detect.adjust_ratio(boxes, ratio_w, ratio_h)

        plot_img = detect.plot_boxes(rotate_img, boxes)
        plot_img = detect.plot_boxes(plot_img, seal_boxes)
        plot_img.save(os.path.join(app.config['UPLOAD_PATH'], img_name[:-4] + "_result" + ".jpg"))
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
                        anno['name'] = '出票日期（年）'
                        shape.append(anno)
                    elif j == 1:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（月）'
                        shape.append(anno)
                    elif j == 2:
                        anno = {}
                        anno['box'] = orig_vertices[int(center[1])]
                        anno['class'] = 'handword'
                        anno['name'] = '出票日期（日）'
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
        results = recognize(shape, rotate_img, rotate_flag)

    else:
        print('the image mode is wrong !')
        return

    with open(os.path.join(app.config['UPLOAD_PATH'], img_name[:-4] + '.txt'), 'w') as writer:
        for result in results:
            line = result[0] + ";" + result[1] + '\n'
            writer.write(line)

    db_image.result_flag = 'Done!!!'
    db.session.commit()
    print("Task is done!")


if __name__ == '__main__':
    do_predict('/home/flask_web/uploads/75190e3e2c254a7c80351d9e548beed3.jpg', '012345678912341')
