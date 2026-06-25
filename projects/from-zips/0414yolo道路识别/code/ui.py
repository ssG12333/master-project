from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
from pathlib import Path
import numpy as np
import time


import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import sqlite3
import random
import string

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

# 初始化数据库
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

# 生成随机验证码
def generate_captcha(length=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.captcha_text = generate_captcha()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("登录")
        self.setFixedSize(400, 300)
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("车道线检测系统登录")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 登录表单
        form_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)
        
        # 验证码
        captcha_layout = QHBoxLayout()
        self.captcha_label = QLabel(self.captcha_text)
        self.captcha_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.captcha_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        
        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_captcha)
        
        captcha_layout.addWidget(QLabel("验证码:"))
        captcha_layout.addWidget(self.captcha_label)
        captcha_layout.addWidget(refresh_btn)
        layout.addLayout(form_layout)
        layout.addLayout(captcha_layout)
        layout.addWidget(self.captcha_input)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        login_btn = QPushButton("登录")
        login_btn.clicked.connect(self.login)
        
        register_btn = QPushButton("注册")
        register_btn.clicked.connect(self.go_to_register)
        
        btn_layout.addWidget(login_btn)
        btn_layout.addWidget(register_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def refresh_captcha(self):
        self.captcha_text = generate_captcha()
        self.captcha_label.setText(self.captcha_text)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        captcha = self.captcha_input.text()
        
        # 验证码校验
        if captcha.upper() != self.captcha_text.upper():
            QMessageBox.warning(self, "错误", "验证码错误，请重新输入")
            return
        
        # 数据库验证
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            QMessageBox.information(self, "成功", "登录成功")
            
            # 创建全局主窗口（这样可以防止被垃圾回收）
            global MainWindow
            MainWindow = QtWidgets.QMainWindow()
            global ui
            ui = Ui_MainWindow()
            ui.setupUi(MainWindow)
            
            # 添加窗口关闭事件处理，确保摄像头资源被释放
            def closeEvent(event):
                try:
                    ui.printf("正在关闭程序，清理资源...")
                    
                    # 停止所有线程 - 使用安全停止方式
                    if hasattr(ui, 'thread_1') and ui.thread_1.isRunning():
                        try:
                            ui.printf("正在停止检测线程...")
                            if hasattr(ui.thread_1, 'stop'):
                                ui.thread_1.stop()
                            # 等待一小段时间但不长时间阻塞
                            ui.thread_1.wait(300)
                        except Exception as e:
                            print(f"停止thread_1时出错: {str(e)}")
                    
                    if hasattr(ui, 'camera_thread') and ui.camera_thread.isRunning():
                        try:
                            ui.printf("正在停止摄像头线程...")
                            ui.camera_thread.stop()
                            # 等待一小段时间但不长时间阻塞
                            ui.camera_thread.wait(300)
                        except Exception as e:
                            print(f"停止camera_thread时出错: {str(e)}")
                            
                    # 释放所有OpenCV窗口和资源
                    ui.printf("正在释放OpenCV资源...")
                    cv2.destroyAllWindows()
                    
                    # 清理CUDA内存
                    ui.printf("正在清理GPU内存...")
                    torch.cuda.empty_cache()
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    ui.printf("资源释放完成，程序即将退出")
                    
                except Exception as e:
                    print(f"关闭时出错: {str(e)}")
                
                # 接受关闭事件
                event.accept()
            
            # 设置关闭事件处理函数
            MainWindow.closeEvent = closeEvent
            
            MainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")
    
    def go_to_register(self):
        self.register_window = RegisterWindow(self)
        self.register_window.show()
        self.hide()


class RegisterWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.captcha_text = generate_captcha()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("注册")
        self.setFixedSize(400, 350)
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("注册新账号")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 注册表单
        form_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")
        form_layout.addRow("用户名:", self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("密码:", self.password_input)
        
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("确认密码:", self.confirm_input)
        
        # 验证码
        captcha_layout = QHBoxLayout()
        self.captcha_label = QLabel(self.captcha_text)
        self.captcha_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.captcha_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        
        self.captcha_input = QLineEdit()
        self.captcha_input.setPlaceholderText("请输入验证码")
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_captcha)
        
        captcha_layout.addWidget(QLabel("验证码:"))
        captcha_layout.addWidget(self.captcha_label)
        captcha_layout.addWidget(refresh_btn)
        layout.addLayout(form_layout)
        layout.addLayout(captcha_layout)
        layout.addWidget(self.captcha_input)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        back_btn = QPushButton("返回登录")
        back_btn.clicked.connect(self.go_to_login)
        
        register_btn = QPushButton("注册")
        register_btn.clicked.connect(self.register)
        
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(register_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def refresh_captcha(self):
        self.captcha_text = generate_captcha()
        self.captcha_label.setText(self.captcha_text)
    
    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        captcha = self.captcha_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "错误", "用户名和密码不能为空")
            return
        
        if password != confirm:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致")
            return
        
        if captcha.upper() != self.captcha_text.upper():
            QMessageBox.warning(self, "错误", "验证码错误，请重新输入")
            return
        
        # 保存到数据库
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            QMessageBox.information(self, "成功", "注册成功，请登录")
            self.go_to_login()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "错误", "用户名已存在")
        finally:
            conn.close()
    
    def go_to_login(self):
        self.parent.show()
        self.close()

class CycleGANModel():
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,
                 generator,
                 discriminator=None,
                 cycle_criterion=None,
                 idt_criterion=None,
                 gan_criterion=None,
                 pool_size=50,
                 direction='a2b',
                 lambda_a=10.,
                 lambda_b=10.):
        """Initialize the CycleGAN class.

        Args:
            generator (dict): config of generator.
            discriminator (dict): config of discriminator.
            cycle_criterion (dict): config of cycle criterion.
        """
        super(CycleGANModel, self).__init__()

        self.direction = direction

        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        # define generators
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.nets['netG_A'] = build_generator(generator)
        self.nets['netG_B'] = build_generator(generator)
        init_weights(self.nets['netG_A'])
        init_weights(self.nets['netG_B'])

        # define discriminators
        if discriminator:
            self.nets['netD_A'] = build_discriminator(discriminator)
            self.nets['netD_B'] = build_discriminator(discriminator)
            init_weights(self.nets['netD_A'])
            init_weights(self.nets['netD_B'])

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool(pool_size)
        # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(pool_size)

        # define loss functions
        if gan_criterion:
            self.gan_criterion = build_criterion(gan_criterion)

        if cycle_criterion:
            self.cycle_criterion = build_criterion(cycle_criterion)

        if idt_criterion:
            self.idt_criterion = build_criterion(idt_criterion)

    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """

        AtoB = self.direction == 'a2b'

        if AtoB:
            if 'A' in input:
                self.real_A = paddle.to_tensor(input['A'])
            if 'B' in input:
                self.real_B = paddle.to_tensor(input['B'])
        else:
            if 'B' in input:
                self.real_A = paddle.to_tensor(input['B'])
            if 'A' in input:
                self.real_B = paddle.to_tensor(input['A'])

        if 'A_paths' in input:
            self.image_paths = input['A_paths']
        elif 'B_paths' in input:
            self.image_paths = input['B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'real_A'):
            self.fake_B = self.nets['netG_A'](self.real_A)  # G_A(A)
            self.rec_A = self.nets['netG_B'](self.fake_B)  # G_B(G_A(A))

            # visual
            self.visual_items['real_A'] = self.real_A
            self.visual_items['fake_B'] = self.fake_B
            self.visual_items['rec_A'] = self.rec_A

        if hasattr(self, 'real_B'):
            self.fake_A = self.nets['netG_B'](self.real_B)  # G_B(B)
            self.rec_B = self.nets['netG_A'](self.fake_A)  # G_A(G_B(B))

            # visual
            self.visual_items['real_B'] = self.real_B
            self.visual_items['fake_A'] = self.fake_A
            self.visual_items['rec_B'] = self.rec_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Args:
            netD (Layer): the discriminator D
            real (paddle.Tensor): real images
            fake (paddle.Tensor): images generated by a generator

        Return:
            the discriminator loss.

        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.gan_criterion(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.gan_criterion(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.nets['netD_A'], self.real_B,
                                              fake_B)
        self.losses['D_A_loss'] = self.loss_D_A

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.nets['netD_B'], self.real_A,
                                              fake_A)
        self.losses['D_B_loss'] = self.loss_D_B

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if self.idt_criterion:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.nets['netG_A'](self.real_B)

            self.loss_idt_A = self.idt_criterion(self.idt_A,
                                                 self.real_B) * self.lambda_b
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.nets['netG_B'](self.real_A)

            # visual
            self.visual_items['idt_A'] = self.idt_A
            self.visual_items['idt_B'] = self.idt_B

            self.loss_idt_B = self.idt_criterion(self.idt_B,
                                                 self.real_A) * self.lambda_a
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.gan_criterion(self.nets['netD_A'](self.fake_B),
                                           True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.gan_criterion(self.nets['netD_B'](self.fake_A),
                                           True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.cycle_criterion(self.rec_A,
                                                 self.real_A) * self.lambda_a
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.cycle_criterion(self.rec_B,
                                                 self.real_B) * self.lambda_b

        self.losses['G_idt_A_loss'] = self.loss_idt_A
        self.losses['G_idt_B_loss'] = self.loss_idt_B
        self.losses['G_A_adv_loss'] = self.loss_G_A
        self.losses['G_B_adv_loss'] = self.loss_G_B
        self.losses['G_A_cycle_loss'] = self.loss_cycle_A
        self.losses['G_B_cycle_loss'] = self.loss_cycle_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    def train_iter(self, optimizers=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # compute fake images and reconstruction images.
        self.forward()
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.nets['netD_A'], self.nets['netD_B']],
                               False)
        # set G_A and G_B's gradients to zero
        optimizers['optimG'].clear_grad()
        # calculate gradients for G_A and G_B
        self.backward_G()
        # update G_A and G_B's weights
        self.optimizers['optimG'].step()
        # D_A and D_B
        self.set_requires_grad([self.nets['netD_A'], self.nets['netD_B']], True)

        # set D_A and D_B's gradients to zero
        optimizers['optimD'].clear_grad()
        # calculate gradients for D_A
        self.backward_D_A()
        # calculate graidents for D_B
        self.backward_D_B()
        # update D_A and D_B's weights
        optimizers['optimD'].step()


    def test_iter(self, metrics=None):
        self.nets['netG_A'].eval()
        self.forward()
        with paddle.no_grad():
            if metrics is not None:
                for metric in metrics.values():
                    metric.update(self.fake_B, self.real_B)
        self.nets['netG_A'].train()

def load_model(
        weights='./best.pt',  # model.pt path(s)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        verbose=True  # 是否输出详细信息
):
    # Load model
    try:
        if verbose:
            print(f"使用设备: {device}")
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        if verbose:
            print(f"模型已加载到设备: {device}")
        return model, stride, names, pt
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        # 尝试使用CPU
        print("尝试使用CPU加载...")
        device = select_device('cpu')
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        return model, stride, names, pt


def run(model, img, stride, pt,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu - 修改默认值为空字符串
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        retina_masks=True,
        verbose=False,  # 是否输出详细信息
        ):
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # 静默模式warmup，避免每次都输出CUDA信息
    if not hasattr(model, 'warmup_done'):
        model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
        model.warmup_done = True  # 标记已经完成warmup

    cal_detect = []
    # 初始化masks_copy为空数组，防止在没有检测到物体时引用未定义变量
    masks_copy = np.array([])
    
    # 确保使用模型的设备，而不是尝试重新选择设备
    device = model.device
    
    if verbose:
        device = select_device(device)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader
    im = letterbox(img, imgsz, stride, pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)  # 使用模型的device
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = model(im, augment=augment)[:2]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        annotator = Annotator(img, line_width=1, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()  # rescale boxes to im0 size
            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], img.shape[:2])  # HWC
            masks_copy = masks.cpu().numpy()
            segments = [
                scale_segments(img.shape if retina_masks else im.shape[2:], x, img.shape, normalize=True)
                for x in reversed(masks2segments(masks))]

            # Write results
            ###############################################
            id_list = []
            for id in range(len(det[:, :6])):
                # print(det[id])
                # print(masks[id])
                class_name = names[int(det[:, :6][id][5])]
                #print(det[:, :6][id])
                #print(class_name)
                # if class_name == 'person':
                # id_list.append(id)

            # print(id_list)

            def del_tensor(arr, id_list):
                if len(id_list) == 0:
                    return arr
                elif len(id_list) == 1:
                    arr1 = arr[:id_list[0]]
                    arr2 = arr[id_list[0] + 1:]
                    return torch.cat((arr1, arr2), dim=0)
                else:
                    arr1 = arr[:id_list[0]]
                    arr2 = arr[id_list[0] + 1:id_list[1]]
                    arr1 = torch.cat((arr1, arr2), dim=0)
                    for id_index in range(len(id_list)):
                        arr2 = arr[id_list[id_index - 1] + 1:id_list[id_index]]
                        arr1 = torch.cat((arr1, arr2), dim=0)
                    return arr1

            det = del_tensor(det, id_list)
            masks = del_tensor(masks, id_list)
            ###############################################
            color_list = []
            for i in range(len(det[:, 5])):
                color_list.append([0,0,255])
            annotator.masks(
                masks,
                colors=color_list,
                im_gpu=torch.as_tensor(img, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                    0).contiguous() /
                       255 if retina_masks else im[i],
                )

            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                c = int(cls)  # integer class
                label = f'{names[c]}'
                #lbl = names[int(cls)]
                contours = segments[j]
                #print(segments[j])
                #if lbl not in [' Chef clothes',' clothes']:
                    #continue
                cal_detect.append([label, xyxy,float(conf),contours])
    return  cal_detect,masks_copy


def det_yolov5v7(info1):
    global model, stride, names, pt
    if info1[-3:] in ['jpg','png','jpeg','tif','bmp','JPG']:
        try:
            image = cv2.imread(info1)  # 读取识别对象
            if image is None:
                ui.printf(f"无法读取图像: {info1}")
                return
                
            results, masks = run(model, image, stride, pt, verbose=False)  # 添加verbose=False参数
            
            # 只有当检测到结果时才处理
            if len(results) > 0:
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [0,0,255]
                    area = cv2.contourArea(i[3])
                    ui.printf(f"{time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))}检测到{i[0]}, 置信度: {i[2]:.2f}")
                    cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)

            # 图像增强
            image = np.power(image, 1.5)  # 对像素值指数变换
            
            # 确保结果目录存在
            os.makedirs('./result', exist_ok=True)
            
            # 保存和显示结果
            result_path = './result/' + os.path.basename(info1)
            cv2.imwrite(result_path, image)
            show = cv2.imread(result_path)
            ui.showimg(show)
            ui.printf(f"处理完成: {info1}")
            
        except Exception as e:
            ui.printf(f"处理图像时出错: {str(e)}")
    elif info1[-3:] in ['mp4','avi','MP4','AVI']:
        try:
            # 确保结果目录存在
            os.makedirs('./result', exist_ok=True)
            
            capture = cv2.VideoCapture(info1)
            if not capture.isOpened():
                ui.printf(f"无法打开视频: {info1}")
                return
                
            score = 0.0  # 初始化score变量，避免未定义错误
            ui.printf(f"开始处理视频: {info1}")
            
            while True:
                ret, image = capture.read()
                if not ret:
                    break
                    
                try:
                    results, masks = run(model, image, stride, pt, verbose=False)  # 添加verbose=False参数
                    # 只有检测到车道线时才处理
                    if len(results) > 0:
                        for i in results:
                            box = i[1]
                            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                            color = [0, 0, 255]
                            ui.printf(str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + '检测到' + str(i[0]))
                            cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                            if i[2] > score:
                                score = float(i[2])  # 更新score变量
                            cv2.putText(image, str(i[0]) + ' ' + str(score), (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                except Exception as e:
                    ui.printf(f"处理视频帧时出错: {str(e)}")
                
                try:    
                    result_path = './result/' + os.path.basename(info1) + '.jpg'
                    cv2.imwrite(result_path, image)
                    ui.showimg(image)
                except Exception as e:
                    ui.printf(f"保存或显示图像时出错: {str(e)}")
                    
                QApplication.processEvents()
                
            capture.release()
            ui.printf(f"视频处理完成: {info1}")
            
        except Exception as e:
            ui.printf(f"处理视频时出错: {str(e)}")
    else:
        ui.printf(f"不支持的文件格式: {info1}")

class Thread_1(QThread):  # 线程1
    def __init__(self, info1):
        super().__init__()
        self.info1 = info1
        self.is_running = True
        
    def run(self):
        try:
            ui.printf(f"开始处理: {self.info1}")
            
            # 检查文件是否存在
            if not os.path.exists(self.info1):
                ui.printf(f"错误：文件不存在: {self.info1}")
                return
                
            # 判断文件类型
            if self.info1.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.bmp')):
                self.process_image()
            elif self.info1.lower().endswith(('.mp4', '.avi')):
                self.process_video()
            else:
                ui.printf(f"不支持的文件格式: {self.info1}")
                
            ui.printf("处理完成")
        except Exception as e:
            ui.printf(f"处理过程中出错: {str(e)}")
            import traceback
            ui.printf(traceback.format_exc())
            
    def process_image(self):
        """处理图像文件"""
        try:
            global model, stride, names, pt
            image = cv2.imread(self.info1)
            if image is None:
                ui.printf(f"无法读取图像: {self.info1}")
                return
                
            # 检查线程是否应该停止
            if not self.is_running:
                ui.printf("检测到停止信号，取消图像处理")
                return
                
            results, masks = run(model, image, stride, pt, verbose=False)
            
            # 只有当检测到结果时才处理
            if len(results) > 0:
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [0,0,255]
                    area = cv2.contourArea(i[3])
                    ui.printf(f"{time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))}检测到{i[0]}, 置信度: {i[2]:.2f}")
                    cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)

            # 图像增强
            image = np.power(image, 1.5)  # 对像素值指数变换
            
            # 检查线程是否应该停止
            if not self.is_running:
                ui.printf("检测到停止信号，取消图像处理")
                return
                
            # 确保结果目录存在
            os.makedirs('./result', exist_ok=True)
            
            # 保存和显示结果
            result_path = './result/' + os.path.basename(self.info1)
            cv2.imwrite(result_path, image)
            show = cv2.imread(result_path)
            ui.showimg(show)
            ui.printf(f"处理完成: {self.info1}")
            
        except Exception as e:
            ui.printf(f"处理图像时出错: {str(e)}")
            
    def process_video(self):
        """处理视频文件"""
        try:
            global model, stride, names, pt
            # 确保结果目录存在
            os.makedirs('./result', exist_ok=True)
            
            capture = cv2.VideoCapture(self.info1)
            if not capture.isOpened():
                ui.printf(f"无法打开视频: {self.info1}")
                return
                
            score = 0.0  # 初始化score变量，避免未定义错误
            ui.printf(f"开始处理视频: {self.info1}")
            
            frame_count = 0
            while self.is_running:  # 使用is_running检查是否应该停止
                ret, image = capture.read()
                if not ret:
                    break
                    
                # 每3帧处理一次，减少计算负担
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                    
                try:
                    results, masks = run(model, image, stride, pt, verbose=False)
                    # 只有检测到车道线时才处理
                    if len(results) > 0:
                        for i in results:
                            box = i[1]
                            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                            color = [0, 0, 255]
                            ui.printf(str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + '检测到' + str(i[0]))
                            cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                            if i[2] > score:
                                score = float(i[2])  # 更新score变量
                            cv2.putText(image, str(i[0]) + ' ' + str(score), (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                except Exception as e:
                    ui.printf(f"处理视频帧时出错: {str(e)}")
                
                # 检查是否应该停止
                if not self.is_running:
                    ui.printf("检测到停止信号，取消视频处理")
                    break
                    
                try:    
                    result_path = './result/' + os.path.basename(self.info1) + '.jpg'
                    cv2.imwrite(result_path, image)
                    ui.showimg(image)
                except Exception as e:
                    ui.printf(f"保存或显示图像时出错: {str(e)}")
                    
                QApplication.processEvents()
                
            capture.release()
            ui.printf(f"视频处理完成: {self.info1}")
            
        except Exception as e:
            ui.printf(f"处理视频时出错: {str(e)}")
            
    def stop(self):
        """安全停止线程"""
        self.is_running = False
        
    def run2(self, info1):
        # 为了兼容性保留此方法，但不再使用
        pass

class Thread_Camera(QThread):  # 摄像头检测线程
    def __init__(self):
        super().__init__()
        self.is_running = True
        self.frame_count = 0  # 添加帧计数器
        # 修改设备参数，使用select_device函数先获取有效设备
        self.device = select_device('') if torch.cuda.is_available() else select_device('cpu')
        self.mutex = QMutex()  # 添加互斥锁，防止UI更新冲突
        
    def run(self):
        global model, stride, names, pt
        capture = None
        
        # 尝试打开摄像头
        try:
            # 确保上一次的显示已经清理
            try:
                self.mutex.lock()
                if ui and hasattr(ui, "label_2"):
                    blank_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(blank_img, "正在初始化摄像头...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    ui.showimg(blank_img)
                    QApplication.processEvents()
            finally:
                self.mutex.unlock()
            
            ui.printf(f"使用设备: {self.device}")
            ui.printf("初始化摄像头...")
            
            # 尝试不同的摄像头索引
            for cam_idx in [0, 1]:  # 尝试索引0和1
                if not self.is_running:
                    ui.printf("检测到停止信号，取消摄像头初始化")
                    return
                    
                try:
                    capture = cv2.VideoCapture(cam_idx)
                    if capture and capture.isOpened():
                        ui.printf(f"成功打开摄像头 {cam_idx}")
                        break
                except Exception as e:
                    ui.printf(f"尝试打开摄像头 {cam_idx} 失败: {str(e)}")
                    if capture:
                        capture.release()
                        capture = None
            
            if capture is None or not capture.isOpened():
                ui.printf("无法打开摄像头，请检查摄像头连接或驱动")
                return
                
            # 设置较低的分辨率以减轻处理负担
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            ui.printf("摄像头已打开，开始检测")
            ui.printf("按ESC键可退出摄像头检测")
            
            # 预热模型以避免第一次推理时的延迟
            if not self.is_running:
                ui.printf("检测到停止信号，取消摄像头检测")
                return
                
            ui.printf("预热模型...")
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            try:
                _ = run(model, dummy_img, stride, pt, verbose=False, device=self.device)
                ui.printf("模型预热完成")
            except Exception as e:
                ui.printf(f"模型预热失败: {str(e)}")

            # 在开始循环前再次检查线程状态
            if not self.is_running:
                ui.printf("检测到停止信号，取消摄像头检测")
                return
                
            # 开始检测前清空UI事件队列
            QApplication.processEvents()
            
            # 帧处理循环
            while self.is_running:
                try:
                    if capture is None or not capture.isOpened():
                        ui.printf("摄像头连接已断开")
                        break
                        
                    ret, frame = capture.read()
                    if not ret:
                        ui.printf("无法读取摄像头画面")
                        # 尝试重新初始化摄像头
                        self.frame_count += 1
                        if self.frame_count % 10 == 0:  # 每10次失败尝试重新初始化一次
                            try:
                                if capture:
                                    capture.release()
                                    capture = None
                                capture = cv2.VideoCapture(0)
                                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                ui.printf("尝试重新初始化摄像头...")
                            except Exception as e:
                                ui.printf(f"重新初始化摄像头失败: {str(e)}")
                        continue
                    
                    # 计数器自增
                    self.frame_count += 1
                    
                    # 每3帧处理一次，减轻计算负担
                    if self.frame_count % 3 != 0:
                        # 只显示不处理
                        try:
                            self.mutex.lock()
                            ui.showimg(frame)
                        finally:
                            self.mutex.unlock()
                            
                        # 处理UI事件，保持界面响应
                        QApplication.processEvents()
                        continue
                    
                    # 确保图像正确
                    if frame is None or frame.size == 0:
                        continue
                        
                    # 创建图像副本以避免引用问题
                    process_frame = frame.copy()
                    
                    try:
                        # 检测车道线，使用verbose=False避免重复输出CUDA信息
                        detections, masks = run(model, process_frame, stride, pt, verbose=False, device=self.device)
                        
                        # 只有当检测到物体时才绘制和记录
                        if len(detections) > 0:
                            # 绘制检测结果
                            for detection in detections:
                                box = detection[1]
                                label = detection[0]
                                conf = detection[2]
                                
                                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                                cv2.rectangle(process_frame, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                                cv2.putText(process_frame, f"{label} {conf:.2f}", (p1[0], p1[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                
                                # 记录检测结果但不要过于频繁
                                if self.frame_count % 15 == 0:  # 每15帧输出一次日志
                                    ui.printf(f"{time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))}检测到{label}, 置信度: {conf:.2f}")
                    except Exception as e:
                        ui.printf(f"检测时出错: {str(e)}")
                        # 在出错时仍然显示原始帧
                        process_frame = frame.copy()
                    
                    # 显示处理后的画面
                    try:
                        self.mutex.lock()
                        ui.showimg(process_frame)
                    finally:
                        self.mutex.unlock()
                    
                    # 确保UI更新完成
                    QApplication.processEvents()
                    
                    # 检查线程状态
                    if not self.is_running:
                        ui.printf("检测到停止信号，停止摄像头检测")
                        break
                    
                    # 释放不再需要的资源
                    del process_frame
                    
                    # 每50帧清理一次GPU内存
                    if self.frame_count % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # 控制帧率，避免过度消耗CPU
                    QThread.msleep(20)
                    
                except Exception as e:
                    ui.printf(f"处理帧时出错: {str(e)}")
                    # 出错时打印更多调试信息
                    import traceback
                    ui.printf(traceback.format_exc().split("\n")[-2])
                    
                    # 短暂暂停，避免错误消息过多
                    QThread.msleep(100)
                
        except Exception as e:
            ui.printf(f"摄像头线程异常: {str(e)}")
            # 打印详细错误信息
            import traceback
            ui.printf(traceback.format_exc())
        finally:
            # 确保资源被正确释放
            try:
                ui.printf("正在释放摄像头资源...")
                if capture is not None:
                    try:
                        capture.release()
                    except:
                        pass
                cv2.destroyAllWindows()
                ui.printf("摄像头检测已停止")
                
                # 清理内存
                torch.cuda.empty_cache()
                
                # 重置UI
                try:
                    self.mutex.lock()
                    if ui and hasattr(ui, "label_2"):
                        # 显示一个简单的"已停止"提示
                        blank_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
                        cv2.putText(blank_img, "摄像头检测已停止", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        ui.showimg(blank_img)
                        QApplication.processEvents()
                finally:
                    # 确保解锁互斥锁
                    self.mutex.unlock()
                    
                # 强制垃圾回收
                import gc
                gc.collect()
            except Exception as e:
                ui.printf(f"清理资源时出错: {str(e)}")
    
    def stop(self):
        self.is_running = False


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        MainWindow.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e3c72, stop:1 #2a5298);
            }
            QPushButton {
                background: rgba(255,142,0,0.9);
                border-radius: 10px;
                padding: 8px 16px;
                color: white;
                font: bold 16px '微软雅黑';
                min-width: 120px;
            }
            QPushButton:hover {
                background: rgba(255,142,0,1);
            }
            QTextBrowser {
                background: rgba(255,255,255,0.8);
                border-radius: 5px;
                font: 14px 'Consolas';
                padding: 10px;
            }
        """)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(168, 60, 600, 60))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setStyleSheet("""
            font-size:32px;
            font-weight:bold;
            font-family:SimHei;
            color: white;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
        """)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 150, 900, 760))
        self.label_2.setStyleSheet("background:rgba(255,255,255,0.3);")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(450, 920, 870, 39))
        self.label3.setAutoFillBackground(False)
        self.label3.setStyleSheet("")
        self.label3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label3.setAlignment(QtCore.Qt.AlignCenter)
        self.label3.setObjectName("label")
        self.label3.setStyleSheet(
            "font-size:16px;font-weight:bold;font-family:SimHei;color:white;background:rgba(255,255,255,0);")

        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(950, 150, 300, 580))
        self.textBrowser.setStyleSheet("background:rgba(255,255,255,0.5);")
        self.textBrowser.setObjectName("textBrowser")
        
        # 选择对象按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1020, 750, 150, 40))
        self.pushButton.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton.setObjectName("pushButton")
        
        # 开始识别按钮
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1020, 800, 150, 40))
        self.pushButton_2.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_2.setObjectName("pushButton_2")
        
        # 摄像头检测按钮
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(1020, 850, 150, 40))
        self.pushButton_3.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_3.setObjectName("pushButton_3")
        
        # 退出系统按钮
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1020, 900, 150, 40))
        self.pushButton_4.setStyleSheet("background:rgba(255,142,0,1);border-radius:10px;padding:2px 4px;")
        self.pushButton_4.setObjectName("pushButton_4")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于深度学习的车道线检测系统"))
        self.label.setText(_translate("MainWindow", "基于深度学习的车道线检测系统"))
        self.label_2.setText(_translate("MainWindow", "请添加对象"))
        self.label3.setText(_translate("MainWindow", ""))
        self.pushButton.setText(_translate("MainWindow", "选择对象"))
        self.pushButton_2.setText(_translate("MainWindow", "开始识别"))
        self.pushButton_3.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_4.setText(_translate("MainWindow", "退出系统"))

        # 点击事件绑定
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.click_1)
        self.pushButton_3.clicked.connect(self.start_camera_detection)  # 绑定摄像头检测
        self.pushButton_4.clicked.connect(self.handleCalc3)
        self.printf('')
        
    def start_camera_detection(self):
        """启动摄像头检测"""
        self.printf("正在启动摄像头检测...")
        try:
            # 先清空显示区域
            self.label_2.setText("正在初始化摄像头...")
            self.label_2.repaint()
            
            # 最关键：停止之前所有运行的线程，用更安全的方式
            if hasattr(self, 'thread_1') and self.thread_1.isRunning():
                try:
                    self.printf("正在安全停止之前的检测任务...")
                    # 只设置停止标志，不使用terminate()
                    if hasattr(self.thread_1, 'is_running'):
                        self.thread_1.is_running = False
                    
                    # 等待有限的时间，避免无限等待
                    if not self.thread_1.wait(500):  
                        self.printf("检测任务仍在运行，继续等待...")
                        # 不强制中断，允许程序继续
                except Exception as e:
                    self.printf(f"停止线程1时出错: {str(e)}")
                    
            if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
                try:
                    self.printf("正在安全停止之前的摄像头任务...")
                    # 使用定义好的stop方法，不调用terminate
                    self.camera_thread.stop()
                    
                    # 短暂等待，但不强制阻塞
                    if not self.camera_thread.wait(500):
                        self.printf("摄像头线程仍在运行，继续等待...")
                        # 不强制中断，允许程序继续
                except Exception as e:
                    self.printf(f"停止摄像头线程时出错: {str(e)}")
            
            # 检查摄像头 - 不阻塞UI主线程
            self.check_camera_thread = QThread()
            self.check_camera_thread.run = self._check_camera_and_continue
            self.check_camera_thread.start()
                
        except Exception as e:
            self.printf(f"启动摄像头检测失败: {str(e)}")
            # 输出更详细的错误信息
            import traceback
            self.printf(traceback.format_exc())
            # 出现异常时清理资源
            torch.cuda.empty_cache()
            
    def _check_camera_and_continue(self):
        """检查摄像头是否可用并继续启动检测线程"""
        try:
            # 检查摄像头
            temp_cap = cv2.VideoCapture(0)
            if not temp_cap.isOpened():
                ui.printf("错误：无法打开摄像头，请检查摄像头连接或驱动")
                temp_cap.release()
                return
            temp_cap.release()
            
            # 清理所有OpenCV窗口
            cv2.destroyAllWindows()
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 再次清理GPU内存和UI
            torch.cuda.empty_cache()
            
            # 强制执行垃圾回收
            import gc
            gc.collect()
            
            # 等待UI更新和资源释放完成
            for i in range(3):
                QThread.msleep(100)
            
            # 确保模型已加载
            global model, stride, names, pt
            if 'model' not in globals() or model is None:
                ui.printf("正在加载模型...")
                model, stride, names, pt = load_model(verbose=True)
                ui.printf("模型加载完成")
                
            # 再次清空显示区域
            ui.label_2.setText("正在启动摄像头...")
            ui.label_2.repaint()
            QApplication.processEvents()
                
            # 创建并启动摄像头线程
            ui.camera_thread = Thread_Camera()
            ui.camera_thread.start()
            
        except Exception as e:
            ui.printf(f"摄像头检查失败: {str(e)}")

    def handleCalc2(self):
        capture = cv2.VideoCapture(0)
        while True:
            _, image = capture.read()
            if image is None:
                break
            try:
                results = run(model, image, stride, pt)  # 识别， 返回多个数组每个第一个为结果，第二个为坐标位置
                for i in results:
                    box = i[1]
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = [0, 0, 255]
                    ui.printf(
                        str(time.strftime('%Y.%m.%d %H:%M:%S ', time.localtime(time.time()))) + '检测到' + str(i[0]))
                    cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                    if i[2] > score:
                        score = float(i[0])
                    cv2.putText(image, str(i[0]) + ' ' + str(score), (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            except:
                pass
            ui.showimg(image)
            QApplication.processEvents()

    def openfile(self):
        global sname, filepath
        fname = QFileDialog()
        fname.setAcceptMode(QFileDialog.AcceptOpen)
        fname, _ = fname.getOpenFileName()
        if fname == '':
            return
        filepath = os.path.normpath(fname)
        sname = filepath.split(os.sep)
        ui.printf("当前选择的文件路径是：%s" % filepath)


    def handleCalc3(self):
        os._exit(0)

    def printf(self,text):
        self.textBrowser.append(text)
        self.cursor = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursor.End)
        QtWidgets.QApplication.processEvents()

    def showimg(self,img):
        try:
            if img is None or img.size == 0:
                return
                
            # 限制图像尺寸，防止处理过大图像
            max_width, max_height = 1280, 720
            h, w = img.shape[:2]
            
            # 如果图像太大，先调整大小再处理
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
            # 转换颜色空间
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 创建QImage
            _image = QtGui.QImage(img2.data, img2.shape[1], img2.shape[0], img2.shape[1] * 3,
                                QtGui.QImage.Format_RGB888)
                                
            # 计算缩放比例以适应标签大小
            label_width = self.label_2.width()
            label_height = self.label_2.height()
            n_width = _image.width()
            n_height = _image.height()
            
            if n_width / label_width >= n_height / label_height:
                ratio = n_width / label_width
            else:
                ratio = n_height / label_height
                
            # 限制最小缩放比例，确保不会过度放大小图像
            ratio = max(ratio, 0.1)
            
            new_width = int(n_width / ratio)
            new_height = int(n_height / ratio)
            
            # 创建缩放后的图像
            new_img = _image.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 显示图像
            self.label_2.setPixmap(QPixmap.fromImage(new_img))
            
            # 确保及时释放图像资源
            del img2
            del _image
            del new_img
            
        except Exception as e:
            self.printf(f"显示图像时出错: {str(e)}")

    def click_1(self):
        global filepath
        try:
            # 先停止所有可能正在运行的线程
            if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
                self.printf("正在安全停止摄像头检测...")
                self.camera_thread.stop()
                # 短暂等待但不阻塞
                if not self.camera_thread.wait(500):
                    self.printf("摄像头线程正在停止中...")
            
            # 停止之前的处理线程
            if hasattr(self, 'thread_1') and self.thread_1.isRunning():
                self.printf("正在安全停止之前的检测任务...")
                if hasattr(self.thread_1, 'stop'):
                    self.thread_1.stop()
                # 短暂等待但不阻塞
                if not self.thread_1.wait(500):
                    self.printf("处理线程正在停止中...")
            
            # 清理资源
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()
            
            # 清理UI
            self.label_2.setText("正在处理...")
            self.label_2.repaint()
            QApplication.processEvents()
            
            # 确保已经选择了文件
            if 'filepath' not in globals() or not filepath:
                self.printf("请先选择要处理的图像或视频文件")
                return
                
            # 创建新的处理线程，不等待之前的线程完全停止
            self.printf(f"开始处理文件: {filepath}")
            self.thread_1 = Thread_1(filepath)  # 创建线程
            self.thread_1.start()  # 开始线程
        except Exception as e:
            self.printf(f"启动处理任务失败: {str(e)}")
            import traceback
            self.printf(traceback.format_exc())




if __name__ == "__main__":
    try:
        # 创建全局变量
        global MainWindow, ui, model, stride, names, pt
        
        # 创建应用
        app = QApplication(sys.argv)
        
        # 初始化数据库
        init_db()
        
        # 只加载一次模型
        print("正在加载模型...")
        model, stride, names, pt = load_model(verbose=False)  # 设置verbose=False减少输出
        print("模型加载完成")
        
        # 创建登录窗口并显示
        login_window = LoginWindow()
        login_window.setWindowTitle("车道线检测系统登录")
        login_window.show()
        print("登录窗口已显示")
        
        # 确保窗口处理完成
        QApplication.processEvents()
        
        # 进入事件循环
        sys.exit(app.exec_())
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

