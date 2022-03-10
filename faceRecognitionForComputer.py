# 导入需要的库
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import mss
import numpy as np
import cv2
import pyautogui
import win32con
import win32gui
from skimage.transform import rescale

# 初始化目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# 导入letterbox
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

weights = ROOT / 'yolov5n.pt'  # 权重文件地址   .pt文件
source = ROOT / 'data/images'  # 测试数据文件(图片或视频)的保存路径
data = ROOT / 'data/coco128.yaml'  # 标签文件地址   .yaml文件

imgsz = (640, 640)  # 输入图片的大小 默认640(pixels)
conf_thres = 0.25  # object置信度阈值 默认0.25  用在nms中
iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
max_det = 1000  # 每张图片最多的目标数量  用在nms中
device = '0'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
dnn = False  # 使用OpenCV DNN进行ONNX推理
line_thickness = 3
# 参考文献 https://zhuanlan.zhihu.com/p/66368987
sct = mss.mss()

monitor = {'left': 290, 'top': 0, 'width': 960, 'height': 640}
while True:
    img = sct.grab(monitor=monitor);

    imgArr = np.array(img)

    # cv2.imshow("test", imgArr)


    # Padded resize
    #缩放图片，添加灰边 后面的[0] 是返回值中添加灰边后的图片返回的序号
    #letterbox要的img 就是变成数组的图片
    im = letterbox(imgArr, imgsz, 32, auto=True)[0]




    # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
    face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



    # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
    faces = face_engine.detectMultiScale(im, scaleFactor=1.3, minNeighbors=5)

    # 对每一张脸，进行如下操作
    for (x, y, w, h) in faces:
        # 画出人脸框，蓝色（BGR色彩体系），画笔宽度为2
        im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # imgArr = numpy.array(img)
    # 在"img2"窗口中展示效果图
    cv2.imshow('img2', im)

    pushKeyboard = cv2.waitKey(1);

    if (pushKeyboard % 256 == 27):
     cv2.destroyAllWindows();
     exit("停止截屏")
