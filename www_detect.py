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

weights = ROOT / 'best.pt'  # 权重文件地址   .pt文件
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
# 获取设备
device = select_device(device)

# 载入模型
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # 检查图片尺寸

# Half
# 使用半精度 Float16 推理
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()


def detect(img):
    # Dataloader
    # 载入数据


    # Run inference
    # 开始预测

    dt, seen = [0.0, 0.0, 0.0], 0

    # 对图片进行处理
    im0 = img
    # Padded resize
    im = letterbox(img, imgsz, stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    # 预测
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # 用于存放结果
    detections = []

    # Process predictions
    for i, det in enumerate(pred):  # per image 每张图片
        seen += 1
        # im0 = im0s.copy()
        # 画框
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            # 写入结果
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]]  # 检测到目标位置，格式：（left，top，w，h）
                # 获取类别索引
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

                location = pyautogui.position()
                print(location)



                cls = names[int(cls)]
                conf = float(conf)
                detections.append({'class': cls, 'conf': conf, 'position': xywh})

    # Stream results
        im0 = annotator.result()

    # 输出结果

    # for i in detections:
    #     print(i)


    cv2.imshow('test', im0)
    pushKeyAsc=cv2.waitKey(1);
    #如果按下的键时esc时
    if pushKeyAsc%256 == 27:
        #关闭所有窗口
        cv2.destroyAllWindows();
        #退出循环并打印出文件
        exit("停止截屏")

    return detections

    # pyautogui.moveTo(int(xywh[0]) + 290 + xywh[3] // 4, int(xywh[1]) + xywh[3] // 2)

    # return detections


# path = 'D://Project//copy//yolov5-master//data//images//1e9e1ca1c7e3d1f3471d601d12066f683cb0bb57.jpg'
# img = cv2.imread(path)


#mss 用于截图
sct =mss.mss();
#monitor 就是我们截取屏幕图片的大小位置的配置，前面两个是起始点的位置 后面的宽度和高度是截取屏幕图片的大小
monitor={'left':290,'top':0,'width':960,'height':640}
#自定义的窗口名字
window_name='test'
#死循环
while True:

    # 目的：截取当前屏幕，然后使用numpy转化为矩阵，再用cv2.imshow将图片展示出来，达到一个实时屏幕的目的

    #抓取屏幕，monitor使我们自定义的截图图片的配置，就在上面

    img =sct.grab(monitor=monitor)
    imgArray= np.array(img)
    # 将图片转 BGR
    imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGRA2BGR)
    # imgArray = np.array(img)
    # 传入一张图片
    res = detect(imgArray)
    # print(*res)
    # print(res)

    aim=[0,0]

    for item in res:
            if(item["class"]=="person" and item["conf"]>=0.55):
                print(item["position"])
                xywh=item["position"]

                location = pyautogui.position()
                aim[0]=location.x
                aim[1]=location.y
                if((location.x ** 2 +location.y ** 2)-(aim[0]**2 +aim[1]**2)<= (location.x ** 2 +location.y ** 2) - ((int(xywh[0]) + 290 + xywh[2] // 2)**2 +(int(xywh[1]) + xywh[3]//2) **2)):
                    pyautogui.moveTo(int(xywh[0]) + 290 + xywh[2] * 0.5  , int(xywh[1]) + xywh[3]*0.5  )
                    # pyautogui.move(0,xywh[2]*0.000)
                    aim[0]=int(xywh[0]) + 290 + xywh[2] // 2
                    aim[1]=int(xywh[1]) + xywh[3]//2
                    break







