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
import threading as th ;
import multiprocessing as mp;
from skimage.transform import rescale
import time


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
classes = 0  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = True  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
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

def moveMouse(res):
    aim = [0, 0]
    minDistanceXY = [1000000, 1000000];
    trueAim = [0, 0]
    location = pyautogui.position()

    for item in res:
            if(item["class"]=="person" and item["conf"]>=0.55):
                print(item["position"])
                xywh=item["position"]


                aim[0]=(int(xywh[0]) + 290 + xywh[2] * 0.5)
                aim[1]=int(xywh[1]) + xywh[3] * 0.5
                # 距离最小
                if(((location.x-aim[0])**2 + (location.y-aim[1])**2)**0.5 <=((minDistanceXY[0]+5)**2+(minDistanceXY[1]+5)**2)**0.5):
                    minDistanceXY[0] = location.x - aim[0]
                    minDistanceXY[1] = location.y - aim[1]
                    trueAim[0] = int(int(xywh[0]) + 290 + xywh[2] * 0.5)
                    trueAim[1] = int(int(xywh[1]) + xywh[3] * 0.5)
                    print(trueAim)

                # #位置总和最小
                # if(( abs(location.x-aim[0]) + abs(location.y-aim[1]) ) <(abs(minDistanceXY[0]-20)+abs(minDistanceXY[1]-20))) :
                #     minDistanceXY[0] = location.x - aim[0]
                #     minDistanceXY[1] = location.y - aim[1]
                #     trueAim[0] = int(int(xywh[0]) + 290 + xywh[2] * 0.5)
                #     trueAim[1] = int(int(xywh[1]) + xywh[3] * 0.5)
                #     print(trueAim)


    if(int(trueAim[0]) != 0 | int(trueAim[1]) !=0):
        # if(len(res) == 1):
        #     pyautogui.moveTo(trueAim[0], trueAim[1],_pause=False)
        # else:
        pyautogui.moveTo(trueAim[0], trueAim[1],duration=0.02)

                # pyautogui.click(x=int(xywh[0]) + 290 + xywh[2] * 0.5, y=int(xywh[1]) + xywh[3] * 0.5, button='left',
                #                 clicks=2, _pause=True,interval=0.1)
                # if((location.x -aim[0]) **2 +(location.y -aim[1])**2 >= (location.x-(int(xywh[0]) + 290 + xywh[2] * 0.5))**2+(location.y -(int(xywh[1]) + xywh[3] * 0.5))**2):





                # if((location.x -aim[0]) **2  >= (location.x-(int(xywh[0]) + 290 + xywh[2] * 0.5))**2):
                #     # pyautogui.click(x=int(xywh[0]) + 290 + xywh[2] * 0.5, y=int(xywh[1]) + xywh[3] * 0.5, button='left',
                #     #                 clicks=2, _pause=True,interval=0.1)
                #     pyautogui.moveTo(int(xywh[0]) + 290 + xywh[2] * 0.5, int(xywh[1]) + xywh[3] * 0.5)
                #     aim[0]=int(xywh[0]) + 290 + xywh[2] // 2
                #     aim[1]=int(xywh[1]) + xywh[3]//2
                #     break
                # else:
                #     pyautogui.moveTo(aim[0],  aim[1])
                #     break


def detect(img):
    # Dataloader
    # 载入数据

    # 记录该帧开始处理的时间
    # 用于FPS的计算
    start_time = time.time()

    # Run inference
    # 开始预测

    dt, seen = [0.0, 0.0, 0.0], 0

    # 对图片进行处理
    im0 = img
    # Padded resize
    #缩放图片，添加灰边 后面的[0] 是返回值中添加灰边后的图片返回的序号
    #letterbox要的img 就是变成数组的图片
    im = letterbox(img, imgsz, stride, auto=pt)[0]

    # Convert
    #原本的im可能是[Channels, H, W] 排列方式
    #transpose将im转置
    #现在的im是[W, Channels, H]
    im = im.transpose((2, 0, 1)) # HWC to CHW, BGR to RGB
    #ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    im = np.ascontiguousarray(im)

    t1 = time_sync()

    #从numpy.im创建一个张量。
    # 说明：返回的张量和im共享同一内存。对张量的修改将反映在im中，反之亦然。返回的张量是尽量不要改变大小，而是改变了大小之后就不共享一块内存空间了。

    im = torch.from_numpy(im).to(device)
    #如果half为True会使用半精度
    im = im.half() if half else im.float()  # uint8 to fp16/32
    #将矩阵的值的范围0-255 变为 0-1
    #归一化
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # 没有batch_size的话则在最前面添加一个轴
    #拓展池化层的维度
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    # 预测


    # augment = False  # 预测是否也要采用数据增强 TTA 默认False
    # visualize = False  # 特征图可视化 默认FALSE


    # 前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
    # h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
    # num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
    # pred[..., 0:4]为预测框坐标
    # 预测框坐标为xywh(中心点+宽长)格式
    # pred[..., 4]为objectness置信度
    # pred[..., 5:-1]为分类结果



    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    # conf_thres = 0.25  # object置信度阈值 默认0.25  用在nms中
    # iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
    # classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
    # agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False


    # pred:前向传播的输出
    # conf_thres:置信度阈值
    # iou_thres:iou阈值
    # classes:是否只保留特定的类别
    # agnostic:进行nms是否也去除不同类别之间的框
    # max_det = 1000  # 每张图片最多的目标数量  用在nms中

    # 经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
    # pred是一个列表list[torch.tensor]，长度为batch_size
    # 每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls


    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # 用于存放结果
    detections = []
    img=im0
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

        # 记录该帧处理完毕的时间
        end_time = time.time()
        # 计算每秒处理图像帧数FPS
        FPS = 1/(end_time - start_time)


        scaler = 1
        # 在图像上写FPS数值，参数依次为：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        img = cv2.putText(img, 'FPS  '+str(int(FPS)), (25 * scaler, 50 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)

    # 输出结果

    # for i in detections:
    #     print(i)




    return detections,img

    # pyautogui.moveTo(int(xywh[0]) + 290 + xywh[3] // 4, int(xywh[1]) + xywh[3] // 2)

    # return detections


# path = 'D://Project//copy//yolov5-master//data//images//1e9e1ca1c7e3d1f3471d601d12066f683cb0bb57.jpg'
# img = cv2.imread(path)

# 获取摄像头，传入0表示获取系统默认摄像头
camera = cv2.VideoCapture(0)

# 打开camera
camera.open(0)


#死循环
while camera.isOpened():
    # 获取画面
    success, frame = camera.read()
    if not success:
        break

    #镜像翻转
    frame = cv2.flip(frame, 1)
    # 将图片转 BGR

    # imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGRA2BGR)
    imgArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # imgArray = np.array(img)
    # 传入一张图片

    res = detect(frame)

    cv2.imshow('test', res[1])



    pushKeyAsc=cv2.waitKey(1);
    #如果按下的键时esc时
    if pushKeyAsc%256 == 27:
        #关闭所有窗口
        # 关闭摄像头
        camera.release()

        cv2.destroyAllWindows();
        #退出循环并打印出文件
        exit("停止截屏")
    # pool.map()
    # moveMouse(res)


    # print(*res)
    # print(res)










