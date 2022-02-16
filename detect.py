# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


#预测的主文件


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

#torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
@torch.no_grad()
def run(weights=ROOT / 'yolov5n.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    #souce 文件就是存放要预测的图片的目录，然后他把他强转成文字类型了 souce有可能不是路径也可能是地址
    source = str(source)
    #not 取反
    #nosave 是我们自定义的值，决定是否保存被检测完毕的图片 默认是false
    #source.endswith('.txt') 要预测的图片的目录不是以txt结尾
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    #[1:] 去除掉数列第一个元素后的数列
    #Path(source).suffix 获取到文件的后缀
    #判定文件的后缀名是否在我们自定义的图片格式数列或者视频格式数列
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    #判断souce是否为网站 先最小化，再检测是否是以http请求开头的
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    #source.isnumeric 判定source是否只由数字组成
    #source.endswith('.txt') 要预测的图片的目录不是以txt结尾
    #is_url and not is_file 是链接，且不是文件
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    #如果即是链接又是文件
    if is_url and is_file:
        source = check_file(source)  # download


    # Directories
    #该部分的主要作用就是创建输出目录
    #increment_path
    # 该函数用于自动生成目录并把训练好的图片放进去，而且目录不会重名因为程序写了自动将文件夹的末尾的数字+1，例如文件夹是exp1下一次就是exp2

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    #该部分的主要作用是加载模型

    #select_device(device) 检测我们使用的device 不过这个device默认是空的
    device = select_device(device)
    #根据我们输入的权重(yolov5n)，获得模型数据 这个是核心，我们目前还没看懂，需要再来多看几次
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    #将模型的各个参数都读取出来赋值
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    #调整图片的大小
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    #a &= b 等价于 a = a & b
    #也就是说若是half参数是false，该命令就不会触发
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        #pt的加载图像
        #里面是图片的路径
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    #Warmup有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳
    # 有助于保持模型深层的稳定性
    #此处的warmup似乎毫无作用
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup

    dt, seen = [0.0, 0.0, 0.0], 0
    #dataset 是我们加载的图像
    #返回值是  path, img, img0, self.cap, s
    for path, im, im0s, vid_cap, s in dataset:
        #计算时间的，可有可无
        t1 = time_sync()
        # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        im = torch.from_numpy(im).to(device)
        #如果half是true会触发该命令，但一般是false
        im = im.half() if half else im.float()  # uint8 to fp16/32
        #矩阵里面的值都会是0-1之间
        #也就是归一化处理
        im /= 255  # 0 - 255 to 0.0 - 1.0

        #一般的长度都会为三，
        if len(im.shape) == 3:
            #im[None]就是增加一个维度，在数组中填了个none
            #此时的im就变成四个维度了[1,x,x,x]
            im = im[None]  # expand for batch dim

         # 计算时间的，可有可无
        t2 = time_sync()
        # 计算时间的，可有可无
        dt[0] += t2 - t1

        #visualize开启时使用，我们也不开启
        # Inference
        # visualize开启时使用，我们也不开启
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        #预测的结果 有用
        pred = model(im, augment=augment, visualize=visualize)

        # 计算时间的，可有可无
        t3 = time_sync()
        # 计算时间的，可有可无
        dt[1] += t3 - t2

        # NMS
        #非极大值抑制（Non-Maximum Suppression，NMS），顾名思义就是抑制不是极大值的元素，可以理解为局部最大搜索。这个局部代表的是一个邻域，邻域有两个参数可变，一是邻域的维数，二是邻域的大小。
        #简单来说就是我们预测出来可能不止一个框，我们使用NMS把最有可能的框给选出来
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        #>>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        # >>> list(enumerate(seasons))
        # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                #frame= 0
                #pt会走这里
                #p=正在检测的图片所在的路径
                #im0是数组矩阵
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #p.name是文件（除开路径的那种）
            save_path = str(save_dir / p.name)  # im.jpg
            #'runs\\detect\\exp10\\labels\\1e46742a4266d2e0d18d7a10bfd4482665f2cf50'
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
           #'image 1/14 D:\\Project\\copy\\yolov5-master\\data\\images\\1e46742a4266d2e0d18d7a10bfd4482665f2cf50.jpg: 416x640 '
            #就是在后面加了个矩阵大小
            s += '%gx%g ' % im.shape[2:]  # print string
            # gn =tensor([3669, 2293, 3669, 2293])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc=im0
            imc = im0.copy() if save_crop else im0  # for save_crop

            #画框
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))


            #如果det有长度，也就是说 NMS后有数据
            if len(det):
                # Rescale boxes from img_size to im0 size
                #im的大小后面的
                #round()四舍五入

                #det[:, :4]  det前4个数组 0-4
                #tensor([[  1.50192,  20.34012, 192.13800, 403.37643],
                # [146.53203,  29.60571, 259.56622, 404.02249],
                 # [195.36108,  34.58754, 425.87671, 400.76617],
                  # [ -1.29509, 302.81317, 172.69252, 409.88971]], device='cuda:0')

                #im.shape[2:] 从第三个元素开始 im.shape[2:] = torch.Size([416, 640])

                #im0.shape =(716, 1146, 3)
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()


                #打印结果可
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        #这段代码我们需要，他里面是结果的类别，x的位置，y的位置，高度和宽度的百分比
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #我们会运行下面这段代码
                    #下面这段是绘制方框代码
                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 获取类别索引
                        c = int(cls)  # integer class
                        #如果隐藏标签参数是true就是none  就没标签了
                        #如果有hide_conf隐藏可能性指标，只显示标签的名字
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            #view_img自定义参数，当打开时会实时显示你的检测的结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


            #保存图片
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    #定义了一个参数的对象，就是用来装载下列配置的一个容器
    #argparse 模块可以让人轻松编写用户友好的命令行接口，程序定义它需要的参数，
    #创建解析器
    # argparse.ArgumentParser()创建一个 ArgumentParser 对象
    #这是argparse模块初始化语句
    parser = argparse.ArgumentParser()
    #通过调用 add_argument() 方法给 ArgumentParser对象添加程序所需的参数信息：
    # 第一个参数是名字，
    #nargs='+' 被parse_args会拼合成一个数列
    #action='store_true'相当于给了个默认值true
    # 选择要使用的权重模型 以及路径 默认路径是根目录
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5n.pt', help='model path(s)')
    #需要预测的图片的主目录
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    #图像resize到的尺寸，可根据自己实际任务需求，改成640等，需要32的倍数
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt',  default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    #通过 parse_args() 方法解析参数   parser使我们自定义的名字 初始化的时候自定义的
    opt = parser.parse_args()
    #如果我们上面设置的imgsz参数的长度是1的话，就把imgsz乘以2 ，不是1的话就乘以1
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #在命令行中打印出我们写入的各项值，第一个参数是log的名字，无所谓的了解下就行
    print_args(FILE.stem, opt)
    # 返回我们parse_args() 方法解析后的参数
    return opt


def main(opt):
    #查看依赖是否正确安装
    check_requirements(exclude=('tensorboard', 'thop'))
    #vars() 函数返回对象object的属性和属性值的字典对象。
    # 举例:
    #>>> print(vars(Runoob))
    # {'a': 1, '__module__': '__main__', '__doc__': None}
    #参数前面加上* 号 ，意味着参数的个数不止一个，另外带一个星号（*）参数的函数传入的参数存储为一个元组（tuple）
    #**将该对象以字典的形式储存

    #def t2(param1, **param2):

     #       print param1
     #       print param2
     #t2(1,a=2,b=3)


    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
