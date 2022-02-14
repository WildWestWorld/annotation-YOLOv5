import mss
import numpy as np
import cv2
import win32con
import win32gui
from skimage.transform import rescale



#mss 用于截图
sct =mss.mss();
#monitor 就是我们截取屏幕图片的大小位置的配置，前面两个是起始点的位置 后面的宽度和高度是截取屏幕图片的大小
monitor={'left':290,'top':0,'width':800,'height':1080}
#自定义的窗口名字
window_name='test'
#死循环
while True:

    # 目的：截取当前屏幕，然后使用numpy转化为矩阵，再用cv2.imshow将图片展示出来，达到一个实时屏幕的目的

    #抓取屏幕，monitor使我们自定义的截图图片的配置，就在上面

    img =sct.grab(monitor=monitor)


    #图片转为数组 np是上面的numpy 就是他的array方法才能转化图片
    imgArray= np.array(img)
    #等比例缩放
    img1 = cv2.resize(imgArray, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_NEAREST)  # 修改图片的尺寸

    #使用cv2的imshow方法展示出来，里面要填窗口的名字和图片的数组。名字我们在上面已经定义好了，图片的数组也转化好了
    cv2.imshow(window_name,img1)




#waitKey() 函数的功能是不断刷新图像 , 频率时间为delay , 单位为ms
#返回值为当前键盘按键值

#1. waitKey()–是在一个给定的时间内(单位ms)等待用户按键触发;
# 如果用户没有按下键,则继续等待 (循环)
# 常见 : 设置 waitKey(0) , 则表示程序会无限制的等待用户的按键事件
# 一般在 imgshow 的时候 , 如果设置 waitKey(0) , 代表按任意键继续

# 2. 显示视频时，延迟时间需要设置为 大于0的参数
# delay > 0时 , 延迟 ”delay”ms , 在显示视频时这个函数是有用的 ,
# 用于设置在显示完一帧图像后程序等待 ”delay”ms 再显示下一帧视频 ;
# 如果使用 waitKey(0) 则只会显示第一帧视频

    #  cv2.waitKey(1)的返回值是按键的ascii码值 esc的ascii码值是27
    pushKeyAsc =cv2.waitKey(1);
    #如果按下的键时esc时
    if pushKeyAsc%256 == 27:
        #关闭所有窗口
        cv2.destroyAllWindows();
        #退出循环并打印出文件
        exit("停止截屏")
