import tensorflow as tf
import keras
from keras.models import load_model
from keras import layers,models, regularizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Conv3D, MaxPooling3D, GlobalAveragePooling3D, Reshape, Lambda

import time
import os
class BasicBlock(layers.Layer):
    def __init__(self,filter_num,name,stride=1, **kwargs):
        super(BasicBlock, self).__init__( **kwargs)
        #filter_num=64
        self.filter_num = filter_num
        self.stride = stride
        #layers
        self.layers = []

        #padding string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式；

        self.conv1=keras.layers.Conv2D(filter_num,3,strides=stride,padding='same', name = name+'_1')
        # self.bn1=layers.BatchNormalization()
        #激活函数是relu函数
        self.relu=layers.Activation('relu')

        self.conv2=layers.Conv2D(filter_num,3,strides=1,padding='same', name = name+'_2')
        # self.bn2 = layers.BatchNormalization()
        self.layers.append(self.conv1)
        self.layers.append(self.conv2)
        # self.layers.append(self.bn1)
        # self.layers.append(self.bn2)
        #默认的stride=1 如果stride！=1 ，
        if stride!=1:
            #Sequential 序贯模型
            #序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。
            #就是初始化模型
            self.downsample=models.Sequential()

            self.downsample.add(layers.Conv2D(filter_num,1,strides=stride))
            self.layers.append(self.downsample)
        else:
            self.downsample=lambda x:x

    def get_layer(self, index):
        return self.layers[index]

    def get_layers(self):
        return self.layers

    def call(self,input,training=None):
        out=self.conv1(input)
        # out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        # out=self.bn2(out)

        identity=self.downsample(input)
        output=layers.add([out,identity])
        output=tf.nn.relu(output)
        return output

    def get_config(self):
        config = {
            'filter_num':
                self.filter_num,
            'stride':
               self.stride
        }

        base_config = super(BasicBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model:
    def __init__(self, input_shape, act_dim):
        # ACTION_DIM = 7
        self.act_dim = act_dim
        # INPUT_SHAPE =(4,200,400,3)
        self.input_shape = input_shape
        #_build_model是自定义函数，在下面可自行查看
        self._build_model()
        self.act_loss = []
        self.move_loss = []

    def load_model(self):

        # self.shared_model = load_model("./model/shared_model.h5", custom_objects={'BasicBlock': BasicBlock})
        #如果目录存在/model/act_part.h5
        if os.path.exists("./model/act_part.h5"):
            print("load action model")
            #Sequential 序贯模型
            #序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。
            #就是初始化模型
            self.act_model = models.Sequential()

            #custom_objects 放入 自定义层"BasicBlock",自定义层就在上面，导入模型act_part
            self.private_act_model = load_model("./model/act_part.h5", custom_objects={'BasicBlock': BasicBlock})
            # self.act_model.add(self.shared_model)
            self.act_model.add(self.private_act_model)
            
        if os.path.exists("./model/move_part.h5"):
            print("load move model")
            self.move_model = models.Sequential()
            self.private_move_model = load_model("./model/move_part.h5", custom_objects={'BasicBlock': BasicBlock})
            # self.move_model.add(self.shared_model)
            self.move_model.add(self.private_move_model)

        
        
        

        
        
        

    def save_mode(self):
        print("save model")
        self.private_act_model.save("./model/act_part.h5")
        self.private_move_model.save("./model/move_part.h5")


    def build_resblock(self,filter_num,blocks,name="Resnet",stride=1):
        res_blocks= models.Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num,name+'_1',stride))
        # just down sample one time
        for pre in range(1,blocks):
            res_blocks.add(BasicBlock(filter_num,name+'_2',stride=1))
        return res_blocks


    # use two groups of net, one for action, one for move
    def _build_model(self):

       # ------------------ build evaluate_net ------------------

       
        self.shared_model = models.Sequential()
        self.private_act_model = models.Sequential()
        self.private_move_model = models.Sequential()

        # shared part
        # pre-process block
        # self.shared_model.add(Conv2D(64, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))
        # # self.shared_model.add(BatchNormalization(name='b1'))
        # self.shared_model.add(Activation('relu'))
        # self.shared_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))
        
        # # resnet blocks
        # self.shared_model.add(self.build_resblock(64, 2, name='Resnet_1'))
        # self.shared_model.add(self.build_resblock(80, 2, name='Resnet_2', stride=2))
        # self.shared_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))

        # output layer for action model
        #private_act_model 动作模型构建

        #32
        # filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。

        #   (2, 3, 3)
        #kernel_size: 一个整数，或者 3 个整数表示的元组或列表， 指明 3D 卷积窗口的深度、高度和宽度。 可以是一个整数，为所有空间维度指定相同的值。

       # strides = (1, 2, 2)
       # strides: 一个整数，或者3个整数表示的元组或列表， 指明沿深度、高度和宽度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。



       # input_shape=(4,200,400,3)

       #  当使用该层作为模型第一层时，需要提供 input_shape 参数 （整数元组，不包含样本表示的轴），
       # 例如， input_shape=(128, 128, 128, 3) 表示尺寸 128x128x128 的 3 通道立体，
        self.private_act_model.add(keras.layers.Conv3D(32, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))

        self.private_act_model.add(Activation('relu'))
        self.private_act_model.add(Conv3D(48, (2,3,3),strides=(1,1,1), input_shape=self.input_shape, name='conv2'))
        self.private_act_model.add(Activation('relu'))
        self.private_act_model.add(Conv3D(64, (2,3,3),strides=(1,1,1), input_shape=self.input_shape, name='conv3'))
        self.private_act_model.add(Activation('relu'))
        self.private_act_model.add(Lambda(lambda x:tf.reduce_sum(x, 1)))
        # self.private_act_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))
        # resnet blocks
        self.private_act_model.add(self.build_resblock(64, 2, name='Resnet_1'))
        self.private_act_model.add(self.build_resblock(96, 2, name='Resnet_2', stride=2))
        self.private_act_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))
        self.private_act_model.add(self.build_resblock(256, 2, name='Resnet_4', stride=2))
        self.private_act_model.add(GlobalAveragePooling2D())
        # self.private_act_model.add(Reshape((1, -1)))
        # self.private_act_model.add(CuDNNLSTM(32))
        self.private_act_model.add(Dense(self.act_dim, name="d1"))        # action model
        self.private_act_model.summary()
        self.act_model = models.Sequential()
        # self.act_model.add(self.shared_model)
        self.act_model.add(self.private_act_model)
 

        # output layer for move model
        self.private_move_model.add(Conv3D(32, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))
        self.private_move_model.add(Activation('relu'))
        self.private_move_model.add(Conv3D(48, (2,3,3),strides=(1,1,1), input_shape=self.input_shape, name='conv2'))
        self.private_move_model.add(Activation('relu'))
        self.private_move_model.add(Conv3D(64, (2,3,3),strides=(1,1,1), input_shape=self.input_shape, name='conv3'))
        self.private_move_model.add(Activation('relu'))
        self.private_move_model.add(Lambda(lambda x:tf.reduce_sum(x, 1)))
        # self.private_move_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))
        
        # resnet blocks
        self.private_move_model.add(self.build_resblock(64, 2, name='Resnet_1'))
        self.private_move_model.add(self.build_resblock(96, 2, name='Resnet_2', stride=2))
        self.private_move_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))
        self.private_move_model.add(self.build_resblock(256, 2, name='Resnet_4', stride=2))
        self.private_move_model.add(GlobalAveragePooling2D())
        # self.private_move_model.add(Reshape((1, -1)))
        # self.private_move_model.add(CuDNNLSTM(32))
        self.private_move_model.add(Dense(4, name="d1"))

        # action model
        self.move_model = models.Sequential()
        # self.move_model.add(self.shared_model)
        self.move_model.add(self.private_move_model)




    #     # ------------------ build target_model ------------------
    #    # shared part
       
    #     self.shared_target_model = models.Sequential()
    #     # pre-process block
    #     self.shared_target_model.add(Conv3D(64, (2,3,3),strides=(1,2,2), input_shape=self.input_shape, name='conv1'))
    #     self.shared_target_model.add(BatchNormalization(name='b1'))
    #     self.shared_target_model.add(Activation('relu'))
    #     self.shared_target_model.add(MaxPooling3D(pool_size=(2,2,2), strides=1, padding="VALID", name='p1'))
        
    #     # resnet blocks
    #     self.shared_target_model.add(self.build_resblock(64, 2, name='Resnet_1'))
    #     self.shared_target_model.add(self.build_resblock(80, 2, name='Resnet_2', stride=2))
    #     self.shared_target_model.add(self.build_resblock(128, 2, name='Resnet_3', stride=2))

    #     # output layer for action model
    #     self.private_act_target_model = models.Sequential()
    #     self.private_act_target_model.add(self.build_resblock(200, 2, name='Resnet_4', stride=2))
    #     self.private_act_target_model.add(GlobalAveragePooling3D())
    #     # self.private_act_target_model.add(Reshape((1, -1)))
    #     # self.private_act_target_model.add(CuDNNLSTM(32))
    #     self.private_act_target_model.add(Dense(self.act_dim, name="d1", kernel_regularizer=regularizers.L2(0.001)))

    #     # action model
    #     self.act_target_model = models.Sequential()
    #     self.act_target_model.add(self.shared_target_model)
    #     self.act_target_model.add(self.private_act_target_model)
 

    #     # output layer for move model
    #     self.private_move_target_model = models.Sequential()
    #     self.private_move_target_model.add(self.build_resblock(200, 2, name='Resnet_4', stride=2))
    #     self.private_move_target_model.add(GlobalAveragePooling3D())
    #     # self.private_move_target_model.add(Reshape((1, -1)))
    #     # self.private_move_target_model.add(CuDNNLSTM(32))
    #     self.private_move_target_model.add(Dense(4, name="d1", kernel_regularizer=regularizers.L2(0.001)))

    #     # action model
    #     self.move_target_model = models.Sequential()
    #     self.move_target_model.add(self.shared_target_model)
    #     self.move_target_model.add(self.private_move_target_model)


    def predict(self, input):
        
        input = tf.expand_dims(input,axis=0)
        # shard_output = self.shared_model.predict(input)
        pred_move = self.private_move_model(input)
        pred_act = self.private_act_model(input)
        return pred_move, pred_act