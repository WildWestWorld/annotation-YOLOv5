# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import cv2
import time
import collections
import matplotlib.pyplot as plt

from Model import Model
from DQN import DQN
from Agent import Agent
from ReplayMemory import ReplayMemory






HP_WIDTH = 768
HP_HEIGHT = 407
WIDTH = 400
HEIGHT = 200
ACTION_DIM = 7
FRAMEBUFFERSIZE = 4
INPUT_SHAPE = (FRAMEBUFFERSIZE, HEIGHT, WIDTH, 3)

MEMORY_SIZE = 200  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 24  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 10  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.00001  # 学习率
GAMMA = 0

action_name = ["Attack", "Attack_Up",
           "Short_Jump", "Mid_Jump", "Skill_Up", 
           "Skill_Down", "Rush", "Cure"]

move_name = ["Move_Left", "Move_Right", "Turn_Left", "Turn_Right"]

DELAY_REWARD = 1




def run_episode(hp, algorithm,agent,act_rmp_correct, move_rmp_correct,PASS_COUNT,paused):
    #重启游戏，自定义的脚本
    restart()
    # learn while load game
    for i in range(8):
        #MEMORY_WARMUP_SIZE = 24   replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   

        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

    
    step = 0
    done = 0
    total_reward = 0

    start_time = time.time()
    # Delay Reward
    #初始化奖励池
    DelayMoveReward = collections.deque(maxlen=DELAY_REWARD)
    DelayActReward = collections.deque(maxlen=DELAY_REWARD)
    DelayStation = collections.deque(maxlen=DELAY_REWARD + 1) # 1 more for next_station
    DelayActions = collections.deque(maxlen=DELAY_REWARD)
    DelayDirection = collections.deque(maxlen=DELAY_REWARD)

    #当游戏画面开始时，会有黑屏的情况，为了保证数据集的纯净度，只有读取到BOSS血量和人物血量时才开始传输数据
    while True:
        boss_hp_value = hp.get_boss_hp()
        self_hp = hp.get_self_hp()
        if boss_hp_value > 800 and  boss_hp_value <= 900 and self_hp >= 1 and self_hp <= 9:
            break
        
    #传输视频
    #我们就直接换成截图工具就行了
    thread1 = FrameBuffer(1, "FrameBuffer", WIDTH, HEIGHT, maxlen=FRAMEBUFFERSIZE)
    thread1.start()

    last_hornet_y = 0
    while True:
        step += 1
        # last_time = time.time()
        # no more than 10 mins
        # if time.time() - start_time > 600:
        #     break

        # in case of do not collect enough frames


        #如果线程传递的帧数小于我们设定的参数
        while(len(thread1.buffer) < FRAMEBUFFERSIZE):
            time.sleep(0.1)

        #stations当前的画面
        stations = thread1.get_buffer()
        #boss当前的血量
        boss_hp_value = hp.get_boss_hp()
        #主角自己的血量
        self_hp = hp.get_self_hp()
        #主角位置
        player_x, player_y = hp.get_play_location()
        #boss的位置
        hornet_x, hornet_y = hp.get_hornet_location()
        #主角当前的灵魂数
        soul = hp.get_souls()

        #用于判断boss是否使用技能的语句 为true的时候boss就使用了技能
        hornet_skill1 = False
        if last_hornet_y > 32 and last_hornet_y < 32.5 and hornet_y > 32 and hornet_y < 32.5:
            hornet_skill1 = True
        last_hornet_y = hornet_y

        #将当前的信息传递给神经网络
        move, action = agent.sample(stations, soul, hornet_x, hornet_y, player_x, hornet_skill1)

        
        # action = 0
        take_direction(move)
        take_action(action)
        
        # print(time.time() - start_time, " action: ", action_name[action])
        # start_time = time.time()


        #再次开启线程
        next_station = thread1.get_buffer()
        next_boss_hp_value = hp.get_boss_hp()
        next_self_hp = hp.get_self_hp()
        next_player_x, next_player_y = hp.get_play_location()
        next_hornet_x, next_hornet_y = hp.get_hornet_location()

        # get reward
        #获取移动反馈/奖励
        move_reward = Tool.Helper.move_judge(self_hp, next_self_hp, player_x, next_player_x, hornet_x, next_hornet_x, move, hornet_skill1)
        # print(move_reward)
        #获取动作反馈/奖励
        #done用于判断人物/BOSS 是否死亡 游戏是否结束
        act_reward, done = Tool.Helper.action_judge(boss_hp_value, next_boss_hp_value,self_hp, next_self_hp, next_player_x, next_hornet_x,next_hornet_x, action, hornet_skill1)
            # print(reward)
        # print( action_name[action], ", ", move_name[d], ", ", reward)
        #把得到的奖励和反馈都放进去
        DelayMoveReward.append(move_reward)
        DelayActReward.append(act_reward)
        DelayStation.append(stations)
        DelayActions.append(action)
        DelayDirection.append(move)



        if len(DelayStation) >= DELAY_REWARD + 1:
            if DelayMoveReward[0] != 0:
                move_rmp_correct.append((DelayStation[0],DelayDirection[0],DelayMoveReward[0],DelayStation[1],done))
            # if DelayMoveReward[0] <= 0:
            #     move_rmp_wrong.append((DelayStation[0],DelayDirection[0],DelayMoveReward[0],DelayStation[1],done))

        if len(DelayStation) >= DELAY_REWARD + 1:
            if mean(DelayActReward) != 0:
                act_rmp_correct.append((DelayStation[0],DelayActions[0],mean(DelayActReward),DelayStation[1],done))
            # if mean(DelayActReward) <= 0:
            #     act_rmp_wrong.append((DelayStation[0],DelayActions[0],mean(DelayActReward),DelayStation[1],done))

        station = next_station
        self_hp = next_self_hp
        boss_hp_value = next_boss_hp_value
            

        # if (len(act_rmp) > MEMORY_WARMUP_SIZE and int(step/ACTION_SEQ) % LEARN_FREQ == 0):
        #     print("action learning")
        #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp.sample(BATCH_SIZE)
        #     algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

        total_reward += act_reward
        paused = Tool.Helper.pause_game(paused)

        if done == 1:
            Tool.Actions.Nothing()
            break
        elif done == 2:
            PASS_COUNT += 1
            Tool.Actions.Nothing()
            time.sleep(3)
            break
        

    thread1.stop()


    #在死亡/胜利之后的黑屏进行训练
    for i in range(8):
        if (len(move_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("move learning")
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_correct.sample(BATCH_SIZE)
            algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   

        if (len(act_rmp_correct) > MEMORY_WARMUP_SIZE):
            # print("action learning")
            #BATCH_SIZE = 10  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
            #sample来自relayMemory
            batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_correct.sample(BATCH_SIZE)
            algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)
    # if (len(move_rmp_wrong) > MEMORY_WARMUP_SIZE):
    #     # print("move learning")
    #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = move_rmp_wrong.sample(1)
    #     algorithm.move_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)   

    # if (len(act_rmp_wrong) > MEMORY_WARMUP_SIZE):
    #     # print("action learning")
    #     batch_station,batch_actions,batch_reward,batch_next_station,batch_done = act_rmp_wrong.sample(1)
    #     algorithm.act_learn(batch_station,batch_actions,batch_reward,batch_next_station,batch_done)

    return total_reward, step, PASS_COUNT, self_hp


if __name__ == '__main__':

    # In case of out of memory
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True      #程序按需申请内存
    sess = tf.compat.v1.Session(config = config)

    
    total_remind_hp = 0
    #MEMORY_SIZE
    act_rmp_correct = ReplayMemory(MEMORY_SIZE, file_name='./act_memory')         # experience pool
    move_rmp_correct = ReplayMemory(MEMORY_SIZE,file_name='./move_memory')         # experience pool
    
    # new model, if exit save file, load it
    model = Model(INPUT_SHAPE, ACTION_DIM)  

    # Hp counter
    hp = Hp_getter()


    model.load_model()
    #DQN的更新算法
    algorithm = DQN(model, gamma=GAMMA, learnging_rate=LEARNING_RATE)
    #初始化agent
    agent = Agent(ACTION_DIM,algorithm,e_greed=0.12,e_greed_decrement=1e-6)
    
    # get user input, no need anymore
    # user = User()


    #按下F1就开始训练
    # paused at the begining
    paused = True
    paused = Tool.Helper.pause_game(paused)

    max_episode = 30000
    # 开始训练
    episode = 0
    PASS_COUNT = 0                                       # pass count
    #    max_episode = 30000
    #episode=0
    while episode < max_episode:    # 训练max_episode个回合，test部分不计算入episode数量
        # 训练
        episode += 1     
        # if episode % 20 == 1:
        #     algorithm.replace_target()


        #run_episode自定义函数，就在上面
        total_reward, total_step, PASS_COUNT, remind_hp = run_episode(hp, algorithm,agent,act_rmp_correct, move_rmp_correct, PASS_COUNT, paused)
       #如果训练的步数episode除以10 余数不等于1时就保存该文件
        if episode % 10 == 1:
            model.save_mode()
        #如果训练的步数刚好能够被5整除就保存
        if episode % 5 == 0:
        #    move_rmp_correct = ReplayMemory(MEMORY_SIZE,file_name='./move_memory')         # experience pool


            move_rmp_correct.save(move_rmp_correct.file_name)

        if episode % 5 == 0:
            act_rmp_correct.save(act_rmp_correct.file_name)

        #剩余的总血量

        total_remind_hp += remind_hp
        print("Episode: ", episode, ", pass_count: " , PASS_COUNT, ", hp:", total_remind_hp / episode)

