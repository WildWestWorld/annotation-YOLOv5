import numpy as np
import pandas
import  time

#np.random.seed(n)函数用于生成指定随机数。
#伪随机数列，简单来说就是mc的地图种子。两个人使用同一种子肯定可以生成一样的地图，但是不同于mc地图的是，第二次调用的时候就不一样了
np.random.seed(2);

#下面是参数设定

#GameState。值就是总的游戏场景的数量
#我们的游戏应该是无限帧的，所以我们后续应该会修改这个参数，
# 这个指代的是一维的寻找宝藏出现的不同场景，此时为6，意味着大约6个场景
gameTotalState=6;
#agentAction=angent可以选择动作。值就是状态的数量
agentAction=['left','right','stay'];
#在公式中自带的是Epsilon是希腊字符，Epsilon是希腊字符，指代当前使用最优策略的可能性，1-Epsilon后得到的数，就是系统随机作出的动作
#个人认为我们不需要随机动作，AI就应该直接作出当前的情况下最优解，因为随机并不可靠
#chooseMaxValueRate选择最大收益的可能性
chooseMaxValueRate=0.99
#learningRate学习效率，在公式中是阿尔法A
learningRate =0.1

#discountFactor代表的是随着时间的奖励收益损失因子,在公式中是伽马也就是y
discountFactor=0.9

#maxEpisodes最大训练次数/步长
maxEpisodes=13


#参数设定止

#下面是自定义函数部分


#初始化Q-tabel表格函数，该函数的目的就是创建一个空的Qtable
# gameState代表当前场景的个数，agentAction就是采取的行动
def build_q_table(gameTotalState,actions):
    #pandas的DataFrame就像numpy中的矩阵，不过它拥有列名和索引名，实际操作起来会更方便一些
    #DataFrame(a,columns=b) a=内容，columns=b
    # #columns代表着表格的属性 =b b数列

    #np.zeros(x,y)，x行y列，
    # gameState=6, len(actions)=3
    #6行3列的0数组

    #columns代表着表格的属性
    table =pandas.DataFrame(np.zeros(
        (gameTotalState,len(actions))),columns=actions
    )
    print(table)
    return table;

#下面是选择动作的自定义函数
def choose_action_function(state,q_table):
    print(state)

    # iloc[:, :]前面的冒号就是取行数，后面的冒号是取列数
    #q_table.iloc[state,:] 意味着取出state行，和全部列,也就是取出当前state这一行
    state_actions =q_table.iloc[state,:]

    print(state_actions)
    #np.random.uniform()就是从0-1中随机产生个数，当然我们也可以自己设置随机数的范围，详情自己点开看
    #np.random.uniform()>chooseMaxValueRate 如果随机产生的数大于我们设置的chooseMaxValueRate选择最大收益的可能性
    #state_actions.all() == 0 如果当前状态的表格全部为0，也就是agent是第一次来到这里
    # 在agentAction数列中随机挑选属性
    if(np.random.uniform()>chooseMaxValueRate) or(state_actions.all() == 0):
        #np.random.choice从数组中随机抽取元素
        action_name =np.random.choice(agentAction)
    else:
        #选择state_actions 表格中的最大的值
        action_name=state_actions.idxmax();
        print(action_name)

    return action_name

#获得反馈
def getRewardAndNewState(state, action):
    # This is how agent will interact with the environment
    if action == 'right':  # move right
        #为什么要减1，因为tabel的序号是从0开始的
        #gameTotalState - 1指代的是终点
        if state == gameTotalState - 1:  # terminate
            next_state = 'end'
            reward = 1
        else:
            #如果没到终点就没有奖励
            next_state = state + 1
            reward = 0
    else:  # move left
        reward = 0
        if state == 0:
            next_state = state  # reach the wall
        else:
            next_state = state - 1
    return next_state, reward

#创建环境
def update_game(state, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (gameTotalState - 1) + ['?']  # '---------T' our environment
    if state == 'end':
        #episode是会使用在循环maxEpisodes使用的index值
        #因为index是从0开始所以加1
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        # 用于显示当前的state位置为?
        env_list[state] = '?'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(0.01)

#这个其实算是主函数，会用到上面写的所有代码
def playGame():
    #初始化表格
    q_table = build_q_table(gameTotalState, agentAction)

    #maxEpisodes最大训练次数/步长
    #其实训练2-3也是可以的，反正他是根据表格来作出决策
    for episode in range(maxEpisodes):
        step_counter = 0
        state = 0
        game_is_end = False
        #初始化游戏
        update_game(state, episode, step_counter)
        #如果游戏没有结束，就会一直循环下去
        while not game_is_end:
            #首先根据表格来选择动作
            choose_action = choose_action_function(state, q_table)
            #再使用刚刚从表格中使用的决策，和当前的state
            next_state, reward = getRewardAndNewState(state, choose_action)  # take action & get next state and reward
            #q_predict是表格中的当前动作的价值
            #刚开始就是表格是空的所以为0
            q_predict = q_table.loc[state, choose_action]
            #如果是
            if next_state != 'end':
                ##discountFactor代表的是随着时间的奖励收益损失因子,在公式中是伽马也就是y
                # discountFactor=0.9
                # q_table.iloc[next_state, :].max()选择下一个场景里面的最大值
                q_target = reward + discountFactor * q_table.iloc[next_state, :].max()  # next state is not terminal
            else:
                q_target = reward  # next state is terminal
                game_is_end = True  # terminate this episode

            q_table.loc[state, choose_action] += learningRate * (q_target - q_predict)  # update
            state = next_state  # move to next state

            update_game(state, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = playGame()
    print('\r\nQ-table:\n')
    print(q_table)

