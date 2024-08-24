import copy
import datetime
import io
import random
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
from scipy.optimize import linear_sum_assignment
from itertools import accumulate
import bisect

'''
多维离散动作空间，只有加速，减速，维持等等

状态为到目标点 距离，方位，自己的速度，自己当前的方向，探测范围内障碍点的距离方位和角度

当前坐标变化为经纬度坐标系变换
'''

# 将list按照另一个数量list进行分割的函数

def split_list(original_list,sizes):
    '''

    eg. original_list=[4]*5 sizes=[2,3], return [[4,4],[4,4,4]
    :param original_list:
    :param sizes:
    :return:
    '''

    # 初始化一个空列表，用于存储划分后的子列表
    split_lists=[]
    # 初始化索引位置为0
    start_index=0

    # 遍历sizes列表中每个切分大小
    for size in sizes:
        # 如果当前索引加上切分大小超过了原始列表的长度，就结束函数
        if start_index + size >len(original_list):
            split_lists.append(original_list[start_index:])
            return split_lists
        # 通过切分划分列表
        end_index=start_index+size
        split_lists.append(original_list[start_index:end_index])
        # 更新下一次切片的起始索引
        start_index=end_index
    # 返回划分后的子列表集合
    return split_lists

def create_grid_map(size,resolutions,reference_point,obstacle_points):
    assert size%2==1, "size must be odd"
    grid_maps=[]

    grid_map = np.zeros((size, size))
    grid_x,grid_y=size // 2, size // 2
    grid_map[grid_x,grid_y] = 1
    grid_maps.append(grid_map)

    crash=0

    for i,r in enumerate(resolutions):
        grid_map = np.zeros((size, size))
        rx, ry = reference_point

        grid_rx, grid_ry = size // 2, size // 2

        max_x = rx + len(grid_map[0]) / 2 * r
        min_x = rx - len(grid_map[0]) / 2 * r
        max_y = ry + len(grid_map) / 2 * r
        min_y = ry - len(grid_map) / 2 * r

        for op in obstacle_points:

            tx, ty = op
            if tx==rx and ty==ry:
                continue

            if tx < max_x and tx > min_x and ty < max_y and ty > min_y:
                grid_x = grid_rx + math.ceil((tx - rx - r / 2) / r)
                grid_y = grid_ry + math.ceil((ty - ry - r / 2) / r)
                grid_map[grid_x, grid_y] = 2
                if grid_x==grid_rx and grid_y==grid_ry and i==0:
                    crash=1
        grid_maps.append(grid_map)
    grid_maps=np.stack(grid_maps)
    return crash,grid_maps



class AgentFollowEnv(gym.Env):
    def __init__(self,target_info,config_info=None):
        self.num_agents = config_info['num_agents']
        self.max_steps=config_info['max_steps']
        self.env_x=config_info['env_x']
        self.env_y=config_info['env_y']
        self.main_base_speed=config_info['main_speed']
        self.hard_crash=config_info['hard_crash']
        self.soft_crash=config_info['soft_crash']
        self.a_speed=config_info['a_speed']
        self.a_direction=config_info['a_direction']
        self.speed_limit=config_info['speed_limit']
        self.save_dir = config_info['save_dir']
        self.num_obstacles =config_info['num_obstacles']
        self.detect_radius = config_info['detect_radius']
        self.main_speed_num=config_info['main_speed_num']
        self.platform_list=config_info['platform_list']
        # 是否对state进行normalization
        self.state_norm=config_info['state_norm']
        # 是否plan
        self.plan=config_info['plan']
        # 是否随机障碍物体数量
        self.random_ob=config_info['random_ob']

        self.base_x=self.env_x
        self.base_y=self.env_y

        # 记录总局数
        self.episode=0

        # 检测最大障碍物数量
        self.obstacle_detect=4

        # 目标状态，记录每个实体相对于航母的目标位置和方位
        self.target_distances = []
        self.target_angles = []

        for i in range(self.num_agents):
            self.target_distances.append(target_info[str(i)][0])
            self.target_angles.append(target_info[str(i)][1])

        self.target_distances = np.array(self.target_distances)
        self.target_angles = np.array(self.target_angles)

        # 运动过程中的最大距离
        self.max_dis=(((2*self.env_x)*2)**2+((self.env_y*2)*2)**2)**0.5

        # main 移动相关参数
        self.move_directions = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]
        self.current_direction_index = 0

        # 定义状态空间：每个智能体包括目标位置、智能体位置、速度和方向
        pi = np.pi
        inf = 10000#np.inf

        # 定义动作空间：每个智能体包括速度和方向两个维度，多维离散,第一个是速度维度，第二个是方向维度
        self.action_space = gym.spaces.MultiDiscrete([3,3])

        self.low=np.array([-inf]*4+[0,0]*self.obstacle_detect)
        self.high=np.array([inf]*4+[inf,inf]*self.obstacle_detect)
        self.observation_space=spaces.Box(self.low,self.high,(len(self.low),),dtype=float)

        self.fig, self.ax = plt.subplots()



        # accumulate list
        self.accumulate_platform_list=list(accumulate(self.platform_list))


    # 更新main点位置，这里以正方形移动
    def update_main_position(self):
        # 沿着当前方向移动
        displacement = self.main_speed * self.move_directions[self.current_direction_index]
        new_main_position = self.main_position + displacement

        # 运行轨迹矩形边框
        x_min, x_max = 3*self.x_min / 6,  3*self.x_max / 6
        y_min, y_max = 3*self.x_min / 6,  3*self.x_max / 6

        # 判断是否需要改变方向
        if not (x_min <= new_main_position[0][0] <= x_max) or \
                not (y_min <= new_main_position[0][1] <= y_max):
            # 如果超出边界，改变方向
            self.current_direction_index = (self.current_direction_index + 1) % 4

            # 重新计算移动
            displacement = self.main_speed * self.move_directions[self.current_direction_index]

            # 计算移动后点位置
            new_main_position = self.main_position + displacement

        # 更新主目标位置
        self.main_position = new_main_position

        # 重置环境状态，随机设置目标位置和每个智能体初始位置、速度、方向
        self.main_position = np.clip(self.main_position, x_min, x_max)

    # 计算每个智能体的目标位置
    def calculate_target_coordinates_xy(self):
        # 威胁点到（航母）main点的向量差值
        danger_to_main = self.danger_position - self.main_position
        # 计算动态角度
        theta = np.arctan2(danger_to_main[0][1], danger_to_main[0][0])
        # 修正目标角度
        target_angles = np.mod(self.target_angles + theta, 2 * np.pi)

        # 根据距离和方位计算目标点的绝对位置
        target_x = self.main_position[0][0] + self.target_distances * np.cos(target_angles)
        target_y = self.main_position[0][1] + self.target_distances * np.sin(target_angles)

        # 将目标点进行组合
        target_coordinates = np.column_stack((target_x, target_y))

        # 是否进行位置分配，这里使用的是匈牙利算法
        if self.plan:
            # 将目标点和实体点按类型数量进行分割
            target_coordinates_list=split_list(target_coordinates,self.platform_list)
            agent_positions_list=split_list(self.agent_positions,self.platform_list)
            # 以下使用匈牙利算法计算
            for k in range(len(target_coordinates_list)):
                agent_position,target_position=agent_positions_list[k],target_coordinates_list[k]
                cost_matrix = np.zeros((len(agent_position), len(target_position)))
                for i in range(len(agent_position)):
                    for j in range(len(target_position)):
                        cost_matrix[i, j],_ = self.compute_distance_bearing_xy(agent_position[i:i+1], target_position[j:j+1])
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                target_coordinates_list[k] = np.array(target_position)[col_ind].tolist()
            target_coordinates=[sublist for sublist1 in target_coordinates_list for sublist in sublist1]


        return target_coordinates

    # 给出两个点，计算距离和方位
    def compute_distance_bearing_xy(self,main_position,agent_positions):
        differences=agent_positions-main_position

        # 直角坐标系下计算
        distances=np.linalg.norm(differences,axis=1)
        bearings=np.arctan2(differences[:,1],differences[:,0])
        bearings=np.mod(bearings,2*np.pi)

        return distances,bearings

    # 计算状态
    def compute_state(self):

        target_positions=self.calculate_target_coordinates_xy()

        distances,bearings=self.compute_distance_bearing_xy(self.agent_positions,target_positions)

        return distances,bearings

    # 计算实体移动后新的xy点，这里是根据实体速度以及移动方向进行计算
    def calculate_new_positions_xy(self,positions,speeds,directions):

        # 计算每个智能体的位移
        displacements = speeds[:, np.newaxis] * np.column_stack([np.cos(directions), np.sin(directions)])

        # 更新每个智能体位置
        positions += displacements


        return positions

    # 阵型改变函数，能够在训练过程中对阵型进行改变
    def change_target_info(self,target_info):
        self.target_distances = []
        self.target_angles = []
        for i in range(self.num_agents):
            self.target_distances.append(target_info[str(i)][0])
            self.target_angles.append(target_info[str(i)][1])
        self.target_distances = np.array(self.target_distances)
        self.target_angles = np.array(self.target_angles)

    # 环境初始化函数
    def reset(self, seed=0):
        # np.random.seed(seed)

        pi = np.pi
        inf = np.inf

        self.frames = []

        self.current_step = 0

        bs=np.random.choice([1,2,3])

        # self.env_x=self.base_x*bs
        # self.env_y=self.base_y*bs
        # # 运动过程中的最大距离
        # self.max_dis=(((2*self.env_x)*2)**2+((self.env_y*2)*2)**2)**0.5
        #

        # 设置矩形边界限制
        self.x_min, self.x_max = -self.env_x, self.env_y
        self.y_min, self.y_max = -self.env_x, self.env_y


        self.main_speed = np.random.choice([self.main_base_speed*(i) for i in range(self.main_speed_num)])

        self.danger_position = np.random.uniform(self.x_min,self.x_max,size=(2,))

        # 重置环境记录
        self.episode_length = 0
        self.episode_rewards = [0] * self.num_agents

        # 重置环境状态，随机设置参考点位置和每个智能体初始位置、速度、方向
        self.main_position = np.random.uniform(3*self.x_min/6, 3*self.x_max/6, size=(1,2))

        # 随机设置每个智能体初始位置，确保距离目标一定距离
        self.agent_positions = np.random.uniform(self.x_min, self.x_max, size=(self.num_agents, 2))

        # 随机障碍物的初始位置

        num_obstacles =np.random.randint(self.num_obstacles) if self.random_ob else self.num_obstacles
        self.obstacle_positions=np.random.uniform(self.x_min,self.x_max,size=(num_obstacles,2))

        # 拼接起所有的位置

        self.all_positions=np.concatenate([self.agent_positions,self.main_position,self.obstacle_positions])

        # 初始化agent的速度和方向
        self.speeds = np.zeros(self.num_agents)
        self.directions = np.zeros(self.num_agents)



        # 返回新的状态，包括目标位置、智能体位置、速度和方向
        states = []

        distances,bearings=self.compute_state()

        # 以下是生成每个智能体状态
        for i in range(self.num_agents):
            state_list = []

            brg2tgt = bearings[i]
            drn = self.directions[i]

            if self.state_norm: # 是否对状态进行normlization，做这个处理是为了更好将环境进行迁移
                dis2tgt = self.map_value_to_range(distances[i], self.max_dis,(1,5))
                spd =self.map_value_to_range(self.speeds[i], self.speed_limit[self.get_kind(i)], (0, 0.2))
            else:
                dis2tgt = distances[i]
                spd = self.speeds[i]

            state_list.append([dis2tgt, brg2tgt, drn, spd])

            _,grid_maps=create_grid_map(size=33,resolutions=[0.01,0.1,1],reference_point=self.agent_positions[i],obstacle_points=self.all_positions)

            state_list.append(grid_maps.reshape(-1).tolist())

            self.episode_rewards[i] = 0

            states.append(np.concatenate(state_list))


        return states

    # 映射函数
    def map_value_to_range(self,x,max_value,target_area=(0,5)):
        if x>target_area[0]:
            x=(x)*(target_area[1]-target_area[0])/(max_value)+target_area[0]
        return x

    # 根据下标获得实体属于哪种类型
    def get_kind(self,index):
        kind=bisect.bisect_left(self.accumulate_platform_list,index+1)
        return kind

    # 根据动作获得下一步状态与奖励
    def step(self, action):

        self.current_step += 1

        cnt=0
        for i in range(len(self.platform_list)):
            for j in range(cnt,cnt+self.platform_list[i]):

                if j>=len(self.agent_positions):
                    break

                speed_action, direction_action = action[j]

                # 根据速度动作更新速度
                if speed_action == 0:
                    self.speeds[j] = np.clip(self.speeds[j] - self.a_speed[i], 0, self.speed_limit[i])
                elif speed_action == 1:
                    self.speeds[j] = self.speeds[j]
                elif speed_action == 2:
                    self.speeds[j] = np.clip(self.speeds[j] + self.a_speed[i], 0, self.speed_limit[i])

                # 根据方向动作更新方向
                if direction_action == 0:
                    self.directions[j] = np.mod(self.directions[j] - self.a_direction[i], 2 * np.pi)
                elif direction_action == 1:
                    self.speeds[j] = self.speeds[j]
                elif direction_action == 2:
                    self.directions[j] = np.mod(self.directions[j] + self.a_direction[i], 2 * np.pi)


            cnt+=self.platform_list[i]

        # 根据智能体速度方向更新智能体位置
        self.agent_positions=self.calculate_new_positions_xy(self.agent_positions,self.speeds,self.directions)

        # 超出范围则进行取余数操作
        self.agent_positions = np.clip(self.agent_positions, self.x_min, self.x_max)

        # 更新航母位置点
        self.update_main_position()

        # 拼接起所有的位置
        self.all_positions=np.concatenate([self.agent_positions,self.main_position,self.obstacle_positions])

        # 计算智能体到目标位置的距离和方位
        distances,bearings=self.compute_state()

        # 计算个人奖励
        distance_reward = -0.1 * np.abs(distances)
        rewards=distance_reward


        # 判断是否达到终止条件，到达最大步数则终止
        done = self.current_step >= self.max_steps
        for i in range(self.num_agents):
            if distances[i]< 0.01:
                rewards[i]+=0.1


        # 返回新的状态，包括目标位置、智能体位置、速度和方向
        states = []

        distances, bearings = self.compute_state()

        # 计算每个实体的状态
        for i in range(self.num_agents):
            # 自身状态
            state_list = []

            brg2tgt = bearings[i]
            drn = self.directions[i]

            if self.state_norm:
                dis2tgt=self.map_value_to_range(distances[i],self.max_dis,(1,5))
                spd=self.map_value_to_range(self.speeds[i],self.speed_limit[self.get_kind(i)],(0,0.2))
            else:
                dis2tgt=distances[i]
                spd=self.speeds[i]

            state_list.append([dis2tgt,brg2tgt,drn,spd])

            penalty,grid_maps=create_grid_map(size=33,resolutions=[0.01,0.1,1],reference_point=self.agent_positions[i],obstacle_points=self.all_positions)

            # 当前智能体奖励加入惩罚
            rewards[i]+=penalty

            self.episode_rewards[i] += rewards[i]


            states.append(np.concatenate(state_list))

        # 组装info
        self.episode_length += 1
        info = {}
        if done:
            self.episode += 1
            details = {}
            details['r'] = self.episode_rewards # 当前局奖励
            details['l'] = self.episode_length # 当前局长度
            details['e'] = self.episode # 共进行了多少局
            info['episode'] = details

            if len(self.frames) != 0:
                timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
                save_path = f"{self.save_dir}/{timestamp}.gif"
                imageio.mimsave(save_path, self.frames, duration=0.1)  # Adjust duration as needed
                print("GIF saved successfully.")

        return states, rewards, done, info

    # 为实体进行箭头添加
    def add_arrow_xy(self, position, direction, color='black'):
        # 添加箭头
        arrow_length = self.x_max / 100
        arrow_head_width = self.x_max / 100

        arrow_dx = arrow_length * np.cos(direction)
        arrow_dy = arrow_length * np.sin(direction)

        self.ax.arrow(position[0], position[1], arrow_dx, arrow_dy, color=color, width=arrow_head_width)

    # 渲染函数，这里根据mode类型来判断是否进行gif生成
    def render(self, mode="human"):
        if not self.plan:
            self.normal_render(mode)
        else:
            self.plan_render(mode)
    # 一般渲染，没有normal类型的
    def normal_render(self, mode="human"):
        # 清空子图内容
        self.ax.clear()

        # 绘制参考点
        self.ax.plot(self.main_position[0][0], self.main_position[0][1], 'bs', label='main')

        # 绘制智能体应该到达的点
        target_positions = self.calculate_target_coordinates_xy()

        self.ax.plot(self.danger_position[0], self.danger_position[1], "r*")

        # 绘制障碍物
        for i in range(len(self.obstacle_positions)):
            # 使用不同颜色和标记标识每个智能体应该到达的目标
            marker = 'p'  # 可以根据需要选择不同的标记
            color = f'k'  # 使用Matplotlib的颜色循环
            label = f'obstackle {i + 1}'
            self.ax.plot(self.obstacle_positions[i][0], self.obstacle_positions[i][1], marker, color=color, label=label)

        # 绘制目标
        for i in range(self.num_agents):
            # 使用不同颜色和标记标识每个智能体应该到达的目标
            marker = '^'  # 可以根据需要选择不同的标记
            color = f'C{i}'  # 使用Matplotlib的颜色循环
            label = f'target {i + 1}'
            self.ax.plot(target_positions[i][0], target_positions[i][1], marker, color=color, label=label)

        # 绘制每个智能体
        for i in range(self.num_agents):
            # 使用不同颜色和标记标识每个智能体
            marker = 'o'  # 可以根据需要选择不同的标记
            color = f'C{i}'  # 使用Matplotlib的颜色循环
            label = f'agent {i + 1}'

            self.ax.plot(self.agent_positions[i, 0], self.agent_positions[i, 1], marker, color=color, label=label)

            # 添加运动方向箭头
            self.add_arrow_xy(self.agent_positions[i], self.directions[i])

            circle = plt.Circle((self.agent_positions[i, 0], self.agent_positions[i, 1]), self.detect_radius,
                                fill=False)

            self.ax.add_patch(circle)

        # 画出威胁和main的连线
        self.ax.plot([self.danger_position[0], self.main_position[0][0]],
                     [self.danger_position[1], self.main_position[0][1]],
                     "k--")

        # 设置图形范围
        self.ax.set_xlim(self.x_min, self.x_max)  # 根据需要调整范围
        self.ax.set_ylim(self.y_min, self.y_max)  # 根据需要调整范围

        # 添加图例
        # self.ax.legend()

        if mode == "human":
            # 显示图形
            plt.pause(0.1)  # 添加短暂的时间间隔，单位为秒
        else:
            # 将当前帧的图形添加到列表中
            self.frames.append(self.fig_to_array())

    # 规划类型的渲染
    def plan_render(self, mode="human"):
        # 清空子图内容
        self.ax.clear()


        # 绘制参考点
        self.ax.plot(self.main_position[0][0], self.main_position[0][1], 'bs', label='main')

        # 绘制智能体应该到达的点
        target_positions = self.calculate_target_coordinates_xy()

        # 绘制威胁点
        self.ax.plot(self.danger_position[0], self.danger_position[1], "r*")

        # 绘制障碍物
        for i in range(len(self.obstacle_positions)):
            # 使用不同颜色和标记标识每个智能体应该到达的目标
            marker = 'p'  # 可以根据需要选择不同的标记
            color = f'k'  # 使用Matplotlib的颜色循环
            label = f'obstackle {i + 1}'
            self.ax.plot(self.obstacle_positions[i][0], self.obstacle_positions[i][1], marker, color=color, label=label)

        # 绘制目标
        cnt = 0
        for i in range(len(self.platform_list)):
            for j in range(cnt, cnt + self.platform_list[i]):
                # 使用不同颜色和标记标识每个智能体应该到达的目标
                if j >= len(self.agent_positions):
                    break
                marker = '^'  # 可以根据需要选择不同的标记
                color = f'C{i}'  # 使用Matplotlib的颜色循环
                label = f'target {i + 1}'
                self.ax.plot(target_positions[j][0], target_positions[j][1], marker, color=color, label=label)
            cnt+=self.platform_list[i]

        # 绘制每个智能体

        cnt = 0
        for i in range(len(self.platform_list)):
            for j in range(cnt, cnt + self.platform_list[i]):
                if j >= len(self.agent_positions):
                    break
                # 使用不同颜色和标记标识每个智能体
                marker = 'o'   # 可以根据需要选择不同的标记
                color = f'C{i}'  # 使用Matplotlib的颜色循环
                label = f'agent {j}'

                self.ax.plot(self.agent_positions[j, 0], self.agent_positions[j, 1], marker, color=color, label=label)

                # 添加运动方向箭头
                self.add_arrow_xy(self.agent_positions[j], self.directions[j])

                circle = plt.Circle((self.agent_positions[j, 0], self.agent_positions[j, 1]), self.detect_radius, fill=False)
                # self.ax.add_patch(circle)

            cnt+=self.platform_list[i]

        # 画出威胁和main的连线
        self.ax.plot([self.danger_position[0], self.main_position[0][0]], [self.danger_position[1], self.main_position[0][1]],
                     "r-")

        for i in range(self.num_agents):
            self.ax.plot([self.agent_positions[i][0], target_positions[i][0]],
                         [self.agent_positions[i][1], target_positions[i][1]],
                         "k--")

        # 设置图形范围
        self.ax.set_xlim(self.x_min, self.x_max)  # 根据需要调整范围
        self.ax.set_ylim(self.y_min, self.y_max)  # 根据需要调整范围

        # 画虚线格
        x_grid=np.arange(self.x_min,self.x_max+1,(self.x_max-self.x_min)//5)
        y_grid=np.arange(self.y_min,self.y_max+1,(self.y_max-self.y_min)//5)

        for y in y_grid:
            self.ax.axhline(y,linestyle='--',color='green',alpha=0.5)

        for x in x_grid:
            self.ax.axvline(x, linestyle='--', color='green', alpha=0.5)

        ## 添加图例
        # self.ax.legend()

        ## 取消坐标轴
        # self.ax.axis('off')

        if mode == "human":
            # 显示图形
            plt.pause(0.1)  # 添加短暂的时间间隔，单位为秒
        else:
            # 将当前帧的图形添加到列表中
            self.frames.append(self.fig_to_array())

    # fig转换成array，生成gif的前置函数
    def fig_to_array(self):
        # 将当前figure转换成pixels序列
        buf = io.BytesIO()
        self.ax.figure.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        return img


if __name__ == "__main__":

    pi = np.pi
    target_info = {"0": (1, 0), "1": (1, pi/6), "2": (1, pi/3),
                   "3": (1, pi / 2),"4": (1, 2*pi / 3),"5": (1, 5*pi / 6),"6": (1, pi),"7": (1, 7*pi / 6)}

    config_info = {"random_ob":True,"state_norm":False,"plan":False,"save_dir": "benchmarks/AgentFollow","detect_radius":1,"hard_crash":False,"soft_crash":True,
                   "num_obstacles":100, "num_agents": 1,"platform_list":[3,1,5], "tau": 0.9, "max_steps": 100,
                   "env_x": 2.5,"env_y":2.5, "main_speed": 0.05,"main_speed_num":3, "speed_limit": [0.1,0.2,0.3],
                   "a_speed":[0.01,0.01,0.01],"a_direction":[pi/6,pi/6,pi/6]}

    # 循环测试代码
    env = AgentFollowEnv(target_info,config_info)

    while True:
        states = env.reset()  # 重置环境
        print(states[0].shape)
        done = False
        while not done:
            # 随机生成每个智能体的离散动作，其中第一维控制速度，有加速、减速、保持，
            # 第二维控制方向，有向左、向右、保持
            actions = np.random.randint((3), size=(env.num_agents, 2))
            states, rewards, done, _ = env.step(actions)  # 执行动作
            env.render(mode="human")  # 渲染环境状态

    print("Testing complete.")
