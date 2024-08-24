import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

import cv2
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
class VectorImageInputWrapper(ObservationWrapper):
    def __init__(self, env, image_shape=(84,84)):
        super().__init__(env)
        # 原始状态空间
        inf=np.inf
        self.image_shape=image_shape
        self.image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=image_shape, # 加上颜色通道维度
            dtype=np.uint8,
        )
        self.vector_space = gym.spaces.Box(
            low=-inf,
            high=inf,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space=gym.spaces.Tuple((self.image_space,self.vector_space))

    def observation(self, observation):
        # 获取渲染的图像并转换为灰度图
        car_v = observation[1]
        pol_v = observation[3]
        vector = np.array([car_v, pol_v],dtype=np.float32)
        image = self.env.render()
        # 转换到灰度图像
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 缩放图像到指定大小
        image_resized = cv2.resize(image_gray, self.image_shape, interpolation=cv2.INTER_AREA)
        # 组合原始状态和图像数据
        state=[vector,image_resized]
        # print(image_resized.flatten(),vector)
        return state


class CartPoleInputWrapper(ObservationWrapper):
    def __init__(self, env, image_shape=(84,84)):
        super().__init__(env)
        # 原始状态空间
        inf=np.inf
        self.image_shape=image_shape
        low=np.array([0]*np.prod(image_shape)+[-inf]*2)
        high=np.array([0]*np.prod(image_shape)+[inf]*2)

        self.observation_space=gym.spaces.Box(
            low=low,
            high=high,
        )

    def observation(self, observation):
        # 获取渲染的图像并转换为灰度图
        obs=observation
        car_v = obs[1]
        pol_v = obs[3]
        vector = np.array([car_v, pol_v],dtype=np.float32)
        image = self.env.render()
        # 转换到灰度图像
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 缩放图像到指定大小
        image_resized = cv2.resize(image_gray, self.image_shape, interpolation=cv2.INTER_AREA)
        # 组合原始状态和图像数据
        state=np.concatenate([image_resized.flatten(),vector])
        return state


def custom_make_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CartPoleInputWrapper(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk




