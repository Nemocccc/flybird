import numpy as np
from gym import ObservationWrapper
# import gymnasium as gym
import gym
import cv2
import gym_examples

class SnakeInputWrapper(ObservationWrapper):
    def __init__(self, env, image_shape=(84,84)):
        super().__init__(env)
        # 原始状态空间
        inf=np.inf
        self.image_shape=image_shape
        self.observation_space=gym.spaces.Box(
            low=0,
            high=255,
            shape=(3,84,84),
            dtype=np.uint8
        )
        self.i=0

    def observation(self, observation):
        # 获取渲染的图像并转换为灰度图
        image = self.env.render()
        # 缩放图像到指定大小
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image, self.image_shape,interpolation=cv2.INTER_AREA)
        image_reshaped = np.transpose(image_resized, (2, 0, 1))
        image_reshaped_unit8=image_reshaped.astype(np.uint8)

        # cv2.imwrite('image.jpg', image)
        # image_inter = cv2.resize(image, self.image_shape,interpolation=cv2.INTER_AREA)
        # image_nointer = cv2.resize(image, self.image_shape,)
        # cv2.imwrite('resized_image_no_inter.jpg', image_nointer)
        # cv2.imwrite(f'imgs/reshape_image{self.i}.jpg', np.transpose(image_reshaped_unit8,(1,2,0)))
        # self.i+=1

        return image_reshaped_unit8

def custom_make_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SnakeInputWrapper(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

class SnakeImgaeLikeWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 原始状态空间

        self.observation_space=gym.spaces.Box(
            low=0,
            high=255,
            shape=(3,self.env.size,self.env.size),
            dtype=np.uint8
        )

    def observation(self, observation):
        # 获取渲染的图像并转换为灰度图
        snake_image = np.zeros((self.env.size,self.env.size))
        food_image=np.zeros((self.env.size,self.env.size))

        for index,snake_body in enumerate(self.env.snake):
            bx,by=snake_body
            snake_image[bx,by] = index+1

        fx,fy=self.env.food
        food_image[fx,fy] = 1
        image=np.stack([snake_image,food_image])
        print(image.shape)
        return image


def custom_make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SnakeInputWrapper(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk



if __name__ == "__main__":

    # gym.register(id='Snake-v0', entry_point='gym_examples.envs.snake:SnakeEnv',max_episode_steps=300)
    env = gym.make('Snake-v0',render_mode="rgb_array")
    env = SnakeImgaeLikeWrapper(env)
    # envs = gym.vector.SyncVectorEnv(
    #     [custom_make_env("Snake-v0") for i in range(2)],
    # )

    done =False

    state,info=env.reset()

    while not done:
        action=env.action_space.sample()
        action=0
        obs,_,te,tr,info=env.step(action)
        done = te or tr
        print(np.array(obs).shape)
        # exit(0)
