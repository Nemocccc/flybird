import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
# import gymnasium as gym

import gym_examples

class SnakeInputWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 原始状态空间

        self.observation_space=gym.spaces.Box(
            low=0,
            high=1,
            shape=(3,self.env.size,self.env.size),
            dtype=np.uint8
        )

    def observation(self, observation):
        # 获取渲染的图像并转换为灰度图
        base_image = np.zeros((self.env.size,self.env.size))
        head_image = base_image.copy()
        food_image = base_image.copy()
        body_image = base_image.copy()

        hx,hy=self.env.snake[0]
        head_image[hx,hy]=1
        for index,snake_body in enumerate(self.env.snake[1:]):
            bx,by=snake_body
            body_image[bx,by] = 1

        fx,fy=self.env.food
        food_image[fx,fy] = 1
        image=np.stack([head_image,food_image,body_image])
        image=image.astype(np.uint8)
        return image


def custom_make_env(env_id,size=10):
    def thunk():
        env = gym.make(env_id,size=size)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = SnakeInputWrapper(env)
        # env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk



if __name__ == "__main__":

    gym.register(id='Snake-v0', entry_point='gym_examples.envs.snake:SnakeEnv',max_episode_steps=300)
    env = gym.make('Snake-v0', render_mode="rgb_array")
    env = SnakeInputWrapper(env)
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
        print(obs)
        # exit(0)