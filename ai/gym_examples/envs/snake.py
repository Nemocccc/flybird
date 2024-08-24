import gymnasium as gym
import numpy as np
import random
import pygame
import sys
import time

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        super(SnakeEnv, self).__init__()

        self.size = size  # 地图大小

        self.max_length=500
        self.length=0

        # render渲染相关定义
        self.color_bg = (255, 255, 255)
        self.color_head = (0, 120, 120)
        self.color_body = (0, 255, 0)
        self.color_food = (255, 0, 0)

        self.cell_size = 20 # 窗口每格大小
        self.window_size = (self.size * self.cell_size,self.size * self.cell_size) # 窗口大小

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # 定义贪吃蛇环境的观测空间和行动空间
        size=np.float32(self.size)
        low = np.array([-size,-size, 0.0, 0.0, 0.0, 0.0])  # 连续状态空间的最小值
        high = np.array([size, size, 1.0, 1.0, 1.0, 1.0])  # 连续状态空间的最大值
        self.observation_space = gym.spaces.Box(low, high, shape=(6,), dtype=np.float32)

        # 0 左 1上 2右 3下
        # 动作映射字典
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.action_space = gym.spaces.Discrete(4)


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # 获取动作mask
    def _get_mask(self):
        action_mask = [True] * 4
        x, y = self.snake[0]
        for i, (gx, gy) in enumerate([(0, 1), (1, 0), (0, -1), (-1, 0)]):
            dx, dy = x + gx, y + gy
            if dx < 0 or dy < 0 or dx >= self.width or dy >= self.height or (
                    dx, dy) in self.snake:
                action_mask[i] = False
            else:
                action_mask[i] = True  # True则表示动作可以执行
        return action_mask

    # 状态获取函数
    def _get_obs(self):
        state = []

        head=np.array(self.snake[0])
        food=np.array(self.food)
        head2food=food-head
        state=state+head2food.tolist()

        for action in range(4):
            next_node=head+self._action_to_direction[action]
            # 检查数组中是否有元素大于self.size
            greater_than_size = next_node > self.size

            # 检查数组中是否有元素小于0
            less_than_zero = next_node < 0.0

            # 检查是否有任何元素违反了条件
            has_violations = np.any(greater_than_size) or np.any(less_than_zero)

            if has_violations or next_node.tolist() in self.snake:
                state.append(0.0)
                continue
            else:
                state.append(1.0)  # 四个方向可以走
        return np.array(state, dtype=np.float32)

    def _get_info(self):
        return {"snake_length": len(self.snake)}


    # 食物产生函数
    def _generate_food(self):
        self.food=np.random.randint(low=0,high=self.size,size=(2,)).tolist()
        while self.food in self.snake:
            self.food=np.random.randint(low=0,high=self.size,size=(2,)).tolist()
        return self.food

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.length=0
        self.snake = [np.random.randint(low=0,high=self.size,size=(2,)).tolist()]
        self.food = self._generate_food()
        self.update_grid()

        observation=self._get_obs()
        info=self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation,info


    # 更新当前动画状态
    def update_grid(self):
        self.grid = np.zeros((self.size,self.size))
        snake_head = self.snake[0]
        hx,hy=snake_head
        self.grid[hx,hy] = 1
        for snake_body in self.snake[1:]:
            bx,by=snake_body
            self.grid[bx,by] = 2
        fx,fy=self.food
        self.grid[fx,fy] = 3


    def step(self, action):

        # 结束判断
        terminated=False # 到达终止状态
        truncated=False # 对局失败

        head = self.snake[0]
        direction = self._action_to_direction[action]

        next_node=head+direction

        # 检查数组中是否有元素大于self.size
        greater_than_size = next_node >= self.size

        # 检查数组中是否有元素小于0
        less_than_zero = next_node < 0

        # 检查是否有任何元素违反了条件
        has_violations = np.any(greater_than_size) or np.any(less_than_zero)

        next_node=next_node.tolist()

        self.length+=1

        if self.length>self.max_length:
            terminated=True
            reward=-1
        elif has_violations or next_node in self.snake:
            reward = 0
            truncated=True
        elif next_node == self.food:
            reward = 1
            self.snake.insert(0,next_node)
            self.food = self._generate_food()
            self.update_grid()
        else:
            reward = 0
            self.snake.insert(0,next_node)
            self.snake.pop()
            self.update_grid()


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated,truncated, info

    def render(self, mode='human'):

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.display.set_caption("Snake Game")

        # if mode == 'rgb_array':
        #     surface = pygame.Surface(
        #         (self.width * self.cell_size, self.height * self.cell_size))
        #     self.window = surface

        canvas = pygame.Surface(self.window_size)

        canvas.fill(self.color_bg)


        for x in range(self.size):
            for y in range(self.size):
                cell_value = self.grid[x,y]
                cell_rect = pygame.Rect(x * self.cell_size, y * self.cell_size,self.cell_size, self.cell_size)
                if cell_value == 0:  # 白色的空白格子
                    pygame.draw.rect(canvas, (255, 255, 0), cell_rect, 1)
                elif cell_value == 1:  # 贪吃蛇身体
                    pygame.draw.rect(canvas, self.color_head, cell_rect)
                elif cell_value == 2:  # 贪吃蛇身体
                    pygame.draw.rect(canvas, self.color_body, cell_rect)
                elif cell_value == 3:  # 食物
                    # pygame.draw.rect(self.window, self.color_food, cell_rect)
                    pygame.draw.circle(canvas, self.color_food,
                                       (cell_rect.x + self.cell_size // 2, cell_rect.y + self.cell_size // 2),
                                       self.cell_size // 2)

        # snake_length_text = self.font.render("Length: " + str(len(self.snake)),
        #                                      True, (0, 25, 25))
        # self.window.blit(snake_length_text, (0, 0))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        # pygame.display.flip()

        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":


    gym.register(id='Snake-v0', entry_point='snake:SnakeEnv',max_episode_steps=300)
    env = gym.make('Snake-v0',render_mode="human")

    state, info = env.reset()

    for i in range(50):
        action = 2 #env.action_space.sample()
        obs, _, terminated,truncated, info = env.step(action)
        # exit(0)

    env.close()