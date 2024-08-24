import socket
import json
import random
import gymnasium as gym
import numpy as np


class FlappyBirdClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None

        self.observation_space = gym.spaces.Box(0, 1, shape=(8,), dtype=np.float32)

        # 0 左 1上 2右 3下
        # 动作映射字典

        self.action_space = gym.spaces.Discrete(4)

        self.score=0

        self.x_max=80
        self.y_min=1
        self.y_max=26

        self.connect()


    def connect(self):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print("Connected to server")


    def send_message(self, message):
        self.socket.sendall(message.encode('utf-8'))
        response = self.socket.recv(1024).decode('utf-8')
        return json.loads(response)

    def reset(self,is_display=0):
        self.score=0
        if is_display==0:
            info=self.send_message("reset")
        else:
            info=self.send_message("reset_display")

        state=info['bird']
        obs=info['walls']

        state_obs=[]
        state_obs.append(state['x']/self.x_max)
        state_obs.append((state['y']-self.y_min)/(self.y_max-self.y_min))
        for wall in obs:
            state_obs.append(wall['x'] / self.x_max)
            state_obs.append((wall['y'] - self.y_min) / (self.y_max - self.y_min))

        done=info['game_over']

        return state_obs

    def step(self, jump=False):
        info=None
        if jump:
            info = self.send_message("jump")
        else:
            info = self.send_message("step")
        state=info['bird']
        obs=info['walls']

        state_obs=[]
        state_obs.append(state['x']/self.x_max)
        state_obs.append((state['y']-self.y_min)/(self.y_max-self.y_min))
        for wall in obs:
            state_obs.append(wall['x'] / self.x_max)
            state_obs.append((wall['y'] - self.y_min) / (self.y_max - self.y_min))


        done=info['game_over']

        reward=state['score']-self.score
        self.score=state['score']

        if self.score>=300:
            done=True

        # reward=0 if done else 1
        return state_obs,reward,done,{}

    def close(self):
        if self.socket:
            self.socket.close()

def main():
    client = FlappyBirdClient()


    try:
        # Reset the game
        done=False
        obs = client.reset(is_display=1)
        print("Initial state:", obs)

        # Play for 10 steps

        while not done:
            # Alternate between jumping and not jumping
            # state = client.step(jump=(_ % 2 == 0))
            random_number = random.choice([0, 1])
            obs,reward,done,info = client.step(jump=random_number)
            print("obs:", obs,"reward",reward,"done",done,"info",info)
            # if done:
            #     obs=client.reset()

    finally:
        client.close()

if __name__ == "__main__":
    main()