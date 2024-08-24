import socket
import json
import random
import gymnasium as gym
import numpy as np
import threading
import queue

class FlappyBirdClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.message_queue = queue.Queue(maxsize=1)
        self.receive_thread = None
        self.running = False

        self.observation_space = gym.spaces.Box(0, 1, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self.score = 0
        self.x_max = 80
        self.y_min = 1
        self.y_max = 26

        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print("Connected to server")
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_messages)
        self.receive_thread.start()

    def receive_messages(self):
        while self.running:
            try:
                response = self.socket.recv(1024).decode('utf-8')
                if response:
                    self.message_queue.put(response)
                else:
                    break  # 如果接收到空消息，可能表示连接已关闭
            except socket.error:
                print("the socket.error make close")
                break  # 当 socket 被关闭时，退出循环
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
        print("Receive thread ended")

    def send_message(self, message):
        self.socket.sendall(message.encode('utf-8'))
        try:
            response = self.message_queue.get(timeout=5)  # 5秒超时
            return json.loads(response)
        except queue.Empty:
            print("Timeout waiting for response")
            return None

    def reset(self, is_display=0):
        self.score = 0
        if is_display == 0:
            info = self.send_message("reset")
        else:
            info = self.send_message("reset_display")

        state = info['bird']
        obs = info['walls']

        state_obs = []
        state_obs.append(state['x'] / self.x_max)
        state_obs.append((state['y'] - self.y_min) / (self.y_max - self.y_min))
        for wall in obs:
            state_obs.append(wall['x'] / self.x_max)
            state_obs.append((wall['y'] - self.y_min) / (self.y_max - self.y_min))

        return state_obs

    def step(self, jump=False):
        info = None
        if jump:
            info = self.send_message("jump")
        else:
            info = self.send_message("step")
        state = info['bird']
        obs = info['walls']

        state_obs = []
        state_obs.append(state['x'] / self.x_max)
        state_obs.append((state['y'] - self.y_min) / (self.y_max - self.y_min))
        for wall in obs:
            state_obs.append(wall['x'] / self.x_max)
            state_obs.append((wall['y'] - self.y_min) / (self.y_max - self.y_min))

        done = info['game_over']

        reward = state['score'] - self.score
        self.score = state['score']

        if self.score >= 300:
            done = True

        return state_obs, reward, done, {}

    def close(self):
        self.running = False
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
            except socket.error:
                pass  # 忽略关闭时的错误
            self.socket.close()
        if self.receive_thread:
            self.receive_thread.join(timeout=5)  # 等待线程最多5秒
        print("Client closed successfully")

def main():
    client = FlappyBirdClient()
    try:
        done = False
        obs = client.reset(is_display=1)
        print("Initial state:", obs)

        while not done:
            random_number = random.choice([0, 1])
            obs, reward, done, info = client.step(jump=random_number)
            print("obs:", obs, "reward", reward, "done", done, "info", info)
    finally:
        client.close()
    print("Main function completed")

if __name__ == "__main__":
    main()
    import os
    os._exit(0)  # 强制终止所有线程