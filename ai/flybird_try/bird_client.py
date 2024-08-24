import socket
import json
import random



class FlappyBirdClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print("Connected to server")

    def send_message(self, message):
        self.socket.sendall(message.encode('utf-8'))
        response = self.socket.recv(1024).decode('utf-8')
        return json.loads(response)

    def reset(self):
        info=self.send_message("reset")
        state=info['bird']
        obs=info['walls']
        done=info['game_over']
        return obs

    def step(self, jump=False):
        info=None
        if jump:
            info = self.send_message("jump")
        else:
            info = self.send_message("step")
        state=info['bird']
        obs=info['walls']
        done=info['game_over']

        reward=0
        if done == 0:
            reward=1
        return [state,obs],reward,done,{}

    def close(self):
        if self.socket:
            self.socket.close()

def main():
    client = FlappyBirdClient()
    client.connect()

    try:
        # Reset the game
        done=False
        obs = client.reset()
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