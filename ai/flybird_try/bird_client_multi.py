import multiprocessing
from bird_client import FlappyBirdClient
import random

def run_game_instance(instance_id):
    client = FlappyBirdClient()
    client.connect()

    try:
        done = False
        obs = client.reset()
        print(f"Initial state for instance {instance_id}:", obs)

        while not done:
            random_number = random.choice([0, 1])
            obs, reward, done, info = client.step(jump=random_number)
            print(f"Instance {instance_id} - obs:", obs, "reward:", reward, "done:", done)
            # if done:
            #     obs = client.reset()

    finally:
        client.close()

if __name__ == "__main__":
    num_instances = 3  # 你想要启动的实例数量
    processes = []

    for i in range(num_instances):
        process = multiprocessing.Process(target=run_game_instance, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()