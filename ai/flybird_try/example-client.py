from queue import Queue
import socket
import json
import threading

class TCPClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.message_queue = Queue()  # 消息队列
        self.result_queue = Queue()   # 结果队列
        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print("客户端已连接到服务器")

    def send_message(self, message):
        self.socket.sendall(json.dumps(message).encode('utf-8'))

    def receive_message(self):
        response = json.loads(self.socket.recv(1024).decode('utf-8'))
        # self.message_queue.put(response)  # 将接收到的消息放入消息队列
        return response

    def close(self):
        self.socket.close()


    def reset(self):
        message={'type': 'reset', 'data': '请求数据1'}
        self.send_message(message)
        # print(f"发送消息: {message}")
        response=self.receive_message()
        return response


    def setp(self,action):
        message={'type': 'step', 'data':'请求数据1'}
        self.send_message(message)
        # print(f"发送消息: {message}")
        response=self.receive_message()
        return response


    def process_queues(self):
        # 从消息队列中连续取出三条消息，放入结果队列
        messages = [self.message_queue.get() for _ in range(3) if not self.message_queue.empty()]
        if len(messages) == 3:
            combined_result = {'combined_messages': messages}
            self.result_queue.put(combined_result)

    def get_result(self):
        # 从结果队列中取出消息
        if not self.result_queue.empty():
            return self.result_queue.get()

# 使用示例
if __name__ == '__main__':
    # client = TCPClient()

    # 启动通信线程
    # communication_thread = threading.Thread(target=client.communicate, args=(client_messages,))
    # communication_thread.start()

    # 等待通信线程完成
    # communication_thread.join()

    env=TCPClient()

    data=env.reset()
    print(data)
    for i in range(10):
        data=env.setp(action=i)
        print(data)

    # 获取结果队列中的消息
    # result = client.get_result()
    # if result:
    #     print("结果队列中的消息:", result)