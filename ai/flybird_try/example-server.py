import socket
import json
import threading


class TCPServer:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        print("服务器启动，等待客户端连接...")

    def accept_client(self):
        conn, addr = self.socket.accept()
        print(f"客户端 {addr} 已连接")
        return conn

    def close(self):
        if self.socket:
            self.socket.close()

    def run(self):
        self.start()
        while True:
            conn = self.accept_client()
            try:
                cnt=0
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message = json.loads(data.decode('utf-8'))
                    print(f"收到客户端消息: {message}")

                    # 根据消息类型回复不同的内容

                    if message['type'] == 'reset':
                        response = {
                            'type': 'RESPONSE',
                            'data': 'state'
                        }
                    elif message['type'] == 'step':
                        response = {
                            'type': 'RESPONSE',
                            'data': {'state':0}
                        }

                    else :
                        pass
                    cnt+=1
                    conn.sendall(json.dumps(response).encode('utf-8'))
            finally:
                conn.close()


if __name__ == '__main__':
    client_messages = [
        {'type': 'QUERY', 'data': '请求数据'},
        {'type': 'UPDATE', 'data': '更新数据'},
        {'type': 'UNKNOWN', 'data': '未知命令'}
    ]

    # 启动服务端
    server = TCPServer()
    server.run()
    # server_thread = threading.Thread(target=server.run)
    # server_thread.start()
    #
    # # 启动客户端
    # client = TCPClient()
    # client.communicate(client_messages)

    # 等待服务端线程结束
    # server_thread.join()
