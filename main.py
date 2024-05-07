"Server"
import socket
import pandas as pd
#định nghĩa host và port mà sever sẽ chạy và lắng nghe
HOST = '0.0.0.0' #host ='localhost' will become '127.0.0.# 1' if IPv4 or '::1'if IPv6
PORT = 8000 #port = 4000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST,PORT))

s.listen() #chỉ chấp nhận 1 kết nối
print("Sever listening on port", PORT)

conn, addr = s.accept()

print("Connected from", addr)

#sever sử dụng kết nối gửi dữ liệu tới client dưới dạng binary
import sys
print("first message: ")

s=conn.recv(88)#96

import visualization
import numpy as np

while True:
    print(s:=conn.recv(47).decode("utf-8"),":"+str(sys.getsizeof(s))+"\n")
    L=list(map(float,s.split(",")))
    visualization.x_data.append(L[-1])
    # y=np.array(visualization.y_data[0,1,2])
    # y2=np.array(visualization.y_data2[0,1,2])
    # y2p = np.array(np.concatenate([y2,y2])[1,2,3])
    # y_data is deque not list
    y=np.array(L[:3])
    print(y)
    visualization.y_data.append(np.sum(y*y))


    y2=np.abs(np.array(L[:3]))
    y2p = np.array(np.concatenate([y2,y2],axis=0)[1:4])
    # print(y2,y2p)
    visualization.y_data2.append(np.sum(y2*y2p))

    y3p = np.array(np.concatenate([y2, y2], axis=0)[2:5])
    # print(y2, y3p)
    visualization.y_data3.append(np.sum(y2 * y3p))


    visualization.update_plot()
    if sys.getsizeof(s)<70:break
visualization.plt.show()
