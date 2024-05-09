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
# conn.send("Connected")
print("Connected from", addr)

#sever sử dụng kết nối gửi dữ liệu tới client dưới dạng binary
import sys
print("first message: ")

s=conn.recv(1000)#96

import visualization
import recordData
import numpy as np
import pickle
from datetime import datetime
from time import perf_counter
def getFormatedTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")
# file_path = "dataframe_"+formatted_time+".pkl"
def isInt(n):
    while(n[-1]==" "):n=n[:-1]
    return n.isdigit()

batchSize=61*4


L=[]
firstProcess=1
initialTime=0

def func1(df: pd.DataFrame,newData:list):
    ndf = pd.DataFrame([L],columns=df.columns)
    # print(ndf)
    df = pd.concat([df,ndf],axis=0,ignore_index=True)
    return df

def func2(df:pd.DataFrame,fileName):
    with open(fileName,'wb') as f:
        pickle.dump(df,f)

currentPhoneTime=0
def process():
    global firstProcess,initialTime,currentPhoneTime
    if(L[-1]<=currentPhoneTime):
        return
    currentPhoneTime= L[-1]

    if firstProcess==1:
        initialTime=int(L[-1])
        firstProcess=0
    # print(L)
    visualization.x_data.append(int(L[-1]-initialTime))
    # print(visualization.x_data)

    y = np.array(L[:3])
    # print(y)
    visualization.y_data.append(np.sum(y * y))

    y2 = np.abs(np.array(L[:3]))
    y2p = np.array(np.concatenate([y2, y2], axis=0)[1:4])
    # print(y2,y2p)
    visualization.y_data2.append(np.sum(y2 * y2p))

    y3p = np.array(np.concatenate([y2, y2], axis=0)[2:5])
    # print(y2, y3p)
    visualization.y_data3.append(np.sum(np.abs(y2)))

    visualization.update_plot()
SS=""
df = pd.DataFrame(columns=['x','y','z','time'])
firstTimeRecord=1
startTimeRecord=0
maxTimeRecord=30

while True:
    s = conn.recv(batchSize).decode("utf-8")
    # print("s:",s)
    SS+=s
    if len(SS)>80:
        s=SS[:80]
        SS=SS[80:]
        # print(s)
        L=list(map(float,s.split()))
        L[0]-=0.26
        L[1]-=0.05
        L[2]-=9.739
        # print(L)
        process()

        #do not use data of first 5s
        if firstTimeRecord==1:
            firstTimeRecord=0
            startTimeRecord=perf_counter()
            print("c1")
        elif perf_counter()-startTimeRecord<maxTimeRecord :
            if perf_counter()-startTimeRecord>5:

                df= func1(df,L)
                print("c2")
        else:
            func2(df, getFormatedTime() + ".pkl")
            exit(0)
        print(perf_counter()-startTimeRecord)
    #break by irregular received data
    if len(s)==0:break
visualization.plt.show()

