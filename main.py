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

conn.send(b"vailolluon")


print("first message: ")
# while True:
#     data = conn.recv(1024)
#     print("some data:",data.decode('utf-8'))
# s=conn.recv(200).decode("utf-8")
s=conn.recv(88)
while True:
    print(s:=conn.recv(88).decode("utf-8"))
    if s=='':break



    
#generate a code to generate list of 3 random number

# conn.send("troll",encode())
# s.close()

# "without s.close()"
#
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST,PORT))
#
#     s.listen(2)
#     conn, addr = s.accept() #.accept() now create a new socket object and assign it to 'conn', it's different from socket object 's' that able to continuely connect to other client
#     with conn:
#         print(f'Connected from: {addr}')
#         # while True:
#         #     data = conn.recv(50)
#         #     print("troll")
#         #     print(f"Recieved: {data}")
#         #     if not data:
#         #         break
#         #
#         #     conn.sendall(data)
#         data = conn.recv(50)
#         print("troll")
#         print(f"Recieved: {data}")
#         # if not data:
#         #     break

"Client"
# class FormattedList(list):
#     def __str__(self):
#         # Format each number in the list
#         formatted_numbers = [f"{num:.8f}" for num in self]
#         return "[" + ", ".join(formatted_numbers) + "]"
#
# import socket
# s = socket.socket()
# s.connect(('192.168.1.110',8000))
#
# msg = s.recv(1024)
#
# # while msg:
#     # print('Received',msg.decode())
#     # msg = s.recv(1024)
# print("received:",msg.decode())
#
# from decimal import Decimal, getcontext
# getcontext().prec = 8
#
# import random
# import sys
#
# # Generate a list of 3 random real numbers between 0 and 1
# while True:
#     random_numbers = [random.uniform(0, 1) for _ in range(3)]
#     random_numbers = FormattedList(random_numbers)
#     s.send(random_numbers.__str__().encode())
#
#     print("List of random real numbers:", random_numbers, ss:=sys.getsizeof(random_numbers.__str__().encode()))
#     if(ss!=69):break
#
#
# s.close()
