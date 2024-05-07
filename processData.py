import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
deltaT = 0
T = [2.636,2.620,2.636,2.636,2.636]
def processData(T, name):
    print("Processing", name ,"with data",T.__str__())
    aT = np.mean(T)
    print("The average",round(aT,3))
    tmp=0
    for i in range(5):
        tmp+=(T[i]-aT)**2
    deltaT = (tmp/25)**0.5
    print("The difference",round(deltaT,3))
processData([2.636,2.620,2.636,2.636,2.636],"Rod")
processData([2.059,2.067,2.055,2.071,2.086],"Solid Disk")
processData([0.324,0.323,0.321,0.318,0.327],"Support Disk")
processData([1.130,1.126,1.129,1.116,1.120],"Support Disk + Hollow Cylinder")
processData([2.116,2.130,2.115,2.103,2.115],"Solid Sphere")

df = pd.read_csv("data.csv")

#dữ liệu tự định nghĩa, user sẽ định nghĩa xem thế nào là chạy, đi bộ,... nhiều kiểu chạy và đi bộ khác nhau, mỗi kiểu sẽ đo ận tốc trung bình, ->predict lượng calo tiêu(kết hợp các yếu tố khác)
#vẽ biểu đồ quãng đường

