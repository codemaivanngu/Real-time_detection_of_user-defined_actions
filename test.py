# import sys
# import pandas as pd
# s="-0.916380700,4.3526587000,8.3737200000,28119731"
# print(sys.getsizeof(s))
#
# df = pd.DataFrame({'A':[1,2,3],'B':[4,5,6]})
# print(df)
# df=pd.concat([df,pd.DataFrame({'A':[1],'B':[2]})])
# print(df)
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import random
from matplotlib.animation import FuncAnimation
from collections import deque

# # Initialize empty lists to store data
# x_data = deque(maxlen=100)
# y_data = deque(maxlen=100)
#
#
# # Create a function to generate random data
# def generate_data():
#     return random.random()
#
#
# # Create a function to update the plot
# def update_plot(frame):
#     x = next(counter)
#     y = generate_data()
#
#     x_data.append(x)
#     y_data.append(y)
#
#     plt.cla()  # Clear the current plot
#     plt.plot(x_data, y_data)
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.title('Real-time Data Plot')
#
#
# # Create a counter to generate x-axis values
# counter = count()
# fig = plt.figure(figsize=(12, 6))
#
# # Create an animation
# ani = FuncAnimation(plt.gcf(), update_plot, interval=10)  # Update every 1 second
#
# plt.tight_layout()
# plt.show()

L=[1,2,3,4]
y=np.array(L[:3])