import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import numpy as np
import pandas as pd

from datetime import datetime
from time import perf_counter
def getFormatedTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")

model = Sequential([
    Dense(units=100, activation='relu'),
    Dense(units=100, activation='relu'),
    Dense(units=100, activation='relu'),
    Dense(units=3, activation='linear'),
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),)

df0 = pd.read_csv("train0.csv")
df1 = pd.read_csv("train1.csv")
df2 = pd.read_csv("train2.csv")
df = pd.concat([df0, df1,df2])
shuffled_df = df.sample(frac=1,random_state=1).reset_index(drop=True)
# print(shuffled_df.head())
arr = df.to_numpy()
X = arr[:,1:-1]
y = arr[:,-1]

model.fit(X, y, epochs=100)

model.save(getFormatedTime()+"model.h5")
