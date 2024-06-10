import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from time import perf_counter
def getFormatedTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")

df0 = pd.read_csv("train0.csv")
df1 = pd.read_csv("train1.csv")
df2 = pd.read_csv("train2.csv")
df = pd.concat([df0, df1,df2])
shuffled_df = df.sample(frac=1,random_state=1).reset_index(drop=True)
print(shuffled_df.head())

arr = df.to_numpy()
X = arr[:,1:-1]
y = arr[:,-1].astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# history = model.fit(X_train, y_train, epochs=100)
# loss_value = history.history['loss']
###
# plt.figure(figsize=(10,6))
# plt.hist(loss_value,bins=70)
# plt.title('Histogram of Training Loss Values')
epochs = 50
do_not_plot = 10
def train_on(L:list, u = do_not_plot):
    tf.random.set_seed = 0 #fix random state
    tin = perf_counter()
    model = Sequential([
        Dense(units = c,activation = 'relu') for c in L[:-1]
    ]+[Dense(units = L[-1],activation = 'linear')])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    history = model.fit(X_train, y_train, epochs=epochs,validation_data = (X_test,y_test),verbose =0)
    def Min(u,L):
        return [min(u,c) for c in L]
    train_loss = Min(5,history.history['loss'][u:])
    test_loss = Min(5,history.history['val_loss'][u:])
    tout = perf_counter()
    print("train on ", L, "successfull in ", tout -tin,'s')

    return train_loss,test_loss


train_loss1,test_loss1 = train_on([3,5,7,5,3])
train_loss2,test_loss2 = train_on([3,5,3])
train_loss3, test_loss3 = train_on([50,25,12,25,50,3])
train_loss4,test_loss4 = train_on([50,25,12,6,3])

x = np.arange(do_not_plot,epochs)
# plt.plot(x,train_loss1,label = 'train [3,5,7,5,3]',linestyle='-',marker='.')
# plt.plot(x,train_loss2,label = 'train [3,5,3]',linestyle='-',marker='.')
# plt.plot(x,train_loss3,label = 'train [50,25,12,25,50,3]',linestyle='-',marker='.')
# plt.plot(x,train_loss4,label = 'train [50,25,12,6,3]',linestyle='-',marker='.')
plt.plot(x,train_loss1,'--',label = 'train [3,5,7,5,3]')
plt.plot(x,train_loss2,'--',label = 'train [3,5,3]')
plt.plot(x,train_loss3,'--',label = 'train [50,25,12,25,50,3]')
plt.plot(x,train_loss4,'--',label = 'train [50,25,12,6,3]')

plt.plot(x,test_loss1,label = 'test [3,5,7,5,3]')
plt.plot(x,test_loss2,label = 'test [3,5,3]')
plt.plot(x,test_loss3,label = 'test [50,25,12,25,50,3]')
plt.plot(x,test_loss4,label = 'test [50,25,12,6,3]')

plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.legend()
plt.show()



# model.save(getFormatedTime()+"model.keras")
