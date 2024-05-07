import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

model = Sequential([
    Dense(units=5000, activation='relu'),
    Dense(units=500, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=10, activation='linear'),
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),)

