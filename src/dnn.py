from tensorflow.keras import layers, models, utils, datasets, Input, losses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow


model = models.Sequential([
    Input(shape=(1)),
    layers.Dense(20, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(1, activation='relu'),
])
model.summary()
inp = np.random.normal(size=(1,1))
out = model(inp)

t = np.linspace(0, 1)
x = np.sin(t)
y = np.cos(t)
opt = tensorflow.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    name='SGD'
)
loss = losses.MeanSquaredError()
model.compile(opt, loss)
model.fit(x, y)
x = np.array([[0.1]])
print(model(x))
