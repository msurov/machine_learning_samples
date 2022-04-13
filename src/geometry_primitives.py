from genericpath import sameopenfile
from matplotlib.colors import same_color
from tensorflow.keras import layers, models, utils, datasets, Input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def gen_circle_sample():
    im = np.zeros((32, 32), np.uint8)
    p = 2 * np.random.normal(size=2) + 16
    r = 5 + 0.5 * np.random.normal(size=1)
    r = int(r)
    p = np.array(p, int)
    cv2.circle(im, p, r, 255, 2)
    im = cv2.GaussianBlur(im, (3,3), 1)
    return im

def gen_cross_sample():
    im = np.zeros((32, 32), np.uint8)
    p = 2 * np.random.normal(size=2) + 16

    l1 = 5 + 0.5 * np.random.normal()
    l2 = 5 + 0.5 * np.random.normal()
    alpha = np.deg2rad(45 + 10 * np.random.normal())
    beta = np.deg2rad(135 + 10 * np.random.normal())

    p1 = p + l1 * np.array([np.cos(alpha), -np.sin(alpha)])
    p2 = p - l1 * np.array([np.cos(alpha), -np.sin(alpha)])
    p3 = p + l2 * np.array([np.sin(alpha), np.cos(alpha)])
    p4 = p - l2 * np.array([np.sin(alpha), np.cos(alpha)])
    p1 = np.array(p1, int)
    p2 = np.array(p2, int)
    p3 = np.array(p3, int)
    p4 = np.array(p4, int)

    cv2.line(im, p1, p2, thickness=2, color=255)
    cv2.line(im, p3, p4, thickness=2, color=255)
    im = cv2.GaussianBlur(im, (3,3), 1)
    return im

def gen_plus_sample():
    im = np.zeros((32, 32), np.uint8)
    p = 2 * np.random.normal(size=2) + 16

    l1 = 5 + 0.5 * np.random.normal()
    l2 = 5 + 0.5 * np.random.normal()
    alpha = np.deg2rad(0 + 10 * np.random.normal())
    beta = np.deg2rad(90 + 10 * np.random.normal())

    p1 = p + l1 * np.array([np.cos(alpha), -np.sin(alpha)])
    p2 = p - l1 * np.array([np.cos(alpha), -np.sin(alpha)])
    p3 = p + l2 * np.array([np.sin(alpha), np.cos(alpha)])
    p4 = p - l2 * np.array([np.sin(alpha), np.cos(alpha)])
    p1 = np.array(p1, int)
    p2 = np.array(p2, int)
    p3 = np.array(p3, int)
    p4 = np.array(p4, int)

    cv2.line(im, p1, p2, thickness=2, color=255)
    cv2.line(im, p3, p4, thickness=2, color=255)
    im = cv2.GaussianBlur(im, (3,3), 1)
    return im


l = [
    Input(shape=(32, 32, 1)),

    # (3x3 + 1) * 10 parameters
    layers.Conv2D(10, kernel_size=3, activation='relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    # (3x3x10 + 1) * 10 parameters
    layers.Conv2D(10, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=2, strides=2),

    layers.Flatten(),

    # fully connected layer
    layers.Dense(1024, activation='relu'),
    layers.Dense(2, activation='softmax'),
]

model = models.Sequential(l)
model.summary()

lines = [gen_cross_sample() for i in range(100)]
dots = [gen_circle_sample() for i in range(100)]
lines = np.array(lines)
dots = np.array(dots)
line_labels = np.zeros((len(lines), 2))
line_labels[:,0] = 1
dot_labels = np.zeros((len(lines), 2))
dot_labels[:,1] = 1

# loss = tf.keras.losses.MeanSquaredError()
loss = tf.keras.losses.BinaryCrossentropy()
model.compile('adam', loss)
inputs = np.concatenate((lines, dots), axis=0)
labels = np.concatenate((line_labels, dot_labels), axis=0)

model.fit(inputs, labels, epochs=100)

lines = np.array([gen_plus_sample() for i in range(10)])
print(model(lines))

lines = np.array([gen_circle_sample() for i in range(10)])
print(model(lines))
