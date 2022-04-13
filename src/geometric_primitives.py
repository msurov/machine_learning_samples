from tensorflow.keras import layers, models, Input, models, optimizers
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os.path import exists

'''
    The CNN classifies triangles, circles, and crosses
'''

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

def gen_triangle_sample():
    l = 8 + 0.5 * np.random.normal(size=3)
    a = 10 * np.random.normal(size=3) + np.linspace(0, 240, 3)
    a = np.deg2rad(a)
    sa = np.sin(a)
    ca = np.cos(a)
    c = np.random.normal(size=2) + 15
    pts = c + l[:,np.newaxis] * np.array([ca, sa]).T
    pts = np.round(pts).astype(int)
    p1,p2,p3 = pts
    im = np.zeros((32, 32), np.uint8)
    cv2.line(im, p1, p2, thickness=2, color=255)
    cv2.line(im, p2, p3, thickness=2, color=255)
    cv2.line(im, p3, p1, thickness=2, color=255)
    im = cv2.GaussianBlur(im, (3,3), 1)
    return im

def gen_quad_sample():
    l = 8 + 0.5 * np.random.normal(size=4)
    a = 10 * np.random.normal(size=4) + np.linspace(0, 360 * 3 / 4, 4)
    a = np.deg2rad(a)
    sa = np.sin(a)
    ca = np.cos(a)
    c = np.random.normal(size=2) + 15
    pts = c + l[:,np.newaxis] * np.array([ca, sa]).T
    pts = np.round(pts).astype(int)
    p1,p2,p3,p4 = pts
    im = np.zeros((32, 32), np.uint8)
    cv2.line(im, p1, p2, thickness=2, color=255)
    cv2.line(im, p2, p3, thickness=2, color=255)
    cv2.line(im, p3, p4, thickness=2, color=255)
    cv2.line(im, p4, p1, thickness=2, color=255)
    im = cv2.GaussianBlur(im, (3,3), 1)
    return im


def prepare_dataset(sz):
    triangles = [gen_triangle_sample() for i in range(sz//3)]
    circles = [gen_circle_sample() for i in range(sz//3)]
    crosses = [gen_cross_sample() for i in range(sz//3)]

    triangles = np.array(triangles)
    circles = np.array(circles)
    crosses = np.array(crosses)

    triangle_labels = np.zeros((len(triangles), 3))
    triangle_labels[:,0] = 1

    circle_labels = np.zeros((len(circles), 3))
    circle_labels[:,1] = 1

    cross_labels = np.zeros((len(crosses), 3))
    cross_labels[:,2] = 1

    inputs = np.concatenate((triangles, circles, crosses), axis=0)
    labels = np.concatenate((triangle_labels, circle_labels, cross_labels), axis=0)
    titles = ('triangle', 'circle', 'cross')

    return inputs, labels, titles

def create_model1():
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
        layers.Dense(3, activation='softmax'),
    ]
    model = models.Sequential(l)
    model.summary()
    return model

def test1():
    if exists('data/model1'):
        model = models.load_model('data/model1')
    else:
        model = create_model1()

    # learn the model
    if False:
        inputs, labels, titles = prepare_dataset(1000)
        test_inputs, test_labels, _ = prepare_dataset(100)

        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile('adam', loss)
        history = model.fit(inputs, labels, epochs=100, 
            validation_data=(test_inputs, test_labels), batch_size=20)
        model.save('data/model1')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    # test the model
    if True:
        print('triangle')
        samples = np.array([gen_triangle_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 0)

        print('circle')
        samples = np.array([gen_circle_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 1)

        print('cross')
        samples = np.array([gen_cross_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 2)

    # look how new figures recognized
    if True:
        print('quad')
        samples = np.array([gen_quad_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        print(indices)

    if True:
        cross = cv2.imread('data/samples/cross.png', 0)
        circle = cv2.imread('data/samples/circle.png', 0)
        triangle = cv2.imread('data/samples/triangle.png', 0)
        quad = cv2.imread('data/samples/quad.png', 0)
        figures = np.array([cross, circle, triangle, quad])

        ans = model(figures)
        print(ans)

def create_model2():
    l = [
        Input(shape=(32, 32, 1)),
        # (3x3 + 1) * 10 parameters
        layers.Conv2D(10, kernel_size=3, activation='sigmoid'),
        layers.MaxPool2D(pool_size=2, strides=2),

        # (3x3x10 + 1) * 10 parameters
        layers.Conv2D(10, kernel_size=(3,3), activation='sigmoid'),
        layers.MaxPool2D(pool_size=2, strides=2),

        layers.Flatten(),

        # fully connected layer
        layers.Dense(1024, activation='sigmoid'),
        layers.Dense(3, activation='softmax'),
    ]
    model = models.Sequential(l)
    model.summary()
    return model

def test2():
    if exists('data/model2'):
        model = models.load_model('data/model2')
    else:
        model = create_model2()

    # learn the model
    if False:
        inputs, labels, titles = prepare_dataset(5000)
        test_inputs, test_labels, _ = prepare_dataset(500)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile('adam', loss)
        history = model.fit(inputs, labels, epochs=500, 
            validation_data=(test_inputs, test_labels), batch_size=50)
        model.save('data/model2')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    # test the model
    if True:
        print('triangle')
        samples = np.array([gen_triangle_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 0)

        print('circle')
        samples = np.array([gen_circle_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 1)

        print('cross')
        samples = np.array([gen_cross_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(ans)
        assert np.allclose(indices, 2)

    # look how new figures recognized
    if True:
        print('quad')
        samples = np.array([gen_quad_sample() for i in range(10)])
        ans = model(samples)
        indices = np.argmax(ans, axis=1)
        print(indices)

    # manually drawn figures
    if True:
        cross = cv2.imread('data/samples/cross.png', 0)
        circle = cv2.imread('data/samples/circle.png', 0)
        triangle = cv2.imread('data/samples/triangle.png', 0)
        quad = cv2.imread('data/samples/quad.png', 0)
        figures = np.array([cross, circle, triangle, quad])
        ans = model(figures)
        print(ans)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # test1()
    test2()
