from re import L
from tensorflow.keras import layers, models, Input, models, optimizers, losses
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists


im = cv2.imread('data/samples/lenna.png', 0)
h,w = im.shape

inputs = []
outputs = []
for i in range(0, h - 15, 11):
    for j in range(0, w - 15, 11):
        sample = im[i:i+16,j:j+16]
        outputs += [sample[7:-7,7:-7]]
        inputs += [cv2.pyrDown(sample)]
        # print(inputs[-1].shape, outputs[-1].shape)

inputs = np.array(inputs, dtype=np.float32) / 255
outputs = np.array(outputs, dtype=np.float32) / 255

if False:
    l = [
        Input(shape=(8, 8, 1)),
        layers.Conv2D(10, kernel_size=(3,3), activation='sigmoid'),
        layers.Conv2D(10, kernel_size=(3,3), activation='sigmoid'),
        layers.Flatten(),
        layers.Dense(128, activation='sigmoid'),
        layers.Dense(4, activation='sigmoid'),
        layers.Reshape((2, 2))
    ]
    # model = models.Sequential(l)
    model = models.load_model('data/model_superres')
    loss = losses.MeanSquaredError()
    model.compile('adam', loss)
    history = model.fit(inputs, outputs, epochs=100)
    model.save('data/model_superres')
    plt.plot(history.history['loss'])
    plt.show()

if True:
    model = models.load_model('data/model_superres')
    im = cv2.imread('data/samples/roberts.jpg', 0)
    cv2.imwrite('data/inp.png', im)
    im = cv2.pyrDown(im)
    im2 = cv2.pyrUp(im)
    cv2.imwrite('data/out2.png', im2)

    h,w = im.shape
    samples = []
    for i in range(0, h - 7, 1):
        for j in range(0, w - 7, 1):
            sample = im[i:i+8,j:j+8]
            samples += [sample]

    samples = np.array(samples, dtype=np.float32) / 255
    results = model(samples)
    results = np.array(results, dtype=np.float32)

    dst = np.zeros((h * 2, w * 2))
    k = 0
    for i in range(h-7):
        for j in range(w-7):
            dst[2*i+7:2*i+9,2*j+7:2*j+9] = results[k,:,:]
            k += 1

    dst = np.clip(dst * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('data/out.png', dst)
