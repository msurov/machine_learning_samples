from re import L
from tensorflow.keras import layers, models, Input, models, optimizers, losses
from tensorflow.nn.experimental import stateless_dropout
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists


def prepare_dataset(impath):
    src = cv2.imread(impath, 0)
    assert src is not None
    h,w = src.shape
    blurred = cv2.GaussianBlur(src, (5,5), 1)

    inputs = []
    outputs = []
    for i in range(0, h - 15, 3):
        for j in range(0, w - 15, 3):
            inputs += [blurred[i:i+16,j:j+16]]
            outputs += [src[i+7:i+9,j+7:j+9]]
            # print(inputs[-1].shape, outputs[-1].shape)

    inputs = np.array(inputs, dtype=np.float32) / 255
    outputs = np.array(outputs, dtype=np.float32) / 255
    return inputs, outputs

def deblur_1ch(model, src):
    h,w = src.shape
    samples = []
    for i in range(0, h - 15, 2):
        for j in range(0, w - 15, 2):
            sample = src[i:i+16,j:j+16]
            samples += [sample]

    samples = np.array(samples, dtype=np.float32) / 255
    results = model(samples)
    results = np.array(results, dtype=np.float32)

    dst = np.zeros((h, w), dtype=np.float32)
    k = 0
    for i in range(0, h-15, 2):
        for j in range(0, w-15, 2):
            dst[i+7:i+9,j+7:j+9] = results[k,:,:]
            k += 1

    dst = np.clip(dst * 255, 0, 255).astype(np.uint8)
    return dst

def deblur(model, im):
    h,w,c = im.shape
    dst = np.zeros(im.shape, dtype=np.float32)
    for i in range(c):
        dst[:,:,i] = deblur_1ch(model, im[:,:,i])
    return dst

def test():
    if False and exists('data/model_deblur'):
        model = models.load_model('data/model_deblur')
    else:
        l = [
            Input(shape=(16, 16, 1)),
            layers.Conv2D(8, kernel_size=(3,3), activation='sigmoid'),
            layers.Conv2D(8, kernel_size=(3,3), activation='sigmoid'),
            layers.Flatten(),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(4, activation='sigmoid'),
            layers.Reshape((2, 2))
        ]
        model = models.Sequential(l)

    # learn
    if True:
        loss = losses.MeanSquaredError()
        model.compile('adam', loss)

        inputs, outputs = prepare_dataset('data/samples_deblur/flower.jpg')
        n = len(inputs)
        indices = np.arange(n, dtype=int)
        np.random.shuffle(indices)

        i1 = indices[0:n//5]
        i2 = indices[n//5:n//5 + 100]
        training_inputs = inputs[i1]
        training_outputs = outputs[i1]
        test_inputs = inputs[i2]
        test_outputs = outputs[i2]

        history = model.fit(training_inputs, training_outputs, epochs=300, 
            validation_data=(test_inputs, test_outputs))
        model.save('data/model_deblur')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    if True:
        model = models.load_model('data/model_deblur')
        src = cv2.imread('data/samples_deblur/flower-cr.jpg')
        blurred = cv2.pyrUp(src)
        cv2.imwrite('data/samples_deblur/blurred.jpg', blurred)
        dst = deblur(model, blurred)
        cv2.imwrite('data/samples_deblur/deblurred.jpg', dst)


if __name__ == '__main__':
    test()
