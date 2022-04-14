from re import L
from tensorflow.keras import layers, models, Input, models, optimizers, losses
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
    for i in range(0, h - 15, 11):
        for j in range(0, w - 15, 11):
            inputs += [blurred[i:i+16,j:j+16]]
            outputs += [src[i+7:i+9,j+7:j+9]]
            # print(inputs[-1].shape, outputs[-1].shape)

    inputs = np.array(inputs, dtype=np.float32) / 255
    outputs = np.array(outputs, dtype=np.float32) / 255
    return inputs, outputs


def test():
    if False and exists('data/model_deblur'):
        model = models.load_model('data/model_deblur')
    else:
        # l = [
        #     Input(shape=(16, 16, 1)),
        #     layers.Conv2D(10, kernel_size=(3,3), activation='sigmoid'),
        #     layers.Conv2D(10, kernel_size=(3,3), activation='sigmoid'),
        #     layers.Flatten(),
        #     layers.Dense(128, activation='sigmoid'),
        #     layers.Dense(4, activation='sigmoid'),
        #     layers.Reshape((2, 2))
        # ]
        l = [
            Input(shape=(16, 16, 1)),
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

        inputs, outputs = prepare_dataset('data/samples/flower.jpg')
        n = len(inputs)
        indices = np.arange(n, dtype=int)
        np.random.shuffle(indices)
        i1 = indices[0:3*n//4]
        i2 = indices[3*n//4:]
        training_inputs = inputs[i1]
        training_outputs = outputs[i1]
        test_inputs = inputs[i2]
        test_outputs = outputs[i2]

        history = model.fit(training_inputs, training_outputs, epochs=100, 
            validation_data=(test_inputs, test_outputs))
        model.save('data/model_deblur')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    if True:
        model = models.load_model('data/model_deblur')
        src = cv2.imread('data/samples/flower.jpg', 0)
        cv2.imwrite('data/src.png', src)
        blurred = cv2.GaussianBlur(src, (5,5), 1)
        cv2.imwrite('data/blurred.png', blurred)

        h,w = blurred.shape
        samples = []
        for i in range(0, h - 15, 2):
            for j in range(0, w - 15, 2):
                sample = blurred[i:i+16,j:j+16]
                samples += [sample]

        samples = np.array(samples, dtype=np.float32) / 255
        results = model(samples)
        results = np.array(results, dtype=np.float32)

        dst = np.zeros((h, w))
        k = 0
        for i in range(0, h-15, 2):
            for j in range(0, w-15, 2):
                dst[i+7:i+9,j+7:j+9] = results[k,:,:]
                k += 1

        dst = np.clip(dst * 255, 0, 255).astype(np.uint8)
        cv2.imwrite('data/deblurred.png', dst)


if __name__ == '__main__':
    test()
