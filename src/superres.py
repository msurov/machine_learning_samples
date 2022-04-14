from re import L
from tensorflow.keras import layers, models, Input, models, optimizers, losses
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists


def prepare_dataset(impath):
    im = cv2.imread(impath, 0)
    assert im is not None
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
    return inputs, outputs


def test():
    if False and exists('data/model_superres'):
        model = models.load_model('data/model_superres')
    else:
        l = [
            Input(shape=(8, 8, 1)),
            layers.Conv2D(10, kernel_size=(3,3), activation='selu'),
            layers.Conv2D(10, kernel_size=(3,3), activation='selu'),
            layers.Flatten(),
            layers.Dense(128, activation='selu'),
            layers.Dense(4, activation='selu'),
            layers.Reshape((2, 2))
        ]
        model = models.Sequential(l)

    # learn
    if False:
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
        model.save('data/model_superres')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.show()

    if True:
        model = models.load_model('data/model_superres')
        im = cv2.imread('data/samples/flower.jpg', 0)
        # cv2.imwrite('data/inp.png', im)
        # im = cv2.pyrDown(im)
        # im2 = cv2.pyrUp(im)
        # cv2.imwrite('data/out2.png', im2)

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


if __name__ == '__main__':
    test()
