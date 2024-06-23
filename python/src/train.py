import numpy as np
import pickle

import cv2
from pyTsetlinMachine.tm import MultiClassTsetlinMachine, MultiClassConvolutionalTsetlinMachine2D
from pyTsetlinMachine.tools import Binarizer

import util


def train_tm():
    hog = util.create_hog_descriptor()

    binarizer = Binarizer(max_bits_per_feature=5)
    signals, labels = util.load_training_ds(signaller=hog.compute,
                                            basepath='data/96/train/')
    signals = np.matrix(signals)
    # We have to provide an ndarray as the TM implementation does not accept matrices
    shape = signals.shape
    signals = np.asarray(signals).reshape(shape)
    labels = np.array(labels)

    print('Training binarizer...')
    binarizer.fit(signals)

    # I did not find a better way to store the model, so unless we want to store the internal
    # state manually, we have to pickle the object
    print('Saving binarizer...')
    with open('binarizer.model', 'wb') as file:
        pickle.dump(binarizer, file)

    signals = binarizer.transform(signals)

    tm = MultiClassTsetlinMachine(800, 40, 5.0)

    print('Training Tsetlin machine...')
    tm.fit(signals, labels, epochs=100)

    # Again, we can either obtain the internal state and save that, or we can just pickle
    # the object
    print('Saving Tsetlin machine...')
    with open('tsetlin_machine.model', 'wb') as file:
        pickle.dump(tm, file)

    test_signals, test_labels = util.load_training_ds(signaller=hog.compute,
                                                      basepath='data/96/test/')

    test_signals = np.matrix(test_signals)
    test_shape = test_signals.shape
    test_signals = np.asarray(test_signals).reshape(test_shape)

    test_signals = binarizer.transform(test_signals)

    test_labels = np.array(test_labels)

    print('Testing Tsetlin machine...')
    pred_labels = tm.predict(test_signals)
    accuracy = 100*(pred_labels == test_labels).mean()

    print(f'{accuracy=:.4f}')


def convert_img(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 96))


def train_ctm():

    signals, labels = util.load_training_ds(signaller=convert_img,
                                            basepath='data/96/train/')

    # signals is a list of ndarrays containing each image
    # Using np.stack, we can create a 3D array of images
    signals = np.stack(signals)

    # labels should also be stored in a numpy array
    labels = np.array(labels)

    tm = MultiClassConvolutionalTsetlinMachine2D(800, 40, 5.0, (5, 5))

    print('Training Tsetlin machine...')
    for i in range(10):
        tm.fit(signals, labels, epochs=1)
        print(f'Training epoch {i} finished...')

    print('Saving Tsetlin machine...')
    with open('convolutional_tsetlin_machine.model', 'wb') as file:
        pickle.dump(tm, file)

    test_signals, test_labels = util.load_training_ds(signaller=convert_img,
                                                      basepath='data/96/test/')

    test_signals = np.stack(test_signals)
    test_labels = np.array(test_labels)

    print('Testing convolutional Tsetlin machine...')
    pred_labels = tm.predict(test_signals)
    accuracy = 100*(pred_labels == test_labels).mean()

    print(f'{accuracy=:.4f}')


if __name__ == '__main__':
    train_tm()
    print()
    train_ctm()
