import random
import os

import scipy as sp
import cv2 as cv

import util


def img_96(img):
    img = cv.imread(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.resize(img, (96, 96))


def rotate(img):
    avg = 0
    height, width = img.shape[0], img.shape[1]
    for x in range(width):
        for y in range(height):
            avg += img[y][x]

    avg /= width*height

    angle = random.randint(-40, 40)
    return sp.ndimage.rotate(img, angle, reshape=False, cval=avg)


def data_enhancement(img):
    rot = rotate(img)
    return [img, rot]


def resnet_img(img):
    img = cv.imread(img)
    return cv.resize(img, (224, 224))


def save_imgs(imgs, parent, imtype):
    parent = parent.strip('/')
    base_folder = f'{parent}/{imtype}'

    os.makedirs(base_folder, exist_ok=True)

    for idx, img in enumerate(imgs):
        path = f'{base_folder}/{idx}.png'
        cv.imwrite(path, img)


def split(ds) -> tuple[list, list]:
    train, test = [], []

    for i in ds:
        dataset = test if random.random() < 0.2 else train
        dataset.append(i)

    return train, test


def create_96(cars, bikes):
    cars = [img_96(car) for car in cars]
    bikes = [img_96(bike) for bike in bikes]

    cars_train, cars_test = split(cars)
    bikes_train, bikes_test = split(bikes)

    save_imgs(cars_train, 'data/96/train/', 'cars')
    save_imgs(cars_test, 'data/96/test/', 'cars')
    save_imgs(bikes_train, 'data/96/train/', 'bikes')
    save_imgs(bikes_test, 'data/96/test/', 'bikes')


def create_resnet(cars, bikes):
    cars = [resnet_img(car) for car in cars]
    bikes = [resnet_img(bike) for bike in bikes]

    cars_train, cars_test = split(cars)
    bikes_train, bikes_test = split(bikes)

    save_imgs(cars_train, 'data/resnet/train/', 'cars')
    save_imgs(cars_test, 'data/resnet/test/', 'cars')
    save_imgs(bikes_train, 'data/resnet/train/', 'bikes')
    save_imgs(bikes_test, 'data/resnet/test/', 'bikes')


def create_enhanced(cars, bikes):
    cars = [img_96(car) for car in cars]
    bikes = [img_96(bike) for bike in bikes]

    cars_train, cars_test = split(cars)
    bikes_train, bikes_test = split(bikes)

    c = []
    b = []

    for car in cars_train:
        c += data_enhancement(car)

    for bike in bikes_train:
        b += data_enhancement(bike)

    save_imgs(c, 'data/enhanced/train/', 'cars')
    save_imgs(b, 'data/enhanced/train/', 'bikes')

    save_imgs(cars_test, 'data/enhanced/test/', 'cars')
    save_imgs(bikes_test, 'data/enhanced/test/', 'bikes')


def main():
    data = util.load_ds('data/Car-Bike-Dataset/', cars='Car', bikes='Bike')

    cars = [f for f, t in data if t == 1]
    bikes = [f for f, t in data if t == 0]

    create_96(cars, bikes)
    # create_enhanced(cars, bikes)
    # create_resnet(cars, bikes)


if __name__ == '__main__':
    main()
