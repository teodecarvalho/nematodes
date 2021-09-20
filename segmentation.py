from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Conv2DTranspose, concatenate
from tensorflow.keras import Model
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, medial_axis, closing, square
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread
import os
from glob import glob

import pandas as pd
from skan import Skeleton

from stitching import Image_Matrix, Stitcher

plt.rcParams["figure.figsize"] = (10, 10)

class Model_Nematoides():
    def __init__(self):
        self.load_model(model_path = "model_for_nematodes.h5")

    def load_model(self, model_path):
        kw = dict(activation='relu', kernel_initializer='he_normal', padding='same')

        IMG_HEIGHT = 480
        IMG_WIDTH = 480
        IMG_CHANNELS = 3

        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        # s = Lambda(lambda x: x / 255)(inputs)

        # Contraction path
        c1 = Conv2D(32, (3, 3), **kw)(inputs)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(32, (3, 3), **kw)(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(64, (3, 3), **kw)(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(64, (3, 3), **kw)(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(128, (3, 3), **kw)(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(128, (3, 3), **kw)(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(256, (3, 3), **kw)(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(256, (3, 3), **kw)(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(512, (3, 3), **kw)(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(512, (3, 3), **kw)(c5)

        kw_conv2transp = dict(strides=(2, 2), padding='same')
        u6 = Conv2DTranspose(256, (2, 2), **kw_conv2transp)(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), **kw)(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3, 3), **kw)(c6)

        u7 = Conv2DTranspose(64, (2, 2), **kw_conv2transp)(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), **kw)(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(128, (3, 3), **kw)(c7)

        u8 = Conv2DTranspose(128, (2, 2), **kw_conv2transp)(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), **kw)(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(64, (3, 3), **kw)(c8)

        u9 = Conv2DTranspose(64, (2, 2), **kw_conv2transp)(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(64, (3, 3), **kw)(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(64, (3, 3), **kw)(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.model.load_weights(model_path)

    def read_image(self, image_path):
        img = imread(image_path)[:,:,0:3]
        return img

    def preprocess(self, img):
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img, threshold = .5):
        def resize_mask(mask):
            return mask[0, :, :, 0]
        mask = self.model.predict(img)
        mask[mask < threshold] = 0
        mask[mask >= threshold] = 1
        return resize_mask(mask)

    def close_mask(self, mask, square_size = 2):
        mask = closing(mask, square(square_size))
        return mask

    def erase_small_region(self, image, min_area = 160):
        labels = label(image)
        for region in regionprops(labels):
            if region.area < min_area:
                for coord in region.coords:
                    image[coord[0], coord[1]] = 0
        return image

    def segment(self, img_path):
        print(img_path)
        img_or = self.read_image(img_path)
        img = self.preprocess(img_or)
        mask1 = self.predict(img, threshold=.5)
        mask2 = self.close_mask(mask1)
        mask3 = self.erase_small_region(mask2, min_area = 1000)
        return mask3

    def detect_nematodes(self, img_path):
        mask = self.segment(img_path)
        area = mask.sum()
        if(area > 0):
            print("Yes")
        return area > 0

    def make_matrix(self, img_dir):
        xs = [int(f.split("__")[1]) for f in os.listdir(img_dir) if ".png" in f]
        ys = [int(f.split("__")[2]) for f in os.listdir(img_dir) if ".png" in f]
        return(np.zeros(shape=(max(ys) + 1, max(xs) + 1)))

    def populate_matrix(self, matrix, found_nematodes_list):
        xs = [int(f.split("__")[1]) for f in found_nematodes_list]
        ys = [int(f.split("__")[2]) for f in found_nematodes_list]
        for x, y in zip(xs, ys):
            matrix[y, x] = 1
        return matrix

if __name__ == "__main__":
    model_nematodes = Model_Nematoides()
    with open("found_nematodes.txt", "w") as handle:
        for i, img in enumerate(glob("imagens_col/*__.png")):
            print(i)
            detected = model_nematodes.detect_nematodes(img)
            if detected:
                handle.write(img + "\n")
    mat = model_nematodes.make_matrix("./imagens_col")
    with open("found_nematodes.txt", "r") as handle:
        found_nematodes = handle.readlines()
    mat = model_nematodes.populate_matrix(mat, found_nematodes)
    image_matrix = Image_Matrix(mat)
    regions = image_matrix.get_regions()
    stitcher = Stitcher(regions)
    for i, region in enumerate(regions.values()):
        stitcher.get_images("./imagens_col", i)
        stitcher.stitch_region(i)