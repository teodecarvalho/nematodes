from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Conv2DTranspose, concatenate
from tensorflow.keras import Model
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, medial_axis, closing, square
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave, imread

import pandas as pd
from skan import Skeleton

plt.rcParams["figure.figsize"] = (10, 10)

import cv2

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
        img = imread(image_path)
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

    def calculate_width(self, image):
        _, distance = medial_axis(image, return_distance=True)
        max_width = max(distance[image > 0]) * 2
        return(max_width)

    def measure_nematoides(self, mask, show_image=False):
        label_im = label(mask)
        results = []
        if regionprops(label_im):
            for count, region in enumerate(regionprops(label_im)):
                skel = skeletonize(region.image)
                Skel = Skeleton(skel)
                result = {"id": count,
                          "width": self.calculate_width(region.image),
                          "npaths": Skel.n_paths,
                          "length": Skel.path_lengths()[0],
                          "fatness": float(region.area) / Skel.path_lengths()[0],
                          "centroid_x_pix": region.centroid[1],
                          "centroid_y_pix": region.centroid[0],
                          "centroid_x":region.centroid[1]/float(mask.shape[1]),
                          "centroid_y":1 - region.centroid[0]/float(mask.shape[0]),
                          "image":region.image}
                results.append(result)
        else:
            results = [{"id":None, "width":None, "npaths":None, "length":None, "fatness":None, "centroid_x":None,
                        "centroid_y":None,  "centroid_y_pix":None,  "centroid_x_pix":None, "image":None}]
        data = pd.DataFrame(results)
        #if show_image:
        #    show_nematoides(im, label_im)
        print(f"Found {len(data)} nematoides in the image!")
        if any(data.npaths > 1):
            print("Some branching were detected. The total number of nematoides might be underestimated")
        return data

    def get_coordinates(self, img_path):
        (x, y) = (img_path.split("__")[1:3])
        return (x, y)

    def main(self, img_path):
        print(img_path)
        img_or = self.read_image(img_path)
        img = self.preprocess(img_or)
        mask1 = self.predict(img, threshold=.5)
        filename = img_path.split("/")[1]
        imsave(f"predicted_masks/{filename}", mask1)
        mask2 = self.close_mask(mask1)
        mask3 = self.erase_small_region(mask2, min_area = 1550)
        mask4 = clear_border(mask3)
        # #self.show_img(img_list = [resize(img_or, (224, 224)), mask1, mask2, mask3, mask4])
        data = self.measure_nematoides(mask4)
        coordinates = self.get_coordinates(img_path)
        data[["img_x"]] = coordinates[0]
        data[["img_y"]] = coordinates[1]
        self.save_nematode_img(data)
        return data

    def save_nematode_img(self, data):
        if any(data.id):
            for i, row in data.iterrows():
                imsave(f"nematodes/nem__{row.img_x}__{row.img_y}__{row.id}.png", row.image)

    def show_img(self, img_list):
        fig, axs = plt.subplots(len(img_list))
        for i, img in enumerate(img_list):
            axs[i].imshow(img, cmap = "gray")
        plt.show()