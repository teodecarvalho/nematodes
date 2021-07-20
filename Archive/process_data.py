from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import os
import pandas as pd
from segmentation import Model_Nematoides
from glob import glob

class Nematode_df():
    def __init__(self, IMG_SIZE_MM = 1.25, STEP_MM = 1):
        self.model = Model_Nematoides()
        self.IMG_SIZE_MM = IMG_SIZE_MM
        self.STEP_MM = STEP_MM
        self.get_nematode_data()

    def calculate_absolute_x(self, img_coord, centroid_coord, img_size_mm, step_mm):
        nematode_coord = img_coord * step_mm + centroid_coord * img_size_mm
        return(nematode_coord)

    def get_nematode_data(self):
        results = []
        for i, img in enumerate(glob("imagens_col/*__.png")):
            try:
                data = self.model.main(img)
                results.append(data)
                print(i)
            except:
                pass
        data = pd.concat(results)
        data = data[data.centroid_x.notna()]
        data["centroid_x"] = data["centroid_x"] * self.IMG_SIZE_MM + data["img_x"].astype(int) * self.STEP_MM
        data["centroid_y"] = data["centroid_y"] * self.IMG_SIZE_MM + data["img_y"].astype(int) * self.STEP_MM
        data["N"] = range(len(data))
        data[["label"]] = data.id.astype(str) + "_" + data.img_x + "_" + data.img_y
        self.data = data