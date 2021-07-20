from skimage.io import imread, imsave
from glob import glob
import json
import numpy as np
import os
import shutil
from skimage.draw import polygon

def combine_layers():
    images = glob("imagens_gray/*_.png")
    for image in images:
        img = imread(image)
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
        RelR = .5 * B + .5 * G
        imsave('processed_images/' + image.split("imagens_gray/")[1], RelR.astype(int))

class Masks():
    def __init__(self, json_path):
        self.load_masks_from_json(json_path=json_path)

    def load_masks_from_json(self, json_path):
        with open(json_path, "r") as file:
            txt = file.read()
            masks = json.loads(txt)
        self.masks_metadata = masks['_via_img_metadata']

    def get_polygons_coord(self, key):
        return self.masks_metadata[key]['regions']

    def draw_polygons(self, regions):
        img = np.zeros((480, 480), dtype = np.uint8)
        if regions:
            for region in regions:
                x = region['shape_attributes']['all_points_x']
                x = [c_x - 1 for c_x in x]
                y = region['shape_attributes']['all_points_y']
                y = [c_y - 1 for c_y in y]
                rr, cc = polygon(y, x)
                img[rr, cc] = 254
        return img

    def make_binary_masks(self):
        if os.path.isdir("./masks"):
            shutil.rmtree("./masks")
        os.mkdir("./masks")
        for mask in self.masks_metadata.values():
            regions = mask['regions']
            img = self.draw_polygons(regions)
            imsave(f"./masks/{mask['filename']}", img)

if __name__ == "__main__":
    masks = Masks("masks_project_nematodes.json")
    masks.make_binary_masks()