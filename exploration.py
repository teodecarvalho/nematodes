import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from glob import glob
import json
import numpy as np
import os
import shutil
from skimage.draw import polygon
#from process_data import Nematode_df
#nematodes = Nematode_df()
#nematodes.data.to_csv("data_w_images.csv")

from masks import Masks
masks = Masks("masks_project_nematodes.json")
masks.make_binary_masks()


def combine_layers():
    images = glob("imagens_gray/*_.png")
    for image in images:
        img = imread(image)
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
        RelR = .5 * B + .5 * G
        imsave('processed_images/' + image.split("imagens_gray/")[1], RelR.astype(int))

# #plt.imshow(RelR, cmap = "gray")
# data = nematodes.data
# for i, row in data[['img_x', "img_y"]].drop_duplicates().iterrows():
#     image_path = f"imagens_gray/nem__{row.img_x}__{row.img_y}__.png"
#     sub_data = data[(data.img_x == str(row.img_x)) & (data.img_y == str(row.img_y))]
#     img = imread(image_path)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(img, cmap = "gray")
#     for i, row in sub_data.iterrows():
#         ax.text(row.centroid_x_pix * 2.1428, row.centroid_y_pix * 2.1428, row.id, color="black")
#     plt.axis('off')
#     fig.savefig(f"imagens_gray/nem__{row.img_x}__{row.img_y}__2.png",
#                 bbox_inches='tight', pad_inches=0, dpi = 624/10, transparent=False)
