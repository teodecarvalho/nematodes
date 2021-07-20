import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from glob import glob
import json
import numpy as np
import os
import shutil
from skimage.draw import polygon
import pandas as pd

from process_data import Nematode_df
nematodes = Nematode_df()
nematodes.data.to_csv("data_w_images.csv")

data = nematodes.data
#data = pd.read_csv("data_w_images.csv")

for i, row in data[['img_x', "img_y"]].drop_duplicates().iterrows():
    image_path = f"imagens_col/nem__{row.img_x}__{row.img_y}__.png"
    sub_data = data[(data.img_x == row.img_x) & (data.img_y == row.img_y)]
    if image_path == "imagens_col/nem__19__4__.png":
        print("Found!")
    img = imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap = "gray")
    for i, row in sub_data.iterrows():
        ax.text(row.centroid_x_pix, row.centroid_y_pix, row.id, color="black")
    plt.axis('off')
    fig.savefig(f"imagens_col/nem__{row.img_x}__{row.img_y}__2.png",
                bbox_inches='tight', pad_inches=0, dpi = 624/10, transparent=False)


#
# import os
# from skimage.io import imread, imsave
#
# files = [f for f in os.listdir("imagens_col/") if ".png" in f]
# xs = [int(f.split("__")[1]) for f in files]
# lookup_xs = {i:62-i for i in list(range(0,62+1))}
# xs_inv = [lookup_xs[i] for i in xs]
# ys = [f.split("__")[2] for f in files]
# print(xs)
# print(ys)
#
# for file, x, y in zip(files, xs_inv, ys):
#     img = imread("imagens_col/" + file)
#     imsave("imagens_col_inv/" + f"nem__{x}__{y}__.tif", img)
