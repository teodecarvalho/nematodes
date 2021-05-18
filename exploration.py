import matplotlib.pyplot as plt
from skimage.io import imread

from process_data import Nematode_df
nematodes = Nematode_df()

data = nematodes.data
for i, row in data[['img_x', "img_y"]].drop_duplicates().iterrows():
    image_path = f"imagens/nem__{row.img_x}__{row.img_y}__.png"
    sub_data = data[(data.img_x == str(row.img_x)) & (data.img_y == str(row.img_y))]
    img = imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap = "gray")
    for i, row in sub_data.iterrows():
        ax.text(row.centroid_x_pix * 2.1428, row.centroid_y_pix * 2.1428, row.id, color="black")
    plt.axis('off')
    fig.savefig(f"imagens/nem__{row.img_x}__{row.img_y}__2.png",
                bbox_inches='tight', pad_inches=0, dpi = 624/10, transparent=False)
