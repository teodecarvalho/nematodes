from segmentation import Model_Nematoides
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, medial_axis, closing, square
from skimage.io import imread
from skimage.segmentation import clear_border
from skan import Skeleton
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

class Measure():
    def __init__(self):
        self._model = Model_Nematoides()

    def calculate_width(self, image):
        _, distance = medial_axis(image, return_distance=True)
        max_width = max(distance[image > 0]) * 2
        return(max_width)

    def measure_nematoide(self, img_path, mask, show_image=False):
        label_im = label(mask)
        results = []
        if regionprops(label_im):
            for count, region in enumerate(regionprops(label_im)):
                skel = skeletonize(region.image)
                Skel = Skeleton(skel)
                length = Skel.path_lengths()[0] if Skel.n_paths == 1 else 0
                width = self.calculate_width(region.image) if Skel.n_paths == 1 else 0
                fatness = float(region.area) / Skel.path_lengths()[0] if Skel.n_paths == 1 else 0
                result = {"id": count,
                          "width": width,
                          "npaths": Skel.n_paths,
                          "length": length,
                          "coords": region.coords,
                          "bbox": region.bbox,
                          "skel": skel,
                          "fatness": fatness,
                          "centroid_x_pix": region.centroid[1],
                          "centroid_y_pix": region.centroid[0],
                          "centroid_x": region.centroid[1] / float(mask.shape[1]),
                          "centroid_y": 1 - region.centroid[0] / float(mask.shape[0]),
                          "image": region.image}
                results.append(result)
            data = pd.DataFrame(results)
            self.annotate_img(img_path, data)
        else:
            results = [{"id": None, "width": None, "npaths": None, "length": None, "coords": None, "bbox": None, "skel": None, "fatness": None, "centroid_x": None,
                        "centroid_y": None, "centroid_y_pix": None, "centroid_x_pix": None, "image": None}]
            data = pd.DataFrame(results)
        return data

    def find_and_measure_one(self, img_path):
        mask = self._model.segment(img_path, save_masks=True)
        data = self.measure_nematoide(mask = mask, img_path = img_path, show_image = True)
        return data

    def superimpose_skeleton(self, skel, bbox, image, coords):
        for x, y in coords:
            if skel[x - bbox[0], y - bbox[1]] > 0:
                image[x, y] = 1
        return image

    def annotate_img(self, img_path, data):
        img = imread(img_path)
        filename = img_path.split("/")[2]
        for i, row in data.iterrows():
            img = self.superimpose_skeleton(skel = row.skel, bbox = row.bbox, image = img, coords = row.coords)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray")
        for i, row in data.iterrows():
            text = f"ID: {row.id}; Length: {row.length:.2f}; Width: {row.width:.2f}"
            ax.text(row.centroid_x_pix, row.centroid_y_pix, text, color="black")
        plt.axis('off')
        fig.savefig("./annotated_imgs/" + filename,
                    bbox_inches='tight', pad_inches=0, dpi=624 / 10, transparent=False)

    def find_and_measure_all(self, img_dir = "./nem_to_measure/"):
        results = []
        for file in glob(img_dir + "*.png"):
            data = self.find_and_measure_one(file)
            results.append(data)
        return pd.concat(results)

if __name__ == "__main__":
    from measure_nematodes import Measure
    measure = Measure()
    data = measure.find_and_measure_all()