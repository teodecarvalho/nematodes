import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from skimage.io import imsave, imread
import imagej

class Image_Matrix():
    def __init__(self, matrix):
        self.matrix = matrix
        self.regions = {}
        self.found = []
        self.new_cells = []
        self.nrows = self.matrix.shape[0]
        self.ncols = self.matrix.shape[1]

    def get_neighbours(self, coord):
        i, j = coord
        return ([(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                 (i, j - 1), (i, j + 1),
                 (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)])

    def create_region(self, pivot):
        self.found.append(pivot)
        self.new_cells += self.get_neighbours(pivot)
        self.regions[pivot] = [pivot] + self.get_neighbours(pivot)
        self.current_region = pivot

    def add_cell(self, cell):
        self.found.append(cell)
        self.new_cells += self.get_neighbours(cell)
        self.regions[self.current_region] += [cell] + self.get_neighbours(cell)

    def check_new_cells(self):
        while self.new_cells:
            i, j = self.new_cells.pop()
            try:
                if self.matrix[i, j] == 1:
                    if not (i, j) in self.found:
                        self.add_cell(cell=(i, j))
            except:
                print(i)

    def get_regions(self):
        for i in range(self.nrows - 2)[1:]:  # The first and last rows are not considered
            for j in range(self.ncols - 2)[1:]:  # The first and last cols are not considered
                if self.matrix[i, j] == 1:
                    if not (i, j) in self.found:
                        self.create_region(pivot=(i, j))
                        self.check_new_cells()
        return self.regions

class Stitcher():
    def __init__(self, regions):
        self.regions = regions

    def make_region_square(self, region_idx):
        key = list(self.regions.keys())[region_idx]
        region = self.regions[key]
        i_s = [r[0] for r in region]
        j_s = [r[1] for r in region]
        topleft_corner = (min(i_s), min(j_s))
        bottom_right_corner = (max(i_s), max(j_s))
        for i in range(topleft_corner[0], bottom_right_corner[0] + 1):
            for j in range(topleft_corner[1], bottom_right_corner[1] + 1):
                if not (i, j) in region:
                    region.append((i, j))
        return (topleft_corner, bottom_right_corner, region)

    def get_images(self, image_dir, region_idx):
        topleft_corner, bottom_right_corner, square = self.make_region_square(region_idx)
        if os.path.isdir("./temp"):
            shutil.rmtree("./temp")
        os.mkdir("./temp")
        empty_image = np.zeros(shape = (480, 480, 3))
        for i, j in square:
            file_name = f"{image_dir}/nem__{j}__{i}__.png"
            if os.path.isfile(file_name):
                img = imread(file_name)[:,:,0:3]
            else:
                img = empty_image
            out_file = f"./temp/nem__{j}__{i}__.tif"
            imsave(out_file, img)

    def stitch_region(self, region_idx):
        topleft_corner, bottom_right_corner, square = self.make_region_square(region_idx)
        max_x, max_y = bottom_right_corner
        min_x, min_y = topleft_corner
        fiji_macro = """run("Grid/Collection stitching", """ +\
        """ "type=[Filename defined position] order=[Defined by filename         ] """ +\
        f"""grid_size_x={(max_y-min_y) + 1} grid_size_y={(max_x - min_x) + 1} """ +\
        f"""tile_overlap=15 first_file_index_x={min_y} """+\
        f"""first_file_index_y={min_x} directory={os.path.abspath("./temp")} """ +\
        """file_names=nem__{x}__{y}__.tif output_textfile_name=TileConfiguration.txt """ +\
        """fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 """ +\
        """absolute_displacement_threshold=3.50 compute_overlap """ +\
        """computation_parameters=[Save memory (but be slower)] """ +\
        """image_output=[Fuse and display]"); """ +\
        f"""saveAs("PNG", "{os.path.abspath("./")}/Fused_{region_idx}.png");""" +\
        """close();"""
        with open("./temp/macro.ijm", "w") as handle:
            handle.write(fiji_macro)
        cmd = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx --headless ' +\
              f'-macro {os.path.abspath("./temp")}/macro.ijm'
        os.system(cmd)
















