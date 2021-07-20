import numpy as np
import matplotlib.pyplot as plt

matrix = np.zeros((10, 15))

matrix[1,1] = 1
matrix[2,2] = 1
matrix[3,3] = 1
matrix[3,4] = 1
matrix[1,4] = 1
matrix[5,13] = 1


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
            if matrix[i, j] == 1:
                if not (i, j) in self.found:
                    self.add_cell(cell=(i, j))

    # def make_region_square(self):
    #   region = self.current_region
    #   i_s = [r[0] for r in region]
    #   j_s = [r[1] for r in region]
    #   topleft_corner = (min(i_s), min(j_s))
    #   bottom_right_corner = (max(i_s), max(j_s))
    #   for i in range(topleft_corner[0], bottom_right_corner[0] + 1):
    #     for j in range(topleft_corner[1], bottom_right_corner[1] + 1)
    #       if not (i, j) in self.found

    def get_regions(self):
        for i in range(self.nrows - 1)[1:]:  # The first and last rows are not considered
            for j in range(self.ncols - 1)[1:]:  # The first and last cols are not considered
                if matrix[i, j] == 1:
                    if not (i, j) in self.found:
                        self.create_region(pivot=(i, j))
                        self.check_new_cells()
        return self.regions


image_matrix = Image_Matrix(matrix)
regions = image_matrix.get_regions()
plt.imshow(matrix, cmap="gray")
for region in regions.values():
  for coord in region:
    matrix[coord[0], coord[1]] = 3

plt.imshow(matrix, cmap="gray")