from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import os
import pandas as pd

from segmentation import Model_Nematoides

from classes import Stage, Camera
stage = Stage("COM4")
camera = Camera(0)

if __name__ == "__main__":
    stage.scan(xstart = 0, xend = 2, step_x = .5,
               ystart = 0, yend = 3, step_y = .5,
               camera = camera, invert_x=False,
               invert_y = False,
               extension="tif")
    stage.go_home()
    stage.move_x(.5) ## To avoid homing fail
    stage.move_y(.5) ## To avoid homing fail

