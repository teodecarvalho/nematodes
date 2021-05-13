from time import sleep

from classes import Stage, Camera
stage = Stage("COM3")
camera = Camera(0)

if __name__ == "__main__":
    stage.scan(xstart = 8, xend = 12, step_x = 1,
               ystart = 2, yend = 6, step_y = 1,
               camera = camera, invert_x=False,
               invert_y = False)
