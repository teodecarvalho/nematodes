from classes import Stage, Camera
stage = Stage("/dev/tty.usbserial-1460")
camera = Camera(0)
if __name__ == "__main__":
    stage.scan(xstart = 0, xend = 1, step_x = .1,
               ystart = 0, yend = 1, step_y = .1,
               camera = camera, invert_x=True,
               invert_y = True)