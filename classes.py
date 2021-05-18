import serial
from time import sleep
from math import ceil
import cv2

class Stage():
    def __init__(self, port):
        self.connect(port)
        self.go_home()
        self.wait_move()

    def connect(self, port):
        self.serial = serial.Serial(port, 115200, timeout=3)
        sleep(3)

    def send_gcode_str(self, gcode_str, debug = True):
        self.serial.flushInput()
        sleep(.01)
        if debug:
            print(gcode_str)
        gcode_str += "\n"
        self.serial.write(gcode_str.encode())
        grbl_out = self.serial.readline()  # Wait for robot response with carriage return
        if debug:
            print(grbl_out.strip())
        return (grbl_out.strip().decode("ascii"))

    def is_done(self):
        status = self.send_gcode_str("G4 P0")
        if status == '':
            self.send_gcode_str("?")
        return status == 'ok'

    def go_home(self):
        self.send_gcode_str("$H")

    def wait_move(self):
        while not self.is_done():
            sleep(.5)
            pass
        return True

    def move_x(self, x, invert_x = False):
        if invert_x:
            x = -x
        self.send_gcode_str(f"G90 X{x}")
        return self.wait_move()

    def move_y(self, y, invert_y = False):
        if invert_y:
            y = -y
        self.send_gcode_str(f"G90 Y{y}")
        return self.wait_move()

    def scan(self,camera, xend, yend, step_x, step_y, xstart = 0, ystart = 0,
             invert_x = False, invert_y = False):
        range_x = xend - xstart
        range_y = yend - ystart
        nsteps_x = ceil(range_x/float(step_x)) + 1
        nsteps_y = ceil(range_y/ float(step_y)) + 1
        flip = False
        for y in range(nsteps_y):
            self.move_y(ystart + y * step_y, invert_y = invert_y)
            if flip:
                x_seq = range(nsteps_x)[::-1]
            else:
                x_seq = range(nsteps_x)
            flip = not flip
            for x in x_seq:
                self.move_x(xstart + x * step_x, invert_x = invert_x)
                camera.capture_during_scan(x, y)
        camera.cap.release()

class Camera():
    def __init__(self, camera):
        self.connect(camera=camera)

    def connect(self, camera):
        self.cap = cv2.VideoCapture(camera)

    def capture(self):
        image = .1 * self.cap.read()[1]
        for i in range(9):
            image += .1 * self.cap.read()[1] # return a single frame in variable `frame`
        return image

    def save(self, image_obj, filepath):
        cv2.imwrite(filepath, image_obj)

    def crop(self, image, xstart, xend, ystart, yend):
        return image[xstart:xend, ystart:yend]

    def capture_red_layer_and_make_square(self, image):
        return image[:, 0:480, 0]

    def capture_during_scan(self, x, y):
        self._image = self.capture()
        self._image = self.capture_red_layer_and_make_square(self._image)
        self.save(self._image, f"imagens/nem__{x}__{y}__.png")