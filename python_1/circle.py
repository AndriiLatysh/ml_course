import math


class Circle:

    def __init__(self, x, y, radius):
        self.x_coordinate = x
        self.y_coordinate = y
        self.radius = radius

    def move(self, x_shift=0, y_shift=0):
        self.x_coordinate += x_shift
        self.y_coordinate += y_shift

    def calculate_area(self):
        area = math.pi * self.radius ** 2
        return area

    def __str__(self):
        representation = "x: {0} ; y = {1} ; r = {2}".format(self.x_coordinate, self.y_coordinate, self.radius)
        return representation
