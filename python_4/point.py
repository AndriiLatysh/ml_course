import math


class Point:

    def __init__(self, x, y):
        self.x_coordinate = x
        self.y_coordinate = y

    def calculate_distance_to_origin(self):
        distance = math.sqrt(self.x_coordinate ** 2 + self.y_coordinate ** 2)
        return distance

    def move(self, x_shift, y_shift):
        self.x_coordinate += x_shift
        self.y_coordinate += y_shift

    def __str__(self):
        point_str = "({}, {})".format(self.x_coordinate, self.y_coordinate)
        return point_str


class ColoredPoint(Point):

    def __init__(self, x, y, color):
        super().__init__(x, y)
        self.__color = color

    def recolor(self, color):
        self.__color = color

    def __str__(self):
        colored_point_str = super().__str__() + " {}".format(self.__color)
        return colored_point_str


cp = ColoredPoint(1, 1, "red")
print(cp)
cp.move(2, 3)
print(cp)
print(cp.calculate_distance_to_origin())
print(cp._ColoredPoint__color)
