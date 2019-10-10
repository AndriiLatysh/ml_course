import math


class Circle:

    def __init__(self, x, y, r):
        self.x_coordinate = x
        self.y_coordinate = y
        self.radius = r

    def calculate_area(self):
        area = math.pi * self.radius ** 2
        return area

    def move(self, delta_x, delta_y):
        self.x_coordinate += delta_x
        self.y_coordinate += delta_y

    def __str__(self):
        circle_str = "Center: ({}, {}); Radius: {}".format(self.x_coordinate, self.y_coordinate, self.radius)
        return circle_str


class Sphere(Circle):

    def __init__(self, x, y, z, r):
        super().__init__(x, y, r)
        self.z_coordinate = z

    def calculate_area(self):
        area = 4 * math.pi * self.radius ** 2
        return area

    def move(self, delta_x, delta_y, delta_z):
        super().move(delta_x, delta_y)
        self.z_coordinate += delta_z

    def __str__(self):
        circle_str = "Center: ({}, {}, {}); Radius: {}".format(self.x_coordinate, self.y_coordinate, self.z_coordinate,
                                                               self.radius)
        return circle_str


s = Sphere(1, 1, 1, 2)
print(s)
s.move(-1, -2, -3)
print(s)
