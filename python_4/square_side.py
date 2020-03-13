import math


# print(1/0)


def get_side_by_area(area):
    # if area < 0:
    #     raise ValueError("Area value received: {}. Area can't be negative.".format(area))
    return math.sqrt(area)


area = float(input("Area: "))
print("Side: {}".format(get_side_by_area(area)))
try:
    print("Side: {}".format(get_side_by_area(area)))
except ValueError as value_error:
    print(value_error)
    print("Method received illegal value.")
