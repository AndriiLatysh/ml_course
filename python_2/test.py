import math

# 2
#
# n = int(input())
#
# max = int(input())
#
# for z in range(1, n):
#     t = int(input())
#     if t > max:
#         max = t
#
# print(max)

# # 7
#
# n = int(input())
#
# x_prev = x0 = float(input())
# y_prev = y0 = float(input())
#
# perimeter = 0
#
# for z in range(1, n):
#     x_last = float(input())
#     y_last = float(input())
#
#     perimeter += math.sqrt((x_last - x_prev) ** 2 + (y_last - y_prev) ** 2)
#
#     x_prev = x_last
#     y_prev = y_last
#
# perimeter += math.sqrt((x_prev - x0) ** 2 + (y_prev - y0) ** 2)
#
# print(perimeter)

primes = [2, 3, 5, 7, 11]

# print(primes[4::-1])

print(sum(primes) / len(primes))
