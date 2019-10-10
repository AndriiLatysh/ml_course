import math


def factorial(n):
    n_factorial = 1
    for z in range(1, n + 1):
        n_factorial *= z
    return n_factorial


def calculate_number_of_combinations(n, k=40):
    number_of_combinations = factorial(n) // (factorial(k) * factorial(n - k))
    return number_of_combinations


total_number_of_users = int(input("Total: "))
selected_number_of_users = int(input("Selected: "))

print("Number of combinations: {}".format(
    calculate_number_of_combinations(n=total_number_of_users, k=selected_number_of_users)))
