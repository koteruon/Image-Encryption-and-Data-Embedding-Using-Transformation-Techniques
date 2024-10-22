# -*- coding: utf8 -*-

import math

import cv2
import numpy as np


def do_rectangular_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array) -> np.array:
    encrypt_image = np.full((M, N), -1)
    MN = np.array([M, N])
    A = np.array([[a, b], [c, d]])
    row_indices, col_indices = np.indices(image.shape)
    coordinates = np.stack((row_indices, col_indices), axis=-1)
    new_coordinates = np.remainder(np.matmul(coordinates, A.T), MN).astype(int)
    encrypt_image[new_coordinates[..., 0], new_coordinates[..., 1]] = image[row_indices, col_indices]
    return encrypt_image


def is_the_same_picture(image: np.array, encrypt_image: np.array):
    return np.array_equal(image, encrypt_image)


def is_rectangular_transformations(M: int, N: int, a: int, b: int, c: int, d: int) -> bool:
    p = math.gcd(M, N)
    l1 = M / p
    l2 = N / p
    if b % l1 != 0 and c % l2 != 0:  # b mod l1 == 0 or c mod l2 == 0
        return False
    if math.gcd(a * d - b * c, p) != 1:
        return False
    if math.gcd(a, l1) != 1:
        return False
    if math.gcd(d, l2) != 1:
        return False
    return True


def repeat_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array) -> int:
    if is_rectangular_transformations:
        encrypt_image = image.copy()
        for period in range(1, (M * N) // 2):
            print(period)
            encrypt_image = do_rectangular_transformations(M, N, a, b, c, d, encrypt_image)
            if is_the_same_picture(image, encrypt_image):
                return period
    return None


def write_results(M: int, N: int, period: int, a: int, b: int, c: int, d: int):
    if period:
        print(f"M = {M}, N = {N}, (a, b, c, d) = ({a}, {b}, {c}, {d}), Period = {period}")
    else:
        print("Invalid RT matrix!")


def read_parameters():
    M, N = map(str, input("Please input M N: ").split())
    a, b, c, d = map(str, input("Place input a b c d: ").split())
    return int(M), int(N), int(a), int(b), int(c), int(d)


def main():
    # read parameters
    M, N, a, b, c, d = read_parameters()
    image = np.arange(M * N).reshape(M, N)

    period = repeat_transformations(M, N, a, b, c, d, image)
    write_results(M, N, period, a, b, c, d)


if __name__ == "__main__":
    main()
