# -*- coding: utf8 -*-

import math
import os

import cv2
import numpy as np

SOURCE = "./source"


def do_rectangular_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array) -> np.array:
    encrypt_image = np.full((M, N), -1)
    MN = np.array([M, N])
    A = np.array([[a, b], [c, d]])

    # 執行矩陣乘法和餘數計算
    row_indices, col_indices = np.indices(image.shape)
    coordinates = np.stack((row_indices, col_indices), axis=-1)
    new_coordinates = np.remainder(np.matmul(coordinates, A.T), MN).astype(int)

    # 將原像素值複製到新位置
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


def repeat_transformations(
    M: int, N: int, a1: int, a2: int, b1: int, b2: int, c1: int, c2: int, d1: int, d2: int, image: np.array
) -> list:
    parameters_result = []
    for a in range(a1, a2 + 1):
        if not parameters_result:
            for b in range(b1, b2 + 1):
                if not parameters_result:
                    for c in range(c1, c2 + 1):
                        if not parameters_result:
                            for d in range(d1, d2 + 1):
                                if not parameters_result:
                                    if is_rectangular_transformations:
                                        print(a, b, c, d)
                                        encrypt_image = image.copy()
                                        for period in range(1, (M * N) // 2):
                                            encrypt_image = do_rectangular_transformations(
                                                M, N, a, b, c, d, encrypt_image
                                            )
                                            if -1 in encrypt_image:
                                                break
                                            if is_the_same_picture(image, encrypt_image):
                                                parameters_result.append((a, b, c, d, period))
                                                break
    return parameters_result


def write_results(M: int, N: int, parameters_result: list):
    with open(f"{M}_{N}_parameters.csv", "w") as f:
        f.write(f"{'No.':>8},{'a':>8},{'b':>8},{'c':>8},{'d':>8},{'period':>8}\n")
        print(f"{'No.':>8},{'a':>8},{'b':>8},{'c':>8},{'d':>8},{'period':>8}")
        if parameters_result:
            for idx, (a, b, c, d, period) in enumerate(parameters_result):
                f.write(f"{idx + 1:>8},{a:>8},{b:>8},{c:>8},{d:>8},{period:>8}\n")
                print(f"{idx + 1:>8},{a:>8},{b:>8},{c:>8},{d:>8},{period:>8}")
        else:
            f.write("No legal matrix within the range!")
            print("No legal matrix within the range!")


def read_parameters():
    M, N = map(str, input("Input M N: ").split())
    a1, a2 = map(str, input("Input a1 a2: ").split())
    b1, b2 = map(str, input("Input b1 b2: ").split())
    c1, c2 = map(str, input("Input c1 c2: ").split())
    d1, d2 = map(str, input("Input d1 d2: ").split())
    return int(M), int(N), int(a1), int(a2), int(b1), int(b2), int(c1), int(c2), int(d1), int(d2)


def main():
    # read parameters
    M, N, a1, a2, b1, b2, c1, c2, d1, d2 = read_parameters()
    image = np.arange(M * N).reshape(M, N)
    parameters_result = repeat_transformations(M, N, a1, a2, b1, b2, c1, c2, d1, d2, image)
    write_results(M, N, parameters_result)


if __name__ == "__main__":
    main()
