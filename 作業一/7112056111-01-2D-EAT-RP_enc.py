# -*- coding: utf8 -*-

import os

import cv2
import numpy as np

# eat cycle numbers
EAT_CYCLE_NUMBERS = {512: 384}

# EAT's a, b variables
a = 10
b = 20

# times of EAT
G = 89

# seed number
seed = 100


def equilateral_arnold_transform(a: int, b: int, xy: np.array, N: int, G: int) -> np.array:
    """EAT method

    Args:
        a (int): variable
        b (int): variable
        xy (np.array): pixel coordinates
        N (int): resolution
        G (int): times of EAT

    Returns:
        np.array: coordinates after transformation
    """

    at_matrix = np.array([[1, a], [b, a * b + 1]])
    for _ in range(G):
        xy = np.remainder(np.matmul(at_matrix, xy), [N, N])  # (at_matrix * xy) mod N
    return xy


def durstenfeld_random_permutation(seed: int, bit_pixels: np.array) -> np.array:
    """durstenfeld random permutation

    Args:
        seed (int): seed number
        bit_pixels (np.array): bits array in one pixel

    Returns:
        np.array: after transformation bits array
    """

    for pivot in reversed(range(1, bit_pixels.size)):
        target = seed % pivot
        bit_pixels[target], bit_pixels[pivot] = bit_pixels[pivot], bit_pixels[target]  # swap two bits
    return bit_pixels


def encrypt_the_image(image: np.array) -> np.array:
    """encrypt the image

    Args:
        image (np.array): pixels array

    Returns:
        np.array: after encrypt pixels array
    """
    # encrypt image
    encrypt_image = np.zeros_like(image)
    # iterate every bit in the image
    for row, col in np.ndindex(image.shape[:2]):
        # do EVA
        xy = np.array([row, col])
        N = image.shape[0]
        xy_prime = equilateral_arnold_transform(a, b, xy, N, G)
        row_prime, col_prime = xy_prime[0], xy_prime[1]

        # do Durstenfeld PR
        bit_pixels = np.unpackbits(image[row][col])
        bit_shuffling_pixels = durstenfeld_random_permutation(seed, bit_pixels)
        shuffling_pixel = np.packbits(bit_shuffling_pixels)

        # output
        encrypt_image[row_prime][col_prime] = shuffling_pixel
    return encrypt_image


def main():
    # iterate over files in the directory
    source_dir = ".\\source"
    encrypt_dir = ".\\encryp"
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        image = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        # encrypt the image
        encrypt_image = encrypt_the_image(image)
        encrypt_file = os.path.join(encrypt_dir, "{0}_{2}{1}".format(*os.path.splitext(filename) + ("enc",)))

        # RGB color
        if is_rgb:
            encrypt_image = cv2.cvtColor(encrypt_image, cv2.COLOR_RGB2BGR)
        else:
            # squeeze dimensions
            encrypt_image = np.squeeze(encrypt_image, axis=-1)

        # saving the image
        cv2.imwrite(encrypt_file, encrypt_image)


if __name__ == "__main__":
    main()
