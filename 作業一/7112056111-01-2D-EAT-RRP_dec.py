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


def reverse_equilateral_arnold_transform(a: int, b: int, xy_prime: np.array, N: int, G: int) -> np.array:
    """reverse EAT method

    Args:
        a (int): variable
        b (int): variable
        xy_prime (np.array): pixel coordinates
        N (int): resolution
        G (int): times of reverse EAT

    Returns:
        np.array: coordinates after transformation
    """

    at_matrix = np.array([[a * b + 1, -a], [-b, 1]])
    for _ in range(G):
        xy_prime = np.remainder(np.matmul(at_matrix, xy_prime), [N, N])  # (at_matrix * xy_prime) mod N
    return xy_prime


def reverse_durstenfeld_random_permutation(seed: int, bit_pixels: np.array) -> np.array:
    """reverse durstenfeld random permutation

    Args:
        seed (int): seed number
        bit_pixels (np.array): bits array in one pixel

    Returns:
        np.array: after transformation bits array
    """

    for pivot in range(1, bit_pixels.size):
        target = seed % pivot
        bit_pixels[target], bit_pixels[pivot] = bit_pixels[pivot], bit_pixels[target]  # swap two bits
    return bit_pixels


def decrypt_the_image(image: np.array) -> np.array:
    """decrypt the image

    Args:
        image (np.array): pixels array

    Returns:
        np.array: after decrypt pixels array
    """
    # decrypt image
    decrypt_image = np.zeros_like(image)
    # iterate every bit in the image
    for row_prime, col_prime in np.ndindex(image.shape[:2]):
        # do reverse EVA
        xy_prime = np.array([row_prime, col_prime])
        N = image.shape[0]
        xy = reverse_equilateral_arnold_transform(a, b, xy_prime, N, G)
        row, col = xy[0], xy[1]

        # do Durstenfeld PR
        bit_shuffling_pixels = np.unpackbits(image[row_prime][col_prime])
        bit_pixels = reverse_durstenfeld_random_permutation(seed, bit_shuffling_pixels)
        pixel = np.packbits(bit_pixels)

        # output
        decrypt_image[row][col] = pixel
    return decrypt_image


def main():
    # iterate over files in the directory
    encrypt_dir = ".\\encryp"
    decrypt_dir = ".\\decryp"
    for filename in os.listdir(encrypt_dir):
        encrypt_file = os.path.join(encrypt_dir, filename)
        image = cv2.imread(encrypt_file, cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        # decrypt the image
        decrypt_image = decrypt_the_image(image)
        decrypt_file = os.path.join(
            decrypt_dir, "{0}_{2}{1}".format(*os.path.splitext(filename.replace("_enc", "")) + ("dec",))
        )

        # RGB color
        if is_rgb:
            decrypt_image = cv2.cvtColor(decrypt_image, cv2.COLOR_RGB2BGR)
        else:
            # squeeze dimensions
            decrypt_image = np.squeeze(decrypt_image, axis=-1)

        # saving the image
        cv2.imwrite(decrypt_file, decrypt_image)


if __name__ == "__main__":
    main()
