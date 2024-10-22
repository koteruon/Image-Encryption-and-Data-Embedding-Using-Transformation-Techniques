# -*- coding: utf8 -*-

import hashlib
import math
import os
import queue

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

# Transient effect constant
g = 1000

# directory hierarchy
SOURCE_DIR = ".\\source"
ENCRYPT_DIR = ".\\encryp"


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


def generating_attribute_values(plaintext_image: np.array) -> tuple[float, float, float, float, int, int, int, int]:
    """generate attribute values

    Args:
        plaintext_image (np.array): original plaintext image

    Returns:
        tuple[float, float, float, float, int, int, int, int]: a_hat, b_hat, x_0, y_0, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prim
    """
    hash_image = hashlib.sha256(plaintext_image).hexdigest()
    binary_hash_image = bin(int(hash_image, 16))
    binary_idx = 2  # 0b
    a_hat = 500.0 + round(int(binary_hash_image[binary_idx : binary_idx + 16], 2) / 2**16, 7)
    binary_idx += 16
    b_hat = 500.0 + round(int(binary_hash_image[binary_idx : binary_idx + 16], 2) / 2**16, 7)
    binary_idx += 16
    x_0 = 0.1 + round(int(binary_hash_image[binary_idx : binary_idx + 8], 2) / 2**8, 7) * (10**-1)
    binary_idx += 8
    y_0 = 0.1 + round(int(binary_hash_image[binary_idx : binary_idx + 8], 2) / 2**8, 7) * (10**-1)
    binary_idx += 8
    p_0_r_prime = int(binary_hash_image[binary_idx : binary_idx + 8], 2)
    binary_idx += 8
    p_0_g_prime = int(binary_hash_image[binary_idx : binary_idx + 8], 2)
    binary_idx += 8
    p_0_b_prime = int(binary_hash_image[binary_idx : binary_idx + 8], 2)
    binary_idx += 8
    c_0_b_prime = int(binary_hash_image[binary_idx : binary_idx + 8], 2)
    binary_idx += 8
    return a_hat, b_hat, x_0, y_0, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prime


def enhanced_two_dimensional_henon_map(a_hat: float, b_hat: float, x_n: float, y_n: float) -> tuple[float, float]:
    """enhanced two dimensional Henon map

    Args:
        a_hat (float): control parameters a_hat
        b_hat (float): control parameters b_hat
        x_n (float): nth of x sequence
        y_n (float): nth of y sequence

    Returns:
        tuple[float, float]: (n + 1)th of x sequence, (n + 1)th of y sequence
    """
    x_n_plus_1 = math.sin(math.pi * (1 - (a_hat * (x_n**2)) + y_n))
    y_n_plus_1 = math.sin(math.pi * (b_hat * x_n))
    return x_n_plus_1, y_n_plus_1


def operating_pixel_diffusion(
    g: int, a_hat: float, b_hat: float, x_n: float, y_n: float, image: np.array
) -> queue.Queue:
    """operating pixel diffusion

    Args:
        g (int): ignore g element
        a_hat (float): a hat value
        b_hat (float): b hat value
        x_n (float): x_0
        y_n (float): y_0
        image (np.array): plaintext image array

    Returns:
        queue.Queue: random decimal sequence
    """
    r = queue.Queue()
    for i in range(g):
        x_n, y_n = enhanced_two_dimensional_henon_map(a_hat, b_hat, x_n, y_n)

    for i in range(image.shape[0] * image.shape[1] * image.shape[2]):  # row * col * channel
        x_n, y_n = enhanced_two_dimensional_henon_map(a_hat, b_hat, x_n, y_n)
        r.put(int(round((x_n * (10**7)) % 256, 0)), int(round((y_n * (10**7)) % 256, 0)))
    return r


def pixel_scrambling_using_exclusive_or(r: queue.Queue, p_plus_1: np.array, p: np.array, c_b_prime: int) -> np.array:
    """pixel scrambling using exclusive or

    Args:
        r (queue.Queue): random decimal sequence
        p_plus_1 (np.array): plaintext image pixel
        p (np.array): previous plaintext image pixel
        c_b_prime (int): previous encrypted pixel

    Returns:
        np.array: encrypted pixel
    """
    c_plus_1 = p_plus_1.copy()
    if p_plus_1.size == 1:
        c_plus_1_prime = r.get() ^ p_plus_1[0] ^ p[0] ^ c_b_prime
        c_plus_1[0] = c_plus_1_prime
    else:
        # rgb
        c_plus_1_r_prime = r.get() ^ p_plus_1[0] ^ p[0] ^ c_b_prime
        c_plus_1_g_prime = r.get() ^ p_plus_1[1] ^ p[1] ^ c_plus_1_r_prime
        c_plus_1_b_prime = r.get() ^ p_plus_1[2] ^ p[2] ^ c_plus_1_g_prime
        c_plus_1[0], c_plus_1[1], c_plus_1[2] = c_plus_1_r_prime, c_plus_1_g_prime, c_plus_1_b_prime
    return c_plus_1


def save_secret_keys(
    secret_file: str,
    a_hat: float,
    b_hat: float,
    x_0: float,
    y_0: float,
    p_0_r_prime: np.array,
    p_0_g_prime: np.array,
    p_0_b_prime: np.array,
    c_0_b_prime: int,
):
    """save secret keys

    Args:
        secret_file (str): secret file path
        a_hat (float): a_hat
        b_hat (float): b_hat
        x_0 (float): x_0
        y_0 (float): y_0
        p_0_r_prime (np.array): p_0_r_prime
        p_0_g_prime (np.array): p_0_g_prime
        p_0_b_prime (np.array): p_0_b_prime
        c_0_b_prime (int): c_0_b_prime
    """
    with open(secret_file, "w") as f:
        f.write(
            f"{a} {b} {G} {seed} {a_hat} {b_hat} {x_0} {y_0} {g} {p_0_r_prime} {p_0_g_prime} {p_0_b_prime} {c_0_b_prime}"
        )


def encrypt_the_image(image: np.array, filename: str) -> np.array:
    """encrypt the image

    Args:
        image (np.array): pixels array
        filename (str): image file name

    Returns:
        np.array: after encrypt pixels array
    """
    # encrypt image
    encrypt_image = np.zeros_like(image)

    # generating attribute values
    a_hat, b_hat, x_0, y_0, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prime = generating_attribute_values(image)
    previous_pixel = np.array([p_0_r_prime, p_0_g_prime, p_0_b_prime])
    c_b_prime = c_0_b_prime

    # operating pixel diffusion
    r = operating_pixel_diffusion(g, a_hat, b_hat, x_0, y_0, image)

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

        # pixel scrambling using exclusive OR
        pixel = shuffling_pixel
        encrypt_pixel = pixel_scrambling_using_exclusive_or(r, pixel, previous_pixel, c_b_prime)
        previous_pixel = pixel
        c_b_prime = encrypt_pixel[-1]

        # output
        encrypt_image[row_prime][col_prime] = encrypt_pixel

    # save secret keys
    secret_keys_file = os.path.join(ENCRYPT_DIR, "{0}-{2}".format(*os.path.splitext(filename) + ("Secret-Key.txt",)))
    save_secret_keys(secret_keys_file, a_hat, b_hat, x_0, y_0, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prime)

    return encrypt_image


def main():
    # iterate over files in the directory

    for filename in os.listdir(SOURCE_DIR):
        source_file = os.path.join(SOURCE_DIR, filename)
        image = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        # encrypt the image
        encrypt_image = encrypt_the_image(image, filename)
        encrypt_file = os.path.join(ENCRYPT_DIR, "{0}_{2}{1}".format(*os.path.splitext(filename) + ("enc",)))

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
