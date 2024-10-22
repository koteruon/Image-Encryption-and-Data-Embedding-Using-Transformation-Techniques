# -*- coding: utf8 -*-

import hashlib
import math
import os
import queue

import cv2
import numpy as np

# eat cycle numbers
EAT_CYCLE_NUMBERS = {512: 384}

# directory hierarchy
SOURCE_DIR = ".\\source"
ENCRYPT_DIR = ".\\encryp"
DECRYPT_DIR = ".\\decryp"


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


def pixel_de_scrambling_using_exclusive_or(r: queue.Queue, c_plus_1: np.array, p: np.array, c_b_prime: int) -> np.array:
    """pixel de-scrambling using exclusive or

    Args:
        r (queue.Queue): random decimal sequence
        c_plus_1 (np.array): encrypt image pixel
        p (np.array): previous plaintext image pixel
        c_b_prime (int): previous encrypted pixel

    Returns:
        np.array: encrypted pixel
    """
    p_plus_1 = c_plus_1.copy()
    if c_plus_1.size == 1:
        p_plus_1_prime = r.get() ^ c_plus_1[0] ^ p[0] ^ c_b_prime
        p_plus_1[0] = p_plus_1_prime
    else:
        # rgb
        p_plus_1_r_prime = r.get() ^ c_plus_1[0] ^ p[0] ^ c_b_prime
        p_plus_1_g_prime = r.get() ^ c_plus_1[1] ^ p[1] ^ c_plus_1[0]
        p_plus_1_b_prime = r.get() ^ c_plus_1[2] ^ p[2] ^ c_plus_1[1]
        p_plus_1[0], p_plus_1[1], p_plus_1[2] = p_plus_1_r_prime, p_plus_1_g_prime, p_plus_1_b_prime
    return p_plus_1


def read_secret_keys(secret_file: str):
    """read secret keys

    Args:
        secret_file (str): secret file path
    """
    with open(secret_file, "r") as f:
        secret_keys = f.read()
        (
            a,
            b,
            G,
            seed,
            a_hat,
            b_hat,
            x_0,
            y_0,
            g,
            p_0_r_prime,
            p_0_g_prime,
            p_0_b_prime,
            c_0_b_prime,
        ) = secret_keys.split()
    a = int(a)
    b = int(b)
    G = int(G)
    seed = int(seed)
    a_hat = float(a_hat)
    b_hat = float(b_hat)
    x_0 = float(x_0)
    y_0 = float(y_0)
    g = int(g)
    p_0_r_prime = int(p_0_r_prime)
    p_0_g_prime = int(p_0_g_prime)
    p_0_b_prime = int(p_0_b_prime)
    c_0_b_prime = int(c_0_b_prime)
    return a, b, G, seed, a_hat, b_hat, x_0, y_0, g, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prime


def decrypt_the_image(image: np.array, origin_filename: str) -> np.array:
    """decrypt the image

    Args:
        image (np.array): pixels array
        origin_filename (str): image file name

    Returns:
        np.array: after decrypt pixels array
    """
    # decrypt image
    decrypt_image = np.zeros_like(image)

    # read secret keys
    secret_keys_file = os.path.join(
        ENCRYPT_DIR, "{0}-{2}".format(*os.path.splitext(origin_filename) + ("Secret-Key.txt",))
    )
    a, b, G, seed, a_hat, b_hat, x_0, y_0, g, p_0_r_prime, p_0_g_prime, p_0_b_prime, c_0_b_prime = read_secret_keys(
        secret_keys_file
    )
    previous_pixel = np.array([p_0_r_prime, p_0_g_prime, p_0_b_prime])
    c_b_prime = c_0_b_prime

    # operating pixel diffusion
    r = operating_pixel_diffusion(g, a_hat, b_hat, x_0, y_0, image)

    # iterate every bit in the image
    for row_prime, col_prime in np.ndindex(image.shape[:2]):
        # do reverse EVA
        xy_prime = np.array([row_prime, col_prime])
        N = image.shape[0]
        xy = reverse_equilateral_arnold_transform(a, b, xy_prime, N, G)
        row, col = xy[0], xy[1]

        # output
        decrypt_image[row][col] = image[row_prime][col_prime]

    # iterate every bit in the image
    for row, col in np.ndindex(image.shape[:2]):
        # pixel scrambling using exclusive OR
        pixel = decrypt_image[row][col]
        decrypt_pixel = pixel_de_scrambling_using_exclusive_or(r, pixel, previous_pixel, c_b_prime)
        previous_pixel = decrypt_pixel
        c_b_prime = pixel[-1]

        # do Durstenfeld PR
        bit_shuffling_pixels = np.unpackbits(decrypt_pixel)
        bit_pixels = reverse_durstenfeld_random_permutation(seed, bit_shuffling_pixels)
        pixel = np.packbits(bit_pixels)

        # output
        decrypt_image[row][col] = pixel
    return decrypt_image


def mean_square_error(source_image_file_path: str, decrypt_image_file_path: str):
    """mean square error

    Args:
        source_image_file_path (str): source path
        decrypt_image_file_path (str): decrypt path
    """
    source_image_file_name = os.path.basename(source_image_file_path)
    decrypt_image_file_name = os.path.basename(decrypt_image_file_path)
    source_image = cv2.imread(source_image_file_path, cv2.IMREAD_UNCHANGED)
    decrypt_image = cv2.imread(decrypt_image_file_path, cv2.IMREAD_UNCHANGED)
    err = (np.square(source_image - decrypt_image)).mean()
    if err == 0.0:
        print(f"{source_image_file_name} and {decrypt_image_file_name} are the same.")
    else:
        print(f"{source_image_file_name} and {decrypt_image_file_name} are different.")


def main():
    # iterate over files in the directory
    image_files = (filename for filename in os.listdir(ENCRYPT_DIR) if filename.endswith(".png"))
    for filename in image_files:
        encrypt_file = os.path.join(ENCRYPT_DIR, filename)
        image = cv2.imread(encrypt_file, cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        # decrypt the image
        origin_filename = filename.replace("_enc", "")
        decrypt_image = decrypt_the_image(image, origin_filename)
        decrypt_file = os.path.join(DECRYPT_DIR, "{0}_{2}{1}".format(*os.path.splitext(origin_filename) + ("dec",)))

        # RGB color
        if is_rgb:
            decrypt_image = cv2.cvtColor(decrypt_image, cv2.COLOR_RGB2BGR)
        else:
            # squeeze dimensions
            decrypt_image = np.squeeze(decrypt_image, axis=-1)

        # saving the image
        cv2.imwrite(decrypt_file, decrypt_image)

        # MSE
        source_file = os.path.join(SOURCE_DIR, origin_filename)
        mean_square_error(source_file, decrypt_file)


if __name__ == "__main__":
    main()
