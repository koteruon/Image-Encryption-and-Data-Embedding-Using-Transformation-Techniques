# -*- coding: utf8 -*-

import os
import time

import cv2
import numpy as np

# directory hierarchy
SOURCE_DIR = ".\\source"
ENCRYP_DIR = ".\\encryp"


def do_rectangular_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array) -> np.array:
    encrypt_image = np.zeros_like(image)
    MN = np.array([M, N])
    A = np.array([[a, b], [c, d]])
    for y, x in np.ndindex(image.shape[:2]):
        xy = np.array([x, y])
        xy_prime = np.remainder(np.matmul(A, xy), MN)
        x_prime, y_prime = xy_prime[0], xy_prime[1]
        encrypt_image[y_prime][x_prime] = image[y][x]
    return encrypt_image


def repeat_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array, G: int) -> np.array:
    encrypt_image = image.copy()
    for _ in range(G):
        encrypt_image = do_rectangular_transformations(M, N, a, b, c, d, encrypt_image)
    return encrypt_image


def read_images_filename():
    images_file = os.listdir(SOURCE_DIR)
    images_filename = [os.path.splitext(f)[0] for f in images_file]
    return images_filename


def read_parameters(filename: str):
    with open(os.path.join(ENCRYP_DIR, filename + "-Secret-Key.txt")) as file:
        line = file.readline()
        values = line.split()
        a, b, c, d, M, N = values
        line = file.readline()
        G = line
    return int(a), int(b), int(c), int(d), int(M), int(N), int(G)


def main():
    # read parameters
    images_filename = read_images_filename()
    for filename in images_filename:
        image = cv2.imread(os.path.join(SOURCE_DIR, filename + ".png"), cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        a, b, c, d, M, N, G = read_parameters(filename)
        start_time = time.time()
        encrypt_image = repeat_transformations(M, N, a, b, c, d, image, G)
        end_time = time.time()

        # RGB color
        if is_rgb:
            encrypt_image = cv2.cvtColor(encrypt_image, cv2.COLOR_RGB2BGR)
        else:
            # squeeze dimensions
            encrypt_image = np.squeeze(encrypt_image, axis=-1)

        # saving the image
        encrypt_file = os.path.join(ENCRYP_DIR, filename + "_enc.png")
        cv2.imwrite(encrypt_file, encrypt_image)

        encrypt_time = os.path.join(ENCRYP_DIR, filename + "_enc_time.txt")
        with open(encrypt_time, "w+") as f:
            f.write(f"{G}")
            f.write("\n")
            f.write(f"{end_time - start_time:.2f}")


if __name__ == "__main__":
    main()
