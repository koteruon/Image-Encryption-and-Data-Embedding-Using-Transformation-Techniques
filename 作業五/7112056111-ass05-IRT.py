# -*- coding: utf8 -*-

import math
import os
import time

import cv2
import numpy as np

# directory hierarchy
SOURCE_DIR = ".\\source"
ENCRYP_DIR = ".\\encryp"
DECRYP_DIR = ".\\decryp"


def do_inverse_rectangular_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array) -> np.array:
    decrypt_image = np.zeros_like(image)

    p = math.gcd(M, N)
    h = int(M / p)
    v = int(N / p)
    t = a * d - b * c
    S = euler_method(t, 1, p)
    S = S[0]

    Stp_matrix = np.array([[d, (p - 1) * b], [(p - 1) * c, a]])
    p_matrix = np.array([p, p])
    for y_prime, x_prime in np.ndindex(image.shape[:2]):
        # if row_prime == 1 and col_prime == 2:
        #     pass
        # g1
        xy_prime = np.array([x_prime, y_prime])
        xy_p = np.remainder(np.matmul(S * Stp_matrix, xy_prime), p_matrix)
        x_p, y_p = xy_p[0], xy_p[1]
        # g2
        H = ((x_prime - a * x_p - b * y_p) / p) + math.ceil((a * p) / h) * h + math.ceil((b * p) / h) * h
        V = ((y_prime - c * x_p - d * y_p) / p) + math.ceil((c * p) / v) * v + math.ceil((d * p) / v) * v
        # g3
        S_ah = euler_method(a, 1, h)[0]
        S_dv = euler_method(d, 1, v)[0]
        if b % h == 0:
            x_h = (S_ah * H) % h
            y_v = (S_dv * (V + math.ceil((c * h) / v) * v - c * x_h)) % v
        elif c % v == 0:
            y_v = (S_dv * V) % v
            x_h = (S_ah * (H + math.ceil((b * v) / h) * h - b * y_v)) % h
        # g4
        x = x_p + p * x_h
        y = y_p + p * y_v
        decrypt_image[int(y)][int(x)] = image[y_prime][x_prime]
    return decrypt_image


def repeat_transformations(M: int, N: int, a: int, b: int, c: int, d: int, image: np.array, G: int) -> np.array:
    encrypt_image = image.copy()
    for period in range(G):
        encrypt_image = do_inverse_rectangular_transformations(M, N, a, b, c, d, encrypt_image)
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


def euler_method(a, b, m):
    gcd_am = math.gcd(a, m)

    if b % gcd_am == 0:
        a_bar = a // gcd_am
        b_bar = b // gcd_am
        m_bar = m // gcd_am

        inv_a = pow(a_bar, -1, m_bar)

        x0 = (b_bar * inv_a) % m_bar
        return [x0 + k * (m // gcd_am) for k in range(gcd_am)]
    else:
        return None


def main():
    # read parameters
    images_filename = read_images_filename()
    for filename in images_filename:
        image = cv2.imread(os.path.join(ENCRYP_DIR, filename + "_enc.png"), cv2.IMREAD_UNCHANGED)
        is_rgb = True if image.ndim == 3 else False

        # RGB color
        if is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # expand dimensions
            image = np.expand_dims(image, axis=-1)

        a, b, c, d, M, N, G = read_parameters(filename)
        start_time = time.time()
        decrypt_image = repeat_transformations(M, N, a, b, c, d, image, G)
        end_time = time.time()

        # RGB color
        if is_rgb:
            decrypt_image = cv2.cvtColor(decrypt_image, cv2.COLOR_RGB2BGR)
        else:
            # squeeze dimensions
            decrypt_image = np.squeeze(decrypt_image, axis=-1)

        # saving the image
        encrypt_file = os.path.join(DECRYP_DIR, filename + "_dec.png")
        cv2.imwrite(encrypt_file, decrypt_image)

        decrypt_time = os.path.join(DECRYP_DIR, filename + "_dec_time.txt")
        with open(decrypt_time, "w+") as f:
            f.write(f"{G}")
            f.write("\n")
            f.write(f"{end_time - start_time:.2f}")


if __name__ == "__main__":
    main()
