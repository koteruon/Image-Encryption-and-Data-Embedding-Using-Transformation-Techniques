# -*- coding: utf8 -*-

import math
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
from numpy import mod

# directory hierarchy
ORIGIN_DIR = ".\\1-origin"
MARKED_DIR = ".\\2-marked"
CHANNE_DIR = ".\\3-channe"
PERMUT_DIR = ".\\4-permut"
ENCRY_DIR = ".\\5-encry"
DECRY_DIR = ".\\6-decry"
INVMUT_DIR = ".\\7-invmut"
DECOM_DIR = ".\\8-decom"
RESTOR_DIR = ".\\9-restor"
RPATAB_DIR = ".\\10-rpatab"
MESMEA_DIR = ".\\11-mesmea"
ENCPAR_DIR = ".\\12-encpar"
DECPAR_DIR = ".\\13-decpar"
MESEXT_DIR = ".\\14-mesext"
IMGRES_DIR = ".\\15-imgres"

# [PK]
SEED = [100, 200]


def read_pa_table():
    pa_tables = []
    pa_table_files = os.listdir(RPATAB_DIR)
    for filename in pa_table_files:
        matches = re.search(r"PA_(\d+)_(\d+)_\((\d+)_(\d+)_(\d+)\)_(\d+).csv", filename)
        N = int(matches.group(1))
        M = int(matches.group(2))
        W1 = int(matches.group(3))
        W2 = int(matches.group(4))
        W3 = int(matches.group(5))
        Z = int(matches.group(6))
        pa_table_param = {"N": N, "M": M, "W1": W1, "W2": W2, "W3": W3, "Z": Z}
        pa_table = {}
        df = pd.read_csv(os.path.join(RPATAB_DIR, filename), index_col=0)
        df_pa_table = df.iloc[:, [0, 2, 3, 4]]
        for _, row_data in df_pa_table.iterrows():
            if not row_data.iloc[0]:
                continue
            if not row_data.iloc[0].isdigit():
                continue
            pa_table[int(row_data.iloc[0])] = {
                "w1": int(row_data.iloc[1]),
                "w2": int(row_data.iloc[2]),
                "w3": int(row_data.iloc[3]),
            }
        pa_table_param["pa_table"] = pa_table
        pa_tables.append(pa_table_param)
    return pa_tables


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


def GWMRDH_EXT_pixel(pixel, pa_tables, pa_table_num):
    r = mod(
        pixel[0] * pa_tables[pa_table_num]["W1"]
        + pixel[1] * pa_tables[pa_table_num]["W2"]
        + pixel[2] * pa_tables[pa_table_num]["W3"],
        pa_tables[pa_table_num]["M"],
    )
    average_pixel = np.round(float(pixel[0] + pixel[1] + pixel[2]) / 3)
    pixel[0] = average_pixel
    pixel[1] = average_pixel
    pixel[2] = average_pixel
    return r


def split_into_groups(arr, group_size=3):
    # 計算能夠整除的部分
    full_groups = len(arr) // group_size
    groups = np.array_split(arr[: full_groups * group_size], full_groups)

    # 處理剩餘的部分
    if len(arr) % group_size != 0:
        remaining_elements = arr[full_groups * group_size :]
        groups.append(remaining_elements)

    return groups


def GWMRDH_EXT(pa_tables, pa_table_num, image, random_num_list):
    w, h, c = image.shape
    image_one_dim = image.flatten()
    three_groups = split_into_groups(image_one_dim)
    for index, pixel in enumerate(three_groups):
        if len(pixel) != 3:
            break
        random_num = GWMRDH_EXT_pixel(pixel, pa_tables, pa_table_num)
        random_num_list.append(str(random_num))
    stego_image = np.concatenate(three_groups).reshape((w, h, c))
    return stego_image


def read_parameters(filename):
    with open(os.path.join(DECPAR_DIR, filename + "-Secret-Key.txt")) as file:
        line = file.readline()
        values = line.split()
        a, b, c, d, M, N = values
        line = file.readline()
        G = line
    return int(a), int(b), int(c), int(d), int(M), int(N), int(G)


def write_decry_image(image, filename, pa_tables, pa_table_num, extension):
    # saving the image
    decry_file = os.path.join(
        DECRY_DIR,
        filename
        + "_decry"
        + "_N"
        + str(pa_tables[pa_table_num]["N"])
        + "_M"
        + str(pa_tables[pa_table_num]["M"])
        + "_"
        + str(pa_tables[pa_table_num]["W1"])
        + "_"
        + str(pa_tables[pa_table_num]["W2"])
        + "_"
        + str(pa_tables[pa_table_num]["W3"])
        + "_Z"
        + str(pa_tables[pa_table_num]["Z"])
        + extension,
    )
    cv2.imwrite(decry_file, image)


def write_channel_inverse_permutation(image, pk, filename, pa_tables, pa_table_num, extension):
    # 將陣列拆分成三個 (3, 3) 的陣列
    split_arrays = np.split(image, image.shape[-1], axis=-1)

    # 轉換每個子陣列的形狀
    result_images = [arr.squeeze(axis=-1) for arr in split_arrays]

    np.random.seed(pk)
    view = np.array([0, 1, 2])
    np.random.shuffle(view)
    shuffled_image = [None, None, None]
    for index, value in enumerate(view):
        shuffled_image[value] = result_images[index]
    shuffled_image = np.stack(shuffled_image, axis=-1)

    # saving the image
    encrypt_file = os.path.join(
        INVMUT_DIR,
        filename
        + "_invmut"
        + "_N"
        + str(pa_tables[pa_table_num]["N"])
        + "_M"
        + str(pa_tables[pa_table_num]["M"])
        + "_"
        + str(pa_tables[pa_table_num]["W1"])
        + "_"
        + str(pa_tables[pa_table_num]["W2"])
        + "_"
        + str(pa_tables[pa_table_num]["W3"])
        + "_Z"
        + str(pa_tables[pa_table_num]["Z"])
        + extension,
    )
    cv2.imwrite(encrypt_file, shuffled_image)

    return shuffled_image


def write_decom_image(image, filename, pa_tables, pa_table_num, extension):
    # 將陣列拆分成三個 (3, 3) 的陣列
    split_arrays = np.split(image, image.shape[-1], axis=-1)

    # 轉換每個子陣列的形狀
    result_images = [arr.squeeze(axis=-1) for arr in split_arrays]

    for result_index, result_image in enumerate(result_images):
        # saving the image
        encrypt_file = os.path.join(
            DECOM_DIR,
            filename
            + "_decom"
            + "_N"
            + str(pa_tables[pa_table_num]["N"])
            + "_M"
            + str(pa_tables[pa_table_num]["M"])
            + "_"
            + str(pa_tables[pa_table_num]["W1"])
            + "_"
            + str(pa_tables[pa_table_num]["W2"])
            + "_"
            + str(pa_tables[pa_table_num]["W3"])
            + "_Z"
            + str(pa_tables[pa_table_num]["Z"])
            + "_I"
            + str(result_index + 1)
            + extension,
        )
        cv2.imwrite(encrypt_file, result_image)


def write_restor_image(image, filename, pa_tables, pa_table_num, extension):
    # saving the image
    restor_file = os.path.join(
        RESTOR_DIR,
        filename
        + "_restor"
        + "_N"
        + str(pa_tables[pa_table_num]["N"])
        + "_M"
        + str(pa_tables[pa_table_num]["M"])
        + "_"
        + str(pa_tables[pa_table_num]["W1"])
        + "_"
        + str(pa_tables[pa_table_num]["W2"])
        + "_"
        + str(pa_tables[pa_table_num]["W3"])
        + "_Z"
        + str(pa_tables[pa_table_num]["Z"])
        + extension,
    )
    cv2.imwrite(restor_file, image)


def write_mes_ext(random_num_list, index):
    with open(os.path.join(MESEXT_DIR, f"mes_ext_{index + 1}.txt"), "w") as file:
        file.write(" ".join(random_num_list))


def write_imgres_csv(filename, ori_image, image, pa_tables, pa_table_num):
    mse = np.mean((ori_image - image) ** 2)
    if mse != 0:
        psnr = str(np.round(10 * np.log10((255**2) / mse), 3))
    else:
        psnr = "infinity"
    h, v = ori_image.shape
    ec = np.round(h * v * np.log2(pa_tables[pa_table_num]["M"]), 0)
    er = np.round((h * v * np.log2(pa_tables[pa_table_num]["M"])) / 3, 5)
    data = {
        "0": ["RPA", "Index", "MSE", "PSNR", "EC", "ER"],
        "1": [pa_tables[pa_table_num]["N"], "d", str(mse), psnr, ec, er],
        "2": [pa_tables[pa_table_num]["M"], "SE", "", "", "", ""],
        "3": ["w1", pa_tables[pa_table_num]["W1"], "", "", "", ""],
        "4": ["w2", pa_tables[pa_table_num]["W2"], "", "", "", ""],
        "5": ["w3", pa_tables[pa_table_num]["W3"], "", "", "", ""],
        "6": [pa_tables[pa_table_num]["Z"], "", "", "", "", ""],
    }
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(
            IMGRES_DIR,
            f"{filename}_qualit_N{pa_tables[pa_table_num]['N']}_M{pa_tables[pa_table_num]['M']}_{pa_tables[pa_table_num]['W1']}_{pa_tables[pa_table_num]['W2']}_{pa_tables[pa_table_num]['W3']}_Z{pa_tables[pa_table_num]['Z']}.csv",
        ),
        index=False,
        header=None,
    )


def main():
    # read parameters
    pa_tables = read_pa_table()
    encry_images = os.listdir(ENCRY_DIR)
    for file_index, filename in enumerate(encry_images):
        encry_image = cv2.imread(os.path.join(ENCRY_DIR, filename), cv2.IMREAD_UNCHANGED)
        image = encry_image.copy()
        filename, extension = os.path.splitext(filename)
        matches = re.search(r"(.+)_encry_N\d+_M\d+_\d+_\d+_\d+_Z\d+", filename)
        filename = matches.group(1)
        pa_table_num = file_index

        # IRT Decryption
        a, b, c, d, M, N, G = read_parameters(filename)
        image = repeat_transformations(M, N, a, b, c, d, image, G)
        write_decry_image(image, filename, pa_tables, pa_table_num, extension)

        # Channel Inverse Permutation
        write_channel_inverse_permutation(image, SEED[file_index], filename, pa_tables, pa_table_num, extension)

        # 解密的影像儲存成 IG1, IG2, IG3
        write_decom_image(image, filename, pa_tables, pa_table_num, extension)

        # Channel Decomposition
        image = image.astype("int16")
        random_num_list = []
        image = GWMRDH_EXT(pa_tables, pa_table_num, image, random_num_list)
        image = image.astype("uint8")

        # squeeze dimensions
        image = image.reshape(image.shape[0], image.shape[1], -1)[:, :, :1]
        image = np.squeeze(image, axis=-1)

        write_restor_image(image, filename, pa_tables, pa_table_num, extension)

        write_mes_ext(random_num_list, file_index)

        ori_image = cv2.imread(os.path.join(ORIGIN_DIR, filename + ".png"), cv2.IMREAD_UNCHANGED)

        write_imgres_csv(filename, ori_image, image, pa_tables, pa_table_num)


if __name__ == "__main__":
    main()
