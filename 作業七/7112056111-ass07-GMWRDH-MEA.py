# -*- coding: utf8 -*-

import os
import random
import re

import cv2
import numpy as np
import pandas as pd
from numpy import mod

# directory hierarchy
IMGRES_DIR = ".\\imgres"
MARKED_DIR = ".\\marked"
MESEXT_DIR = ".\\mesext"
MESMEA_DIR = ".\\mesmea"
ORIGIN_DIR = ".\\origin"
RESTOR_DIR = ".\\restor"
RPATAB_DIR = ".\\rpatab"

SEED = [100, 200, 300, 400]


def split_into_groups(arr, group_size=3):
    # 計算能夠整除的部分
    full_groups = len(arr) // group_size
    groups = np.array_split(arr[: full_groups * group_size], full_groups)

    # 處理剩餘的部分
    if len(arr) % group_size != 0:
        remaining_elements = arr[full_groups * group_size :]
        groups.append(remaining_elements)

    return groups


def GWMRDH_pixel(pixel, pa_tables, pa_table_num, random_num):
    r = mod(
        pixel[0] * pa_tables[pa_table_num]["W1"]
        + pixel[1] * pa_tables[pa_table_num]["W2"]
        + pixel[2] * pa_tables[pa_table_num]["W3"],
        pa_tables[pa_table_num]["M"],
    )
    d = mod(random_num - r, pa_tables[pa_table_num]["M"])
    pixel[0] += pa_tables[pa_table_num]["pa_table"][d]["w1"]
    pixel[1] += pa_tables[pa_table_num]["pa_table"][d]["w2"]
    pixel[2] += pa_tables[pa_table_num]["pa_table"][d]["w3"]


def write_mes_mea(random_num_list, index):
    with open(os.path.join(MESMEA_DIR, f"mes_mea_{index + 1}.txt"), "w") as file:
        file.write(" ".join(random_num_list))


def GWMRDH(pa_tables, pa_table_num, image, random_num_list, ramdom_seed):
    random.seed(ramdom_seed)
    w, h, c = image.shape
    image_one_dim = image.flatten()
    three_groups = split_into_groups(image_one_dim)
    for index, pixel in enumerate(three_groups):
        if len(pixel) != 3:
            break
        random_num = random.randrange(0, pa_tables[pa_table_num]["M"])
        random_num_list.append(str(random_num))
        origin_pixel = pixel.copy()
        GWMRDH_pixel(pixel, pa_tables, pa_table_num, random_num)
        if check_overflow(origin_pixel, pixel, pa_tables[pa_table_num]["Z"]):
            pixel[0] = origin_pixel[0]
            pixel[1] = origin_pixel[1]
            pixel[2] = origin_pixel[2]
            GWMRDH_pixel(pixel, pa_tables, pa_table_num, random_num)
        # Message Extraction
        if random_num != mod(
            pixel[0] * pa_tables[pa_table_num]["W1"]
            + pixel[1] * pa_tables[pa_table_num]["W2"]
            + pixel[2] * pa_tables[pa_table_num]["W3"],
            pa_tables[pa_table_num]["M"],
        ):
            raise Exception
    stego_image = np.concatenate(three_groups).reshape((w, h, c))
    return stego_image


def check_overflow(origin_pixel, pixel, Z):
    recalculate = False
    if pixel[0] > 255 or pixel[1] > 255 or pixel[2] > 255:
        origin_pixel[0] = 255 - Z
        origin_pixel[1] = 255 - Z
        origin_pixel[2] = 255 - Z
        recalculate = True
    if pixel[0] < 0 or pixel[1] < 0 or pixel[2] < 0:
        origin_pixel[0] = Z
        origin_pixel[1] = Z
        origin_pixel[2] = Z
        recalculate = True
    return recalculate


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


def main():
    # read parameters
    pa_tables = read_pa_table()
    origin_images = os.listdir(ORIGIN_DIR)
    for file_index, filename in enumerate(origin_images):
        pa_table_num = file_index
        filename, extension = os.path.splitext(filename)
        ori_image = cv2.imread(os.path.join(ORIGIN_DIR, filename + extension), cv2.IMREAD_UNCHANGED)

        # expand dimensions
        ori_image = np.tile(
            ori_image[:, :, np.newaxis],
            (1, 1, 3),
        )
        ori_image = ori_image.astype("int16")

        random_num_list = []
        image = ori_image.copy()
        image = GWMRDH(pa_tables, pa_table_num, image, random_num_list, SEED[file_index])
        image = image.astype("uint8")

        # 將陣列拆分成三個 (3, 3) 的陣列
        split_arrays = np.split(image, image.shape[-1], axis=-1)

        # 轉換每個子陣列的形狀
        result_image = [arr.squeeze(axis=-1) for arr in split_arrays]

        for result_index, result_image in enumerate(result_image):
            # saving the image
            encrypt_file = os.path.join(
                MARKED_DIR,
                filename
                + "_mark"
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

        write_mes_mea(random_num_list, file_index)


if __name__ == "__main__":
    main()
