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


def split_into_groups(arr, group_size=3):
    # 計算能夠整除的部分
    full_groups = len(arr) // group_size
    groups = np.array_split(arr[: full_groups * group_size], full_groups)

    # 處理剩餘的部分
    if len(arr) % group_size != 0:
        remaining_elements = arr[full_groups * group_size :]
        groups.append(remaining_elements)

    return groups


def GWMRDH_pixel(pixel, pa_tables, pa_table_num):
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


def write_mes_ext(random_num_list, index):
    with open(os.path.join(MESEXT_DIR, f"mes_ext_{index + 1}.txt"), "w") as file:
        file.write(" ".join(random_num_list))


def GWMRDH(pa_tables, pa_table_num, image, random_num_list):
    w, h, c = image.shape
    image_one_dim = image.flatten()
    three_groups = split_into_groups(image_one_dim)
    for index, pixel in enumerate(three_groups):
        if len(pixel) != 3:
            break
        random_num = GWMRDH_pixel(pixel, pa_tables, pa_table_num)
        random_num_list.append(str(random_num))
    stego_image = np.concatenate(three_groups).reshape((w, h, c))
    return stego_image


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
    mark_images = os.listdir(MARKED_DIR)
    mark_images_dict = {}
    for filename in mark_images:
        matches = re.search(r"(.+)_mark_N\d+_M\d+_\d+_\d+_\d+_Z\d+_I\d+.png", filename)
        if matches.group(1) in mark_images_dict:
            mark_images_dict[matches.group(1)].append(filename)
        else:
            mark_images_dict[matches.group(1)] = [filename]
    for file_index, (filename, filename_list) in enumerate(mark_images_dict.items()):
        pa_table_num = file_index
        stego_image_1 = cv2.imread(os.path.join(MARKED_DIR, filename_list[0]), cv2.IMREAD_UNCHANGED)
        stego_image_2 = cv2.imread(os.path.join(MARKED_DIR, filename_list[1]), cv2.IMREAD_UNCHANGED)
        stego_image_3 = cv2.imread(os.path.join(MARKED_DIR, filename_list[2]), cv2.IMREAD_UNCHANGED)

        stego_image = np.stack((stego_image_1, stego_image_2, stego_image_3), axis=-1)
        stego_image = stego_image.astype("int16")

        random_num_list = []

        image = stego_image.copy()
        image = GWMRDH(pa_tables, pa_table_num, image, random_num_list)
        image = image.astype("uint8")

        # squeeze dimensions
        image = image.reshape(image.shape[0], image.shape[1], -1)[:, :, :1]
        image = np.squeeze(image, axis=-1)

        # saving the image
        encrypt_file = os.path.join(
            RESTOR_DIR,
            filename
            + "_rest"
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
            + ".png",
        )
        cv2.imwrite(encrypt_file, image)

        write_mes_ext(random_num_list, file_index)

        ori_image = cv2.imread(os.path.join(ORIGIN_DIR, filename + ".png"), cv2.IMREAD_UNCHANGED)

        write_imgres_csv(filename, ori_image, image, pa_tables, pa_table_num)


if __name__ == "__main__":
    main()
