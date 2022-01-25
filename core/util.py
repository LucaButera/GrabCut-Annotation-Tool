#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy

import cv2 as cv
import numpy as np
from PIL import Image


def execute_grabcut(
        image,
        mask,
        bgd_model,
        fgd_model,
        iteration,
        mask_alpha,
        mask_beta,
        roi=None,
):
    mask, bgd_model, fgd_model = cv.grabCut(image, mask, roi, bgd_model,
                                            fgd_model, iteration,
                                            cv.GC_INIT_WITH_RECT if roi is not None else cv.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    debug_image = image * mask[:, :, np.newaxis]
    debug_image = cv.addWeighted(debug_image, mask_alpha, image, mask_beta, 0)
    return mask, bgd_model, fgd_model, debug_image


def get_palette():
    palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
               [128, 64, 128]]
    return np.asarray(palette)


def save_index_color_png(output_path, filename, mask_image):
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(output_path, base_filename + '_mask.png')

    color_palette = get_palette().flatten()
    color_palette = color_palette.tolist()
    with Image.fromarray(mask_image, mode="P") as png_image:
        #png_image.putpalette(color_palette)
        png_image.save(save_path)


def save_resize_image(output_path, filename, image):
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(output_path, base_filename + '.png')
    cv.imwrite(save_path, image)


def save_image_and_mask(
        output_image_path,
        image,
        output_annotation_path,
        mask,
        image_file_path,
        output_size,
        class_id,
):
    debug_mask = copy.deepcopy(mask)
    temp_mask = copy.deepcopy(mask)
    debug_mask = np.where((temp_mask == 2) | (temp_mask == 0), debug_mask, 0).astype('uint8')

    resize_image = cv.resize(image, output_size)
    resize_mask = cv.resize(debug_mask, output_size)

    image_file_path = f'{image_file_path.split(".")[0]}_{class_id}.{image_file_path.split(".")[1]}'
    save_index_color_png(output_annotation_path, image_file_path, resize_mask)
    save_resize_image(output_image_path, image_file_path, resize_image)
