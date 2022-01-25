#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import gc
import glob
import json
import os
import re
from pathlib import Path

import click
import cv2 as cv
import numpy as np
from PIL import Image

import core.util as util
from core.gui import AppGui

image_list, mask_list, debug_image_list = [], [], []
bgd_model_list, fgd_model_list = [], []
prev_class_id = -1


def initialize_grabcut_list(image, mask):
    global image_list, mask_list, bgd_model_list, fgd_model_list, \
        debug_image_list

    image_list = copy.deepcopy(image)
    debug_image_list = copy.deepcopy(image)
    mask_list = copy.deepcopy(mask)
    bgd_model_list = np.zeros((1, 65), dtype=np.float64)
    fgd_model_list = np.zeros((1, 65), dtype=np.float64)
    gc.collect()


def load_mask_image(output_annotation_path, mask_filename):
    filename = os.path.splitext(os.path.basename(mask_filename))[0]
    mask_file_path = os.path.join(output_annotation_path, filename + '_mask.png')
    if os.path.exists(mask_file_path):
        pil_image = Image.open(mask_file_path)
        mask = np.asarray(pil_image).astype('uint8')
        return mask


def draw_roi_mode_image(image, roi=None):
    debug_image = copy.deepcopy(image)
    cv.putText(debug_image, "Select ROI", (5, 25), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(debug_image, "Select ROI", (5, 25), cv.FONT_HERSHEY_SIMPLEX,
               0.9, (103, 82, 51), 1, cv.LINE_AA)
    if roi is not None:
        cv.rectangle(
            debug_image,
            (roi[0], roi[1]),
            (roi[2], roi[3]),
            (255, 255, 255),
            thickness=3,
        )
        cv.rectangle(
            debug_image,
            (roi[0], roi[1]),
            (roi[2], roi[3]),
            (103, 82, 51),
            thickness=2,
        )
    return debug_image


# Draw overlay mask in grabcut mode
def draw_grabcut_mode_image(
        image,
        color,
        mask,
        mask_color,
        point01=None,
        point02=None,
        thickness=4,
):
    debug_image = copy.deepcopy(image)
    debug_mask = copy.deepcopy(mask)

    if point01 is not None and point02 is not None:
        cv.line(debug_image, point01, point02, color, thickness)
        cv.line(debug_mask, point01, point02, mask_color, thickness)

    return debug_image, debug_mask


def draw_processing_image(image):
    image_width, image_height = image.shape[1], image.shape[0]

    # Processing wait display image
    loading_image = copy.deepcopy(image)
    loading_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    loading_image = loading_image * loading_mask[:, :, np.newaxis]
    loading_image = cv.addWeighted(loading_image, 0.7, image, 0.3, 0)
    cv.putText(loading_image, "PROCESSING...",
               (int(image_width / 2) - (6 * 18), int(image_height / 2)),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv.LINE_AA)
    cv.putText(loading_image, "PROCESSING...",
               (int(image_width / 2) - (6 * 18), int(image_height / 2)),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (103, 82, 51), 2, cv.LINE_AA)

    return loading_image


# マウスのドラッグ開始点/終了点を取得
def get_mouse_start_end_point(appgui, mosue_info):
    mouse_event = mosue_info[0]
    mouse_start_point = mosue_info[1]
    mouse_end_point = mosue_info[2]
    mouse_prev_point = mosue_info[3]

    mouse_event, mouse_point = appgui.read_mouse_event()

    if mouse_event == appgui.MOUSE_EVENT_DRAG_START:
        mouse_start_point = mouse_point
        mouse_prev_point = mouse_point
    elif mouse_event == appgui.MOUSE_EVENT_DRAG:
        mouse_prev_point = mouse_end_point
        mouse_end_point = mouse_point
    elif mouse_event == appgui.MOUSE_EVENT_DRAG_END:
        mouse_prev_point = mouse_end_point
        mouse_end_point = mouse_point
    elif mouse_event == appgui.MOUSE_EVENT_NONE:
        mouse_start_point = None
        mouse_end_point = None
        mouse_prev_point = None

    return (mouse_event, mouse_start_point, mouse_end_point, mouse_prev_point)


# Select ROI MODE
def process_select_roi_mode(
        appgui,
        mosue_info,
        image,
        debug_image,
        mask,
        bgd_model,
        fgd_model,
):
    # Get mouse info
    mouse_event = mosue_info[0]
    mouse_start_point = mosue_info[1]
    mouse_end_point = mosue_info[2]

    # Get GUI info
    mask_alpha = appgui.get_setting_mask_alpha()
    mask_beta = 1 - mask_alpha
    iteration = appgui.get_setting_iteration()

    roi = None
    grabcut_execute = False

    # Get ROI
    if mouse_start_point is not None and mouse_end_point is not None:
        min_x = (mouse_start_point[0]) if (
                mouse_start_point[0] < mouse_end_point[0]) else (
            mouse_end_point[0])
        man_x = (mouse_start_point[0]) if (
                mouse_start_point[0] > mouse_end_point[0]) else (
            mouse_end_point[0])
        min_y = (mouse_start_point[1]) if (
                mouse_start_point[1] < mouse_end_point[1]) else (
            mouse_end_point[1])
        man_y = (mouse_start_point[1]) if (
                mouse_start_point[1] > mouse_end_point[1]) else (
            mouse_end_point[1])
        roi = [min_x, min_y, man_x, man_y]

    # Draw roi mode
    if mouse_event == appgui.MOUSE_EVENT_DRAG:
        debug_image = draw_roi_mode_image(image)

    # Draw roi
    if mouse_event == appgui.MOUSE_EVENT_DRAG:
        debug_image = draw_roi_mode_image(image, roi)

    # Get ROI and process
    if mouse_event == appgui.MOUSE_EVENT_DRAG_END:
        # Show wait process image
        loading_image = draw_processing_image(image)
        appgui.draw_image(loading_image)
        appgui.read_window(timeout=100)

        # execute GrabCut
        mask, bgd_model, fgd_model, debug_image = util.execute_grabcut(
            image,
            mask,
            bgd_model,
            fgd_model,
            iteration,
            mask_alpha,
            mask_beta,
            roi,
        )

        # Set GrabCut mode
        appgui.mode = appgui.GRABCUT_MODE

        # Set Drawing color
        label_background = appgui.get_setting_label_background()
        if label_background:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        cv.rectangle(debug_image, (0, 0), (511, 511), color=color, thickness=3)
        grabcut_execute = True

    # Draw step results
    appgui.draw_image(debug_image)
    appgui.draw_mask_image(mask)

    return grabcut_execute, mask, bgd_model, fgd_model, debug_image


# GrabCut MODE
def process_grabcut_mode(
        appgui,
        mosue_info,
        image,
        debug_image,
        mask,
        bgd_model,
        fgd_model,
):
    # Get mouse info
    mouse_event = mosue_info[0]
    mouse_end_point = mosue_info[2]
    mouse_prev_point = mosue_info[3]

    # Get GUI info
    mask_alpha = appgui.get_setting_mask_alpha()
    mask_beta = 1 - mask_alpha
    iteration = appgui.get_setting_iteration()
    thickness = appgui.get_setting_draw_thickness()
    label_background = appgui.get_setting_label_background()

    grabcut_execute = False

    if label_background:
        color = (0, 0, 0)
        manually_label_value = 0
    else:
        color = (255, 255, 255)
        manually_label_value = 1

    # Draw mask manually in grabcut mode
    if mouse_event == appgui.MOUSE_EVENT_DRAG_START or \
            mouse_event == appgui.MOUSE_EVENT_DRAG:
        debug_image, mask = draw_grabcut_mode_image(
            debug_image,
            color,
            mask,
            manually_label_value,
            point01=mouse_prev_point,
            point02=mouse_end_point,
            thickness=thickness,
        )

    # Update image
    if mouse_event == appgui.MOUSE_EVENT_DRAG_END:
        # Show wait processing image
        loading_image = draw_processing_image(image)
        appgui.draw_image(loading_image)
        appgui.read_window(timeout=100)

        # execute GrabCut
        mask, bgd_model, fgd_model, debug_image = util.execute_grabcut(
            image,
            mask,
            bgd_model,
            fgd_model,
            iteration,
            mask_alpha,
            mask_beta,
        )
        cv.rectangle(debug_image, (0, 0), (511, 511), color=color, thickness=3)
        grabcut_execute = True

    # Draw step results
    appgui.draw_image(debug_image)
    appgui.draw_mask_image(mask)

    return grabcut_execute, mask, bgd_model, fgd_model, debug_image


def get_event_kind(event):
    if event.startswith('Up') or event.startswith('p'):
        event_kind = 'Up'
    elif event.startswith('Down') or event.startswith('n'):
        event_kind = 'Down'
    elif event.startswith('s'):
        event_kind = 's'
    elif event.startswith('Control'):
        event_kind = 'Control'
    elif event.startswith('Escape'):
        event_kind = 'Escape'
    elif event.startswith('-CLASS ID-'):
        event_kind = 'Class ID'
    else:
        event_kind = event
    return event_kind


# イベントハンドラー：ファイルリスト選択
def event_handler_file_select(event_kind, appgui, scroll_count=0):
    global mask_list, output_annotation_path

    # インデックス位置を計算
    listbox_size = appgui.get_listbox_size()
    currrent_index = appgui.get_file_list_current_index()
    currrent_index = (currrent_index + scroll_count) % listbox_size

    # インデックス位置へリストを移動
    if scroll_count == 0:
        appgui.set_file_list_current_index(currrent_index, False)
    else:
        appgui.set_file_list_current_index(currrent_index, True)

    # 画像読み込み
    file_path = appgui.get_file_path_from_listbox(currrent_index)
    image = cv.imread(file_path)
    resize_image = cv.resize(image, (512, 512))
    mask = np.zeros(resize_image.shape[:2], dtype=np.uint8)

    # 初期描画
    initialize_grabcut_list(resize_image, mask)

    # 既存のマスクファイルを確認し、存在すれば読み込む
    mask_list = load_mask_image(output_annotation_path, file_path)

    debug_image = draw_roi_mode_image(resize_image)
    appgui.draw_image(debug_image)
    appgui.draw_mask_image(mask_list)

    # 設定リセット
    appgui.set_setting_lable_background(True)

    # ROI選択モード(ROI_MODE)に遷移
    appgui.mode = appgui.ROI_MODE


# イベントハンドラー：ファイルリスト選択(キーアップ)
def event_handler_file_select_up(event_kind, appgui):
    event_handler_file_select(event_kind, appgui, scroll_count=-1)


# イベントハンドラー：ファイルリスト選択(キーダウン)
def event_handler_file_select_down(event_kind, appgui):
    event_handler_file_select(event_kind, appgui, scroll_count=1)


# イベントハンドラー：前景/後景指定選択
def event_handler_change_manually_label(event_kind, appgui):
    global debug_image_list

    label_background = not appgui.get_setting_label_background()
    appgui.set_setting_lable_background(label_background)
    label_background = appgui.get_setting_label_background()

    # 前景/背景情報提示
    if appgui.mode == appgui.GRABCUT_MODE:
        class_id = appgui.get_setting_class_id()
        label_background = appgui.get_setting_label_background()
        if label_background:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv.rectangle(debug_image_list, (0, 0), (511, 511),
                     color=color,
                     thickness=3)

        appgui.draw_image(debug_image_list)


# Change config handler
def event_handler_change_config(
        event_kind,
        appgui,
        config_file_name='config.json',
):
    config_data = {
        'MASK ALPHA': appgui.get_setting_mask_alpha(),
        'ITERATION': appgui.get_setting_iteration(),
        'DRAW THICKNESS': appgui.get_setting_draw_thickness(),
        'OUTPUT WIDTH': appgui.get_setting_output_width(),
        'OUTPUT HEIGHT': appgui.get_setting_output_height()
    }

    auto_save = appgui.get_setting_auto_save()
    if auto_save:
        config_data['AUTO SAVE'] = 1
    else:
        config_data['AUTO SAVE'] = 0

    with open(config_file_name, mode='wt', encoding='utf-8') as file:
        json.dump(config_data, file, ensure_ascii=False, indent=4)


# NOP
def event_handler_change_nop(event_kind, appgui):
    pass


# Event handlers
def get_event_handler_list():
    event_handler = {
        '-IMAGE ORIGINAL-': event_handler_change_nop,
        '-IMAGE ORIGINAL-+UP': event_handler_change_nop,
        '-LISTBOX FILE-': event_handler_file_select,
        '-SPIN MASK ALPHA-': event_handler_change_config,
        '-SPIN ITERATION-': event_handler_change_config,
        '-SPIN DRAW THICKNESS-': event_handler_change_config,
        '-SPIN OUTPUT WIDTH-': event_handler_change_config,
        '-SPIN OUTPUT HEIGHT-': event_handler_change_config,
        '-CHECKBOX AUTO SAVE-': event_handler_change_config,
        '-CLASS ID-': event_handler_change_nop,
        'Up': event_handler_file_select_up,
        'Down': event_handler_file_select_down,
        's': event_handler_change_nop,
        'Control': event_handler_change_manually_label,
        'Escape': event_handler_change_nop,
    }

    return event_handler


@click.command()
@click.option("--image-input", "--input", "-i",
              type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
              default='input', help='Path where to look for images.')
@click.option("--image-output", "--io",
              type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
              default='output/image', help='Path where images are stored.')
@click.option("--annotation-output", "--ao",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default='output/annotation', help='Path where masks are stored.')
@click.option("--config", "-c",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default='config.json', help='Path to app configuration.')
def main(image_input: Path, image_output: Path, annotation_output: Path, config: Path):
    global image_list, mask_list, debug_image_list, bgd_model_list, \
        fgd_model_list, output_annotation_path, prev_class_id

    input_path = os.path.join(str(image_input), '*')
    output_image_path = str(image_output)
    output_annotation_path = str(annotation_output)
    config_file_name = str(config)

    # Find eligible files
    file_paths = sorted([
        p for p in glob.glob(input_path)
        if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))
    ])

    # Configure GUI
    appgui = AppGui(file_paths)
    _ = appgui.load_config(config_file_name)
    grabcut_image_size = (512, 512)

    # Initialize selected image
    current_index = appgui.get_file_list_current_index()
    image = cv.imread(file_paths[current_index])
    resize_image = cv.resize(image, grabcut_image_size)
    debug_image = draw_roi_mode_image(resize_image)
    mask = np.zeros(resize_image.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # Load mask if present
    existing_mask = load_mask_image(output_annotation_path,
                                    file_paths[current_index])
    if existing_mask is not None:
        mask = existing_mask

    # Draw
    appgui.draw_image(debug_image)
    appgui.draw_mask_image(mask)

    # Initialize mouse event variables
    mouse_event = None
    mouse_start_point, mouse_end_point, mouse_prev_point = None, None, None
    mouse_info = [
        mouse_event,
        mouse_start_point,
        mouse_end_point,
        mouse_prev_point,
    ]

    # Set ROI MODE
    appgui.mode = appgui.ROI_MODE

    process_func = [
        process_select_roi_mode,  # ROI_MODE
        process_grabcut_mode,  # GRABCUT_MODE
    ]

    # Get Class ID
    prev_class_id = appgui.get_setting_class_id()

    # Get event handlers
    event_handler_list = get_event_handler_list()

    while True:
        event, _ = appgui.read_window()
        auto_save = appgui.get_setting_auto_save()
        class_id = appgui.get_setting_class_id()
        output_width = appgui.get_setting_output_width()
        output_height = appgui.get_setting_output_height()

        # Get mouse info
        mouse_info = get_mouse_start_end_point(
            appgui,
            mouse_info,
        )

        # Process based on active mode
        grabcut_exec, mask, bgd_model, fgd_model, debug_image = process_func[
            appgui.mode](
            appgui,
            mouse_info,
            resize_image,
            debug_image,
            mask,
            bgd_model,
            fgd_model,
        )

        # GrubCut execution
        if grabcut_exec:
            # Eventually save
            current_index = appgui.get_file_list_current_index()
            file_path = appgui.get_file_path_from_listbox(current_index)
            if auto_save:
                util.save_image_and_mask(
                    output_image_path,
                    image,
                    output_annotation_path,
                    mask,
                    file_path,
                    (output_width, output_height),
                    class_id
                )

            # Reset mouse info
            mouse_info = [None, None, None, None]

        # Get new event
        event_kind = get_event_kind(event)
        # Handle event
        event_handler = event_handler_list.get(event_kind)
        if event_handler is not None:
            event_handler(event_kind, appgui)

        # Change class id
        if event_kind == 'Class ID':
            class_id = appgui.get_setting_class_id()

        # Save
        if event_kind == 's':
            util.save_image_and_mask(
                output_image_path,
                image,
                output_annotation_path,
                mask,
                file_paths[current_index],
                (output_width, output_height),
                class_id
            )
        # Exit
        if event_kind == 'Escape':
            break


if __name__ == '__main__':
    main()
