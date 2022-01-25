#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Any, Tuple, Optional

import PySimpleGUI as sg
import cv2
import numpy as np

from core.event import Mouse


class AppGui(object):
    def __init__(self, file_paths: List[Path], mouse: Mouse, image_size: Tuple[int, int]):
        sg.theme('DarkBlue')
        self.mouse = mouse
        self.image_size = image_size
        self.window = sg.Window(
            title='GrabCut Annotation Tool',
            layout=[[
                sg.Listbox(
                    values=[str(f.name) for f in file_paths],
                    size=(15, 50),
                    bind_return_key=True,
                    enable_events=True,
                    key='-LISTBOX FILE-',
                ),
                sg.Frame(
                    title='',
                    layout=[
                        [sg.Frame(
                            title='',
                            layout=[[
                                sg.Graph(
                                    canvas_size=image_size,
                                    graph_bottom_left=(0, 0),
                                    graph_top_right=image_size,
                                    change_submits=True,
                                    drag_submits=True,
                                    key='-IMAGE ORIGINAL-',
                                ),
                                sg.Graph(
                                    canvas_size=image_size,
                                    graph_bottom_left=(0, 0),
                                    graph_top_right=image_size,
                                    change_submits=True,
                                    drag_submits=True,
                                    key='-IMAGE MASK-',
                                )]],
                            border_width=0)],
                        [sg.Frame(
                            title='Class',
                            layout=[[
                                sg.InputText(
                                    default_text='0',
                                    enable_events=True,
                                    key='-CLASS ID-'
                                )]],
                            border_width=1)],
                        [sg.Checkbox(
                            text='Manually label background',
                            enable_events=True,
                            key='-CHECKBOX BACKGROUND-'
                        )]],
                    border_width=0),
            ]],
            size=(1220, 620),
            return_keyboard_events=True,
            finalize=True,
            location=(50, 50),
        )

        self.window.Element('-LISTBOX FILE-').Update(set_to_index=0)
        self.window['-CHECKBOX BACKGROUND-'].Update(value=True)

    def reset(self):
        self.window['-CHECKBOX BACKGROUND-'].Update(value=True)
        self.mouse.reset()
        self.window.read(timeout=1)

    def read_window(self, timeout: Optional[int] = None) -> Tuple[str, Any]:
        event, values = self.window.read(timeout=timeout)
        if timeout is None:
            self.mouse.update(position=values['-IMAGE ORIGINAL-'], is_active=event == '-IMAGE ORIGINAL-')
        return event, values

    @property
    def class_id(self) -> str:
        return self.window['-CLASS ID-'].get()

    @property
    def is_drawing_bg(self) -> bool:
        return self.window['-CHECKBOX BACKGROUND-'].get()

    @property
    def file_index(self):
        return self.window.Element('-LISTBOX FILE-').GetIndexes()[0]

    def set_file_list_current_index(self, index, scroll=False):
        self.window['-LISTBOX FILE-'].Update(
                set_to_index=index,
                scroll_to_index=index if scroll else None,
        )
        self.window.read(timeout=1)

    @property
    def listbox_size(self):
        return len(self.window['-LISTBOX FILE-'].get_list_values())

    def get_from_listbox(self, index):
        return self.window['-LISTBOX FILE-'].get_list_values()[index]

    def draw(self, data: np.ndarray, is_mask: bool = False):
        graph = self.window['-IMAGE MASK-'] if is_mask else self.window['-IMAGE ORIGINAL-']
        graph.erase()
        graph.draw_image(data=cv2.imencode('.png', data)[1].tobytes(), location=(0, self.image_size[1]))

    def __del__(self):
        self.window.close()
