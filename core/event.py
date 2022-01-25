from typing import Tuple, Optional, Callable

import cv2

from newapp import GrabCutApp


class Mouse:

    STATE_IDLE = 0
    STATE_DRAG_START = 1
    STATE_DRAG = 2
    STATE_DRAG_END = 3

    def __init__(self, image_size: Tuple[int, int]):
        self.image_size = image_size
        self.state = self.STATE_IDLE
        self.location = (0, 0)
        self.prev_location = (0, 0)

    def reset(self):
        self.state = self.STATE_IDLE
        self.location = (0, 0)
        self.prev_location = (0, 0)

    def clamp_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x = min(self.image_size[0], max(1, point[0]))
        y = self.image_size[1] - min(self.image_size[1] - 1, max(0, point[1]))
        return x, y

    def get_spanned_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.prev_location == self.location:
            min_x = min(self.prev_location[0], self.location[0])
            man_x = max(self.prev_location[0], self.location[0])
            min_y = min(self.prev_location[1], self.location[1])
            man_y = max(self.prev_location[1], self.location[1])
            roi = [min_x, min_y, man_x, man_y]
        else:
            roi = None
        return roi

    def update(self, position: Tuple[int, int], is_active: bool):
        position = self.clamp_point(position)
        if is_active:
            if self.state in [self.STATE_IDLE, self.STATE_DRAG_END]:
                self.state = self.STATE_DRAG_START
                self.prev_location = position
                self.location = position
            elif self.state in [self.STATE_DRAG_START, self.STATE_DRAG]:
                self.state = self.STATE_DRAG
                self.prev_location = self.location
                self.location = position
        else:
            if self.state in [self.STATE_DRAG_START, self.STATE_DRAG]:
                self.state = self.STATE_DRAG_END
            elif self.state == self.STATE_DRAG_END:
                self.reset()


class EventHandler:

    def __init__(self, app: GrabCutApp):
        self.app = app

    def get_handler(self, event: str) -> Callable:
        if event.startswith('Up') or event.startswith('p'):
            return lambda: self.event_handler_file_select(scroll_count=-1)
        elif event.startswith('Down') or event.startswith('n'):
            return lambda: self.event_handler_file_select(scroll_count=1)
        elif event.startswith('s'):
            return self.app.save
        elif event.startswith('Control'):
            return self.event_handler_change_manually_label
        elif event.startswith('Escape'):
            return lambda: False
        else:
            return lambda: True

    def handle_event(self, event: str):
        return self.get_handler(event)()

    def event_handler_file_select(self, scroll_count=0):
        self.app.gui.set_file_list_current_index(
            index=(self.app.gui.file_index + scroll_count) % self.app.gui.listbox_size,
            scroll=not scroll_count == 0)
        self.app.reset()
        self.app.draw_roi()
        self.app.gui.draw(self.app.debug_image)
        self.app.gui.draw(self.app.mask, is_mask=True)

    def event_handler_change_manually_label(self):
        if not self.app.is_roi_mode:
            color = self.app.BLACK if self.app.is_drawing_bg else self.app.WHITE
            cv2.rectangle(img=self.app.debug_image,
                          pt1=(0, 0),
                          pt2=(self.app.IMAGE_SIZE[0] - 1, self.app.IMAGE_SIZE[1] - 1),
                          color=color,
                          thickness=3)
            self.app.gui.draw(self.app.debug_image)

