from typing import Callable, Any

import cv2

from core.app import GrabCutApp


class EventHandler:

    def __init__(self, app: GrabCutApp):
        self.app = app

    def get_handler(self, event: str) -> Callable:
        def return_true(a: Any) -> bool:
            return True
        if event.startswith('Up') or event.startswith('p'):
            return lambda: return_true(self.event_handler_file_select(scroll_count=-1))
        elif event.startswith('Down') or event.startswith('n'):
            return lambda: return_true(self.event_handler_file_select(scroll_count=1))
        elif event.startswith('Save'):
            return lambda: return_true(self.app.save())
        elif event.startswith('Control'):
            return lambda: return_true(self.event_handler_change_manually_label())
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

