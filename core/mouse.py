from typing import Tuple, Optional


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
        self.is_roi_mode = True

    def reset(self):
        self.state = self.STATE_IDLE
        self.location = (0, 0)
        self.prev_location = (0, 0)

    def clamp_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        x = min(self.image_size[0], max(1, point[0]))
        y = self.image_size[1] - min(self.image_size[1] - 1, max(0, point[1]))
        return x, y

    def get_spanned_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.prev_location == self.location and self.is_roi_mode:
            min_x = min(self.prev_location[0], self.location[0])
            man_x = max(self.prev_location[0], self.location[0])
            min_y = min(self.prev_location[1], self.location[1])
            man_y = max(self.prev_location[1], self.location[1])
            roi = [min_x, min_y, man_x, man_y]
        else:
            roi = None
        return roi

    def update(self, position: Tuple[int, int], is_active: bool):
        if is_active:
            position = self.clamp_point(position)
            if self.state in [self.STATE_IDLE, self.STATE_DRAG_END]:
                self.state = self.STATE_DRAG_START
                self.prev_location = position
                self.location = position
            elif self.state in [self.STATE_DRAG_START, self.STATE_DRAG]:
                self.state = self.STATE_DRAG
                if not self.is_roi_mode:
                    self.prev_location = self.location
                self.location = position
        else:
            if self.state in [self.STATE_DRAG_START, self.STATE_DRAG]:
                self.state = self.STATE_DRAG_END
            elif self.state == self.STATE_DRAG_END:
                self.reset()
