from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from core.mouse import Mouse
from core.gui import AppGui


class GrabCutApp:
    IMAGE_SIZE = (512, 512)
    MASK_ALPHA = 0.7
    READ_WAIT = 100
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (103, 82, 51)

    def __init__(self, image_in: Path, image_out: Path, annotation_out: Path):
        self.files = sorted(
            p for p in image_in.iterdir() if p.is_file() and p.suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])
        self.gui = AppGui(file_paths=self.files, image_size=self.IMAGE_SIZE)
        self.image_out = image_out
        self.annotation_out = annotation_out
        self.image = cv2.resize(cv2.imread(str(self.filename)), self.IMAGE_SIZE)
        self.debug_image = deepcopy(self.image)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.grabcut_mask = None
        self.help_mask = None
        self.debug_mask = deepcopy(self.mask)
        self.bgd_model = np.zeros((1, 65), dtype=np.float64)
        self.fgd_model = np.zeros((1, 65), dtype=np.float64)
        self.is_roi_mode = True

    @property
    def filename(self):
        return self.files[self.gui.file_index]

    @property
    def is_drawing_bg(self) -> bool:
        return self.gui.is_drawing_bg

    def reset(self):
        self.image = cv2.resize(cv2.imread(str(self.filename)), self.IMAGE_SIZE)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.reset_debug_image()
        self.reset_grabcut()
        self.reset_debug_mask()
        self.set_roi_mode(True)
        self.gui.reset()

    def reset_debug_image(self):
        self.debug_image = deepcopy(self.image)

    def reset_debug_mask(self):
        self.debug_mask = deepcopy(self.mask)

    def reset_grabcut(self):
        self.bgd_model = np.zeros((1, 65), dtype=np.float64)
        self.fgd_model = np.zeros((1, 65), dtype=np.float64)
        self.grabcut_mask = None
        self.help_mask = None

    def set_roi_mode(self, state: bool):
        if state:
            self.reset_grabcut()
        self.is_roi_mode = state
        self.gui.mouse.is_roi_mode = state

    def grabcut(self, roi: Optional[Tuple[int, int, int, int]] = None):
        if self.grabcut_mask is None:
            self.grabcut_mask = deepcopy(self.mask)
        if self.help_mask is not None:
            self.grabcut_mask[self.help_mask == 0] = 0
            self.grabcut_mask[self.help_mask == 255] = 1
        self.grabcut_mask, self.bgd_model, self.fgd_model = cv2.grabCut(
            img=self.image,
            mask=self.grabcut_mask,
            rect=roi,
            bgdModel=self.bgd_model,
            fgdModel=self.fgd_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_RECT if roi is not None else cv2.GC_INIT_WITH_MASK)
        self.mask = np.where((self.grabcut_mask == 2) | (self.grabcut_mask == 0), 0, 255).astype('uint8')
        self.debug_image = self.image * self.mask[:, :, np.newaxis]
        self.debug_image = cv2.addWeighted(src1=self.debug_image, alpha=self.MASK_ALPHA, src2=self.image,
                                           beta=1 - self.MASK_ALPHA, gamma=0)
        self.debug_mask = deepcopy(self.mask)

    def step(self):
        if self.is_roi_mode:
            self.roi_mode_step()
        else:
            self.grabcut_mode_step()

    def grabcut_mode_step(self):
        # Draw mask manually in grabcut mode
        if self.gui.mouse.state in [Mouse.STATE_DRAG_START, Mouse.STATE_DRAG]:
            self.draw_grabcut_help_mask()
        # Update image
        elif self.gui.mouse.state == Mouse.STATE_DRAG_END:
            # Show wait processing image
            self.draw_processing_waitscreen()
            self.gui.draw(self.debug_image)
            self.gui.read_window(timeout=self.READ_WAIT)
            self.reset_debug_image()
            # execute GrabCut
            self.grabcut()
            self.set_roi_mode(False)
            # Set Drawing color
            color = self.BLACK if self.is_drawing_bg else self.WHITE
            cv2.rectangle(self.debug_image, (0, 0), (self.IMAGE_SIZE[0] - 1, self.IMAGE_SIZE[1] - 1), color=color,
                          thickness=3)
            self.gui.mouse.reset()
        # Draw step results
        self.gui.draw(self.debug_image)
        self.gui.draw(self.debug_mask, is_mask=True)

    def roi_mode_step(self):
        # Draw ROI
        if self.gui.mouse.state == Mouse.STATE_DRAG:
            self.draw_roi(roi=self.gui.mouse.get_spanned_roi())
        # Finalize ROI and process
        elif self.gui.mouse.state == Mouse.STATE_DRAG_END:
            # Show wait process image
            self.draw_processing_waitscreen()
            self.gui.draw(self.debug_image)
            self.gui.read_window(timeout=self.READ_WAIT)
            self.reset_debug_image()
            # execute GrabCut
            self.grabcut(roi=self.gui.mouse.get_spanned_roi())
            self.set_roi_mode(False)
            # Set Drawing color
            color = self.BLACK if self.is_drawing_bg else self.WHITE
            cv2.rectangle(self.debug_image, (0, 0), (self.IMAGE_SIZE[0] - 1, self.IMAGE_SIZE[1] - 1), color=color,
                          thickness=3)
            self.gui.mouse.reset()
        # Draw step results
        self.gui.draw(self.debug_image)
        self.gui.draw(self.debug_mask, is_mask=True)

    def draw_roi(self, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        self.reset_debug_image()
        cv2.putText(img=self.debug_image, text="Select ROI", org=(5, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(img=self.debug_image, text="Select ROI", org=(5, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9, color=self.BLUE, thickness=1, lineType=cv2.LINE_AA)
        if roi is not None:
            cv2.rectangle(
                img=self.debug_image,
                pt1=(roi[0], roi[1]),
                pt2=(roi[2], roi[3]),
                color=self.WHITE,
                thickness=3,
            )
            cv2.rectangle(
                img=self.debug_image,
                pt1=(roi[0], roi[1]),
                pt2=(roi[2], roi[3]),
                color=self.BLUE,
                thickness=2,
            )
        return self.debug_image

    def draw_grabcut_help_mask(self):
        if self.help_mask is None:
            self.help_mask = np.full_like(self.mask, 127)
        point1, point2 = self.gui.mouse.prev_location, self.gui.mouse.location
        color = self.BLACK if self.is_drawing_bg else self.WHITE
        cv2.line(img=self.debug_image, pt1=point1, pt2=point2, color=color, thickness=4)
        cv2.line(img=self.help_mask, pt1=point1, pt2=point2, color=color, thickness=4)
        self.debug_mask[self.help_mask == 0] = 0
        self.debug_mask[self.help_mask == 255] = 255

    def draw_processing_waitscreen(self):
        self.reset_debug_image()
        loading_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.debug_image = self.debug_image * loading_mask[:, :, np.newaxis]
        self.debug_image = cv2.addWeighted(self.debug_image, self.MASK_ALPHA, self.image, 1 - self.MASK_ALPHA, 0)
        cv2.putText(img=self.debug_image, text="PROCESSING...",
                    org=(int(self.debug_image.shape[1] / 2) - (6 * 18), int(self.debug_image.shape[0] / 2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=self.WHITE, thickness=4,
                    lineType=cv2.LINE_AA)
        cv2.putText(img=self.debug_image, text="PROCESSING...",
                    org=(int(self.debug_image.shape[1] / 2) - (6 * 18), int(self.debug_image.shape[0] / 2)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=self.BLUE, thickness=2,
                    lineType=cv2.LINE_AA)
        return self.debug_image

    def save(self):
        class_id = self.gui.get_class_id()
        if class_id:
            cv2.imwrite(str(self.image_out.joinpath(f'{self.filename.stem}_{class_id}.png')), self.image)
            cv2.imwrite(str(self.annotation_out.joinpath(f'{self.filename.stem}_{class_id}_mask.png')), self.mask)
