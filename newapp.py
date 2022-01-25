from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional

import click
import cv2
import numpy as np

from core.event import Mouse, EventHandler
from core.newgui import AppGui


class GrabCutApp:

    IMAGE_SIZE = (512, 512)
    MASK_ALPHA = 0.7
    READ_WAIT = 100
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (103, 82, 51)

    def __init__(self, image_in: Path, image_out: Path, annotation_out: Path):
        self.files = sorted(p for p in image_in.iterdir() if p.is_file() and p.suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp'])
        self.gui = AppGui(file_paths=self.files, mouse=Mouse(image_size=self.IMAGE_SIZE), image_size=self.IMAGE_SIZE)
        self.image_out = image_out
        self.annotation_out = annotation_out
        self.image = cv2.resize(cv2.imread(str(self.filename)), self.IMAGE_SIZE)
        self.debug_image = deepcopy(self.image)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
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
        self.bgd_model = np.zeros((1, 65), dtype=np.float64)
        self.fgd_model = np.zeros((1, 65), dtype=np.float64)
        self.reset_debug_image()
        self.reset_debug_mask()
        self.is_roi_mode = True
        self.gui.reset()

    def reset_debug_image(self):
        self.debug_image = deepcopy(self.image)

    def reset_debug_mask(self):
        self.debug_mask = deepcopy(self.mask)

    def grabcut(self, roi: Optional[Tuple[int, int, int, int]] = None):
        grabcut_mask, self.bgd_model, self.fgd_model = cv2.grabCut(
            img=self.image,
            mask=self.debug_mask,
            rect=roi,
            bgdModel=self.bgd_model,
            fgdModel=self.fgd_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_RECT if roi is not None else cv2.GC_INIT_WITH_MASK)
        self.mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype('uint8')
        self.debug_image = self.image * self.mask[:, :, np.newaxis]
        self.debug_image = cv2.addWeighted(src1=self.debug_image, alpha=self.MASK_ALPHA, src2=self.image, beta=1 - self.MASK_ALPHA, gamma=0)

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
            self.is_roi_mode = False
            # Set Drawing color
            color = self.BLACK if self.is_drawing_bg else self.WHITE
            cv2.rectangle(self.debug_image, (0, 0), (self.IMAGE_SIZE[0] - 1, self.IMAGE_SIZE[1] - 1), color=color, thickness=3)
            self.gui.mouse.reset()
        # Draw step results
        self.gui.draw(self.debug_image)
        self.gui.draw(self.mask, is_mask=True)

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
            self.is_roi_mode = False
            # Set Drawing color
            color = self.BLACK if self.is_drawing_bg else self.WHITE
            cv2.rectangle(self.debug_image, (0, 0), (self.IMAGE_SIZE[0] - 1, self.IMAGE_SIZE[1] - 1), color=color, thickness=3)
            self.gui.mouse.reset()
        # Draw step results
        self.gui.draw(self.debug_image)
        self.gui.draw(self.mask, is_mask=True)

    def draw_roi(self, roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
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
        point1, point2 = self.gui.mouse.prev_location, self.gui.mouse.location
        color = self.BLACK if self.is_drawing_bg else self.WHITE
        cv2.line(img=self.debug_image, pt1=point1, pt2=point2, color=color, thickness=4)
        cv2.line(img=self.debug_mask, pt1=point1, pt2=point2, color=color, thickness=4)
        return self.debug_image, self.debug_mask

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
        class_id = self.gui.class_id
        cv2.imwrite(str(self.image_out.joinpath(f'{self.filename.stem}_{class_id}.png')), self.image)
        cv2.imwrite(str(self.annotation_out.joinpath(f'{self.filename.stem}_{class_id}_mask.png')), self.mask)


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
    app = GrabCutApp(image_in=image_input, image_out=image_output, annotation_out=annotation_output)
    handler = EventHandler(app=app)
    # Draw
    app.draw_roi()
    app.gui.draw(app.debug_image)
    app.gui.draw(app.debug_mask, is_mask=True)

    keep_iterating = True
    while keep_iterating:
        event, value = app.gui.read_window()
        app.step()
        keep_iterating = handler.handle_event(event)


if __name__ == '__main__':
    main()
