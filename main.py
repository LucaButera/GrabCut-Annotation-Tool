from pathlib import Path

import click

from core.event import EventHandler
from core.app import GrabCutApp


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
def main(image_input: Path, image_output: Path, annotation_output: Path):
    app = GrabCutApp(image_in=image_input, image_out=image_output, annotation_out=annotation_output)
    handler = EventHandler(app=app)
    # Initialize
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
