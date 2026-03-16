import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_drawing_canvas_component = components.declare_component(
    "drawing_canvas",
    path=_COMPONENT_DIR
)


def drawing_canvas(image_base64: str, width: int, height: int, key: str = None):
    """
    Renders a drawing canvas with a background image.

    Args:
        image_base64: Base64-encoded JPEG background image string
        width: Canvas width in pixels
        height: Canvas height in pixels
        key: Unique Streamlit widget key

    Returns:
        str or None: Base64-encoded PNG of the drawing overlay, or None if no submission yet
    """
    return _drawing_canvas_component(
        image_base64=image_base64,
        width=width,
        height=height,
        key=key,
        default=None
    )
