# GraphPC vendored copy of OSWorld-V2 mm_agents/qwen/images.py
# at tag osworld-v2.1 (commit 3d778a3c9a34a079316f70df023b166700445792), blob 81d10416eed4188db6679a3daabc6baa483d3c24.
# The OSWorld Qwen cell's CONTROL is upstream's own code: this file is byte-identical to
# that blob except this header and the deltas listed below. Each delta is marked in place
# (`# >>> GraphPC Dn` ... `# <<< GraphPC Dn`) and carries the upstream lines it
# replaces as `#|` lines, so the file reverses mechanically to upstream's; the Qwen mode
# of scripts/osworld/verify_runner_deltas.py holds it to that. The delta list with its
# reasons is in the header of scripts/osworld/run_multienv_qwen_baseline.py.
# Deltas carried: D9 (every frame goes out as JPEG, not PNG).
# end of GraphPC header
"""Screenshot processing and coordinate projection for Qwen."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from mm_agents.utils.qwen_vl_utils import smart_resize


RELATIVE_COORDINATE_MAX = 1000.0
ZOOM_OUTPUT_SIZE = (1280, 720)


@dataclass(frozen=True)
class ProcessedScreenshot:
    data_url: str
    coordinate_type: str
    original_width: int
    original_height: int
    model_width: int
    model_height: int


# >>> GraphPC D9: every frame goes out as JPEG at the quality below, PIL defaults otherwise, not PNG
# (20 retained PNG frames at 1920x1088 exceed OpenRouter's 30 MB cap)
#|def _encode_png(image: Image.Image) -> str:
#|    output = BytesIO()
#|    image.convert("RGB").save(output, format="PNG")
#|    return base64.b64encode(output.getvalue()).decode("ascii")
def _encode_jpeg(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("ascii")
# <<< GraphPC D9


def process_screenshot(
    image_bytes: bytes,
    *,
    coordinate_type: str,
    model_screen_width: int | None = None,
    model_screen_height: int | None = None,
) -> ProcessedScreenshot:
    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        original_width, original_height = source.size
        if model_screen_width is not None and model_screen_height is not None:
            size = (int(model_screen_width), int(model_screen_height))
        else:
            resized_height, resized_width = smart_resize(
                height=original_height,
                width=original_width,
                factor=32,
                max_pixels=16 * 16 * 4 * 12800,
            )
            size = (resized_width, resized_height)
        resized = source.resize(size, Image.Resampling.LANCZOS)
        # >>> GraphPC D9
#|        encoded = _encode_png(resized)
        encoded = _encode_jpeg(resized)
        # <<< GraphPC D9
    return ProcessedScreenshot(
        # >>> GraphPC D9
#|        data_url=f"data:image/png;base64,{encoded}",
        data_url=f"data:image/jpeg;base64,{encoded}",
        # <<< GraphPC D9
        coordinate_type=coordinate_type,
        original_width=original_width,
        original_height=original_height,
        model_width=size[0],
        model_height=size[1],
    )


def _bounds_for(
    coordinate_type: str,
    model_width: int,
    model_height: int,
) -> tuple[float, float]:
    if coordinate_type == "relative":
        return RELATIVE_COORDINATE_MAX, RELATIVE_COORDINATE_MAX
    return float(model_width), float(model_height)


def project_point(
    x: float,
    y: float,
    *,
    coordinate_type: str,
    original_width: int,
    original_height: int,
    model_width: int,
    model_height: int,
) -> tuple[int, int]:
    bound_x, bound_y = _bounds_for(coordinate_type, model_width, model_height)
    if not 0 <= x <= bound_x or not 0 <= y <= bound_y:
        raise ValueError(
            f"coordinate ({x:g}, {y:g}) is outside 0..{bound_x:g} x 0..{bound_y:g}"
        )
    projected_x = min(original_width - 1, round(x * original_width / bound_x))
    projected_y = min(original_height - 1, round(y * original_height / bound_y))
    return max(0, projected_x), max(0, projected_y)


def zoom_screenshot(
    image_bytes: bytes,
    region: tuple[float, float, float, float],
    *,
    coordinate_type: str,
    model_width: int,
    model_height: int,
) -> bytes:
    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        x0, y0 = project_point(
            region[0],
            region[1],
            coordinate_type=coordinate_type,
            original_width=source.width,
            original_height=source.height,
            model_width=model_width,
            model_height=model_height,
        )
        x1, y1 = project_point(
            region[2],
            region[3],
            coordinate_type=coordinate_type,
            original_width=source.width,
            original_height=source.height,
            model_width=model_width,
            model_height=model_height,
        )
        if x1 <= x0 or y1 <= y0:
            raise ValueError("zoom region must have positive width and height")
        cropped = source.crop((x0, y0, x1, y1))
        scale = min(
            ZOOM_OUTPUT_SIZE[0] / cropped.width,
            ZOOM_OUTPUT_SIZE[1] / cropped.height,
        )
        target = (
            max(1, round(cropped.width * scale)),
            max(1, round(cropped.height * scale)),
        )
        resized = cropped.resize(target, Image.Resampling.LANCZOS)
        output = BytesIO()
        resized.convert("RGB").save(output, format="PNG")
        return output.getvalue()
