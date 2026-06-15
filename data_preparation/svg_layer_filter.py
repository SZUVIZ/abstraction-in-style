import math
import os
import copy
from pathlib import Path
import re
import csv
import urllib.request
import xml.etree.ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cairocffi as cairo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from svgpathtools import parse_path
from svgpathtools.parser import parse_transform


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

"""
SVG layer filtering has two stages:

1. Subject filtering
- Use SAM to estimate the subject region from the reference image.
- Compute the coverage of each SVG path inside the subject region.
- Remove paths whose coverage is below the threshold.

2. Layer filtering
- Estimate depth for paths that survive subject filtering.
- Keep only the layers listed in `KEEP_LAYERS`.

Output modes:
- Default mode keeps the debug/grouped output structure and writes results to
  `layerXYZ/subject_filtered` and `layerXYZ/subject_filtered_layer_filtered`.
- `direct_output=True` writes only the final filtered result to `output_dir`,
  which is useful for directly replacing `proxy_svg`.

Debug outputs:
- `debug_mask_dir`
- `debug_path_stats_dir`
- `debug_removed_paths_dir`
"""

INPUT_DIR = "./paper_test_data/svgs"
IMAGE_DIR = "./paper_test_data/imgs"
OUTPUT_DIR = "./paper_test_data/filter"
DEBUG_MASK_DIR = f"{OUTPUT_DIR}/debug_subject_masks"
DEBUG_PATH_STATS_DIR = f"{OUTPUT_DIR}/debug_path_stats"
DEBUG_REMOVED_PATHS_DIR = f"{OUTPUT_DIR}/debug_removed_paths"
KEEP_LAYERS = [[0, 1, 2],[0, 1, 2, 3]]
LAYER_IOU_THRESHOLD = 0.2
SUBJECT_COVERAGE_THRESHOLD = 0.6
RECURSIVE = False
SCALE_LIMIT = 1000.0
MIN_CANVAS_SIZE = 1
SUBJECT_MASK_MAX_SIZE = 512
CURRENT_DIR = Path(__file__).resolve().parent
SAM_CHECKPOINT_DIR = CURRENT_DIR / "lv" / "checkpoints"
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = None
SAM_DEVICE = "cuda"
SAM_POINTS_PER_SIDE = 32
SAM_PRED_IOU_THRESH = 0.86
SAM_STABILITY_SCORE_THRESH = 0.92
SAM_CROP_N_LAYERS = 1
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 2
SAM_MIN_MASK_REGION_AREA = 100
SAM_BOX_NMS_THRESH = 0.7
SAM_NEARBY_MASK_MAX_AREA_RATIO = 0.18
SAM_NEARBY_MASK_MIN_OVERLAP = 0.08
SAM_NEARBY_MASK_MAX_CENTER_DISTANCE_RATIO = 0.28

SAM_MODEL_SPECS = {
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "description": "Segment Anything ViT-H",
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "description": "Segment Anything ViT-L",
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "description": "Segment Anything ViT-B",
    },
}

_SAM_MASK_GENERATORS: dict[tuple[str, str, str], SamAutomaticMaskGenerator] = {}


def parse_numeric(value: str | None, default: float) -> float:
    if value is None:
        return default
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    return float(match.group(0)) if match else default


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(x) for x in re.split(r"[\s,]+", view_box.strip()) if x]
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]

    width = parse_numeric(root.get("width"), 500.0)
    height = parse_numeric(root.get("height"), 500.0)
    return 0.0, 0.0, width, height


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def get_path_elements(root: ET.Element) -> list[ET.Element]:
    return [el for el in root.iter() if strip_ns(el.tag) == "path" and el.get("d")]


def normalize_keep_layer_sets(keep_layers_config: list[int] | list[list[int]]) -> list[tuple[int, ...]]:
    if not keep_layers_config:
        return []

    first = keep_layers_config[0]
    if isinstance(first, int):
        layer_sets = [tuple(int(layer) for layer in keep_layers_config)]
    else:
        layer_sets = [tuple(int(layer) for layer in layer_set) for layer_set in keep_layers_config]

    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for layer_set in layer_sets:
        deduped = tuple(dict.fromkeys(layer_set))
        if deduped and deduped not in seen:
            seen.add(deduped)
            normalized.append(deduped)
    return normalized


def format_keep_layers_dirname(keep_layers: tuple[int, ...]) -> str:
    return "layer" + "".join(str(layer) for layer in keep_layers)


def find_image_for_svg(svg_path: Path, input_dir: Path, image_dir: Path) -> Path:
    relative = svg_path.relative_to(input_dir)
    stem = relative.with_suffix("")
    candidates = [
        image_dir / f"{stem}.png",
        image_dir / f"{stem}.jpg",
        image_dir / f"{stem}.jpeg",
        image_dir / f"{stem}.webp",
        image_dir / f"{stem}.bmp",
        image_dir / f"{stem}.tif",
        image_dir / f"{stem}.tiff",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching image found for {svg_path.name} under {image_dir}")


def compute_canvas_size(
    view_box: tuple[float, float, float, float],
    scale_limit: float,
    min_canvas_size: int,
) -> tuple[int, int]:
    _, _, width, height = view_box
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid SVG size from viewBox/width/height: {view_box}")
    scale = min(1.0, scale_limit / max(width, height))
    canvas_w = max(min_canvas_size, int(math.ceil(width * scale)))
    canvas_h = max(min_canvas_size, int(math.ceil(height * scale)))
    return canvas_w, canvas_h


def render_mask_for_path(
    root_attrs: dict[str, str],
    view_box: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    path_attrs: dict[str, str],
) -> np.ndarray:
    vb_x, vb_y, vb_w, vb_h = view_box
    canvas_w, canvas_h = canvas_size
    surface = cairo.ImageSurface(cairo.FORMAT_A8, canvas_w, canvas_h)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_GRAY)
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    d_attr = path_attrs.get("d")
    if not d_attr:
        return np.zeros((canvas_h, canvas_w), dtype=bool)

    transform_matrix = None
    transform_attr = path_attrs.get("transform")
    if transform_attr:
        transform_matrix = parse_transform(transform_attr)

    def transform_point(point: complex) -> tuple[float, float]:
        x = point.real
        y = point.imag
        if transform_matrix is not None:
            tx = transform_matrix[0][0] * x + transform_matrix[0][1] * y + transform_matrix[0][2]
            ty = transform_matrix[1][0] * x + transform_matrix[1][1] * y + transform_matrix[1][2]
            x, y = tx, ty
        x = (x - vb_x) * canvas_w / vb_w
        y = (y - vb_y) * canvas_h / vb_h
        return x, y

    def stroke_scale_factor() -> float:
        base_sx = canvas_w / vb_w
        base_sy = canvas_h / vb_h
        if transform_matrix is None:
            return (base_sx + base_sy) / 2.0
        sx = math.hypot(transform_matrix[0][0], transform_matrix[1][0])
        sy = math.hypot(transform_matrix[0][1], transform_matrix[1][1])
        return ((sx * base_sx) + (sy * base_sy)) / 2.0

    def append_segments() -> None:
        path = parse_path(d_attr)
        current_point: complex | None = None
        for segment in path:
            start_point = segment.start
            if current_point is None or abs(start_point - current_point) > 1e-9:
                x0, y0 = transform_point(start_point)
                ctx.move_to(x0, y0)

            if hasattr(segment, "as_cubic_curves"):
                cubic_segments = list(segment.as_cubic_curves())
            else:
                cubic_segments = [segment]

            for cubic in cubic_segments:
                bpoints = cubic.bpoints()
                if len(bpoints) == 2:
                    x1, y1 = transform_point(bpoints[1])
                    ctx.line_to(x1, y1)
                elif len(bpoints) == 3:
                    p0, p1, p2 = bpoints
                    c1 = p0 + (2.0 / 3.0) * (p1 - p0)
                    c2 = p2 + (2.0 / 3.0) * (p1 - p2)
                    cx1, cy1 = transform_point(c1)
                    cx2, cy2 = transform_point(c2)
                    x2, y2 = transform_point(p2)
                    ctx.curve_to(cx1, cy1, cx2, cy2, x2, y2)
                elif len(bpoints) == 4:
                    _, c1, c2, p3 = bpoints
                    cx1, cy1 = transform_point(c1)
                    cx2, cy2 = transform_point(c2)
                    x3, y3 = transform_point(p3)
                    ctx.curve_to(cx1, cy1, cx2, cy2, x3, y3)
                current_point = cubic.end

    append_segments()

    opacity = parse_numeric(path_attrs.get("opacity"), 1.0)
    fill_opacity = parse_numeric(path_attrs.get("fill-opacity"), 1.0) * opacity
    stroke_opacity = parse_numeric(path_attrs.get("stroke-opacity"), 1.0) * opacity
    fill_value = (path_attrs.get("fill") or "").strip().lower()
    stroke_value = (path_attrs.get("stroke") or "").strip().lower()

    if fill_value != "none" and fill_opacity > 0:
        fill_rule = (path_attrs.get("fill-rule") or "").strip().lower()
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD if fill_rule == "evenodd" else cairo.FILL_RULE_WINDING)
        ctx.set_source_rgba(1.0, 1.0, 1.0, fill_opacity)
        if stroke_value != "none" and stroke_opacity > 0:
            ctx.fill_preserve()
        else:
            ctx.fill()

    if stroke_value != "none" and stroke_opacity > 0:
        linecap_map = {
            "round": cairo.LINE_CAP_ROUND,
            "square": cairo.LINE_CAP_SQUARE,
            "butt": cairo.LINE_CAP_BUTT,
        }
        linejoin_map = {
            "round": cairo.LINE_JOIN_ROUND,
            "bevel": cairo.LINE_JOIN_BEVEL,
            "miter": cairo.LINE_JOIN_MITER,
        }
        stroke_width = parse_numeric(path_attrs.get("stroke-width"), 1.0) * stroke_scale_factor()
        ctx.set_line_width(max(stroke_width, 0.0))
        ctx.set_line_cap(linecap_map.get((path_attrs.get("stroke-linecap") or "").strip().lower(), cairo.LINE_CAP_BUTT))
        ctx.set_line_join(linejoin_map.get((path_attrs.get("stroke-linejoin") or "").strip().lower(), cairo.LINE_JOIN_MITER))
        ctx.set_miter_limit(parse_numeric(path_attrs.get("stroke-miterlimit"), 4.0))
        ctx.set_source_rgba(1.0, 1.0, 1.0, stroke_opacity)
        ctx.stroke()
    else:
        ctx.new_path()

    alpha = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape((canvas_h, surface.get_stride()))[:, :canvas_w]
    return alpha > 0


def render_masks_for_paths(
    path_elements: list[ET.Element],
    root_attrs: dict[str, str],
    view_box: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
) -> list[np.ndarray]:
    return [
        render_mask_for_path(root_attrs, view_box, canvas_size, dict(path_el.attrib))
        for path_el in path_elements
    ]


def load_image_rgb(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"))


def resize_image_for_subject_mask(image: np.ndarray, target_max_size: int) -> np.ndarray:
    if target_max_size <= 0:
        return image

    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= target_max_size:
        return image

    scale = target_max_size / max_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(image).resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def save_subject_mask_debug(
    image_path: Path,
    input_dir: Path,
    debug_mask_dir: Path | None,
    image_rgb: np.ndarray,
    subject_mask: np.ndarray,
) -> None:
    if debug_mask_dir is None:
        return

    relative = image_path.relative_to(input_dir).with_suffix(".png")
    mask_output_path = debug_mask_dir / relative
    overlay_output_path = debug_mask_dir / relative.with_name(f"{relative.stem}_overlay.png")
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_image = Image.fromarray((~subject_mask).astype(np.uint8) * 255, mode="L")
    mask_image.save(mask_output_path)

    overlay = image_rgb.copy()
    subject_region = subject_mask.astype(bool)
    overlay[subject_region] = (
        0.45 * overlay[subject_region] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(overlay_output_path)


def get_sam_model_spec(model_type: str) -> dict[str, str]:
    model_spec = SAM_MODEL_SPECS.get(model_type)
    if model_spec is None:
        supported = ", ".join(sorted(SAM_MODEL_SPECS))
        raise ValueError(f"Unsupported SAM model type: {model_type}. Supported: {supported}")
    return model_spec


def resolve_sam_checkpoint(model_type: str, checkpoint_path: Path | None) -> Path:
    if checkpoint_path is not None:
        return checkpoint_path
    model_spec = get_sam_model_spec(model_type)
    return SAM_CHECKPOINT_DIR / model_spec["filename"]


def ensure_sam_checkpoint(model_type: str, checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path

    model_spec = get_sam_model_spec(model_type)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"SAM checkpoint missing for {model_type} ({model_spec['description']}), "
        f"downloading to {checkpoint_path} ..."
    )
    urllib.request.urlretrieve(model_spec["url"], checkpoint_path)
    return checkpoint_path


def get_sam_mask_generator(model_type: str, checkpoint_path: Path, device: str) -> SamAutomaticMaskGenerator:
    if SamAutomaticMaskGenerator is None or sam_model_registry is None:
        raise ImportError("segment_anything is required to generate subject masks.")

    checkpoint_path = ensure_sam_checkpoint(model_type, checkpoint_path)
    cache_key = (model_type, str(checkpoint_path.resolve()), device)
    mask_generator = _SAM_MASK_GENERATORS.get(cache_key)
    if mask_generator is None:
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=SAM_POINTS_PER_SIDE,
            pred_iou_thresh=SAM_PRED_IOU_THRESH,
            stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
            crop_n_layers=SAM_CROP_N_LAYERS,
            crop_n_points_downscale_factor=SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
            min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
            box_nms_thresh=SAM_BOX_NMS_THRESH,
        )
        _SAM_MASK_GENERATORS[cache_key] = mask_generator
    return mask_generator


def resolve_sam_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def select_subject_mask(masks: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    image_area = height * width
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    best_mask: np.ndarray | None = None
    best_score = float("-inf")
    fallback_mask: np.ndarray | None = None
    fallback_area = -1

    for mask_info in masks:
        mask = np.asarray(mask_info["segmentation"], dtype=bool)
        area = int(mask.sum())
        if area <= 0:
            continue

        if area > fallback_area:
            fallback_area = area
            fallback_mask = mask

        area_ratio = area / image_area
        if area_ratio >= 0.95:
            continue

        x, y, w, h = mask_info["bbox"]
        bbox_center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
        center_distance = np.linalg.norm(bbox_center - image_center)
        norm_center_distance = center_distance / max(np.linalg.norm(image_center), 1.0)

        touches_border = int(x <= 0) + int(y <= 0) + int(x + w >= width - 1) + int(y + h >= height - 1)
        score = area_ratio - 0.15 * norm_center_distance - 0.05 * touches_border
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is not None:
        return best_mask
    if fallback_mask is not None:
        return fallback_mask
    raise ValueError("SAM did not return a valid subject mask.")


def compute_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compute_mask_border_touches(mask: np.ndarray) -> int:
    bbox = compute_mask_bbox(mask)
    if bbox is None:
        return 4
    x0, y0, x1, y1 = bbox
    height, width = mask.shape
    return int(x0 <= 0) + int(y0 <= 0) + int(x1 >= width - 1) + int(y1 >= height - 1)


def score_subject_candidate(mask: np.ndarray, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    area = int(mask.sum())
    if area <= 0:
        return float("-inf")

    image_area = height * width
    area_ratio = area / image_area
    bbox = compute_mask_bbox(mask)
    if bbox is None:
        return float("-inf")
    x0, y0, x1, y1 = bbox
    mask_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    norm_center_distance = np.linalg.norm(mask_center - image_center) / max(np.linalg.norm(image_center), 1.0)
    touches_border = compute_mask_border_touches(mask)

    # Favor medium-to-large centered regions that do not hug the canvas border.
    size_score = 1.0 - min(abs(area_ratio - 0.45) / 0.45, 1.0)
    return size_score - 0.22 * norm_center_distance - 0.10 * touches_border


def expand_subject_mask(
    primary_mask: np.ndarray,
    masks: list[dict],
    image_shape: tuple[int, int],
) -> np.ndarray:
    subject_mask = primary_mask.copy()
    height, width = image_shape
    image_diagonal = max(math.hypot(width, height), 1.0)
    dilated_subject = dilate_mask(subject_mask, radius=max(8, int(round(min(height, width) * 0.015))))

    for mask_info in masks:
        candidate_mask = np.asarray(mask_info["segmentation"], dtype=bool)
        candidate_area = int(candidate_mask.sum())
        if candidate_area <= 0:
            continue
        if np.array_equal(candidate_mask, primary_mask):
            continue

        area_ratio = candidate_area / (height * width)
        if area_ratio > SAM_NEARBY_MASK_MAX_AREA_RATIO:
            continue
        if compute_mask_border_touches(candidate_mask) >= 4:
            continue

        overlap_ratio = compute_mask_coverage(candidate_mask, dilated_subject)
        candidate_bbox = compute_mask_bbox(candidate_mask)
        subject_bbox = compute_mask_bbox(subject_mask)
        if candidate_bbox is None or subject_bbox is None:
            continue

        cx0, cy0, cx1, cy1 = candidate_bbox
        sx0, sy0, sx1, sy1 = subject_bbox
        candidate_center = np.array([(cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0], dtype=np.float32)
        subject_center = np.array([(sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0], dtype=np.float32)
        center_distance_ratio = np.linalg.norm(candidate_center - subject_center) / image_diagonal

        if overlap_ratio >= SAM_NEARBY_MASK_MIN_OVERLAP or center_distance_ratio <= SAM_NEARBY_MASK_MAX_CENTER_DISTANCE_RATIO:
            subject_mask |= candidate_mask
            dilated_subject = dilate_mask(subject_mask, radius=max(8, int(round(min(height, width) * 0.015))))

    return subject_mask


def resolve_subject_mask(masks: list[dict], image_shape: tuple[int, int]) -> np.ndarray:
    primary_mask = select_subject_mask(masks, image_shape)
    direct_subject = expand_subject_mask(primary_mask, masks, image_shape)
    inverse_subject = np.logical_not(primary_mask)

    direct_score = score_subject_candidate(direct_subject, image_shape)
    inverse_score = score_subject_candidate(inverse_subject, image_shape)
    return inverse_subject if inverse_score > direct_score else direct_subject


def build_subject_mask(
    image_path: Path,
    model_type: str,
    checkpoint_path: Path,
    device: str,
    input_dir: Path | None = None,
    debug_mask_dir: Path | None = None,
) -> np.ndarray:
    image = load_image_rgb(image_path)
    image = resize_image_for_subject_mask(image, SUBJECT_MASK_MAX_SIZE)
    mask_generator = get_sam_mask_generator(model_type, checkpoint_path, device)
    masks = mask_generator.generate(image)
    if not masks:
        raise ValueError(f"SAM returned no masks for image: {image_path}")
    subject_mask = resolve_subject_mask(masks, image.shape[:2])
    if input_dir is not None:
        save_subject_mask_debug(
            image_path=image_path,
            input_dir=input_dir,
            debug_mask_dir=debug_mask_dir,
            image_rgb=image,
            subject_mask=subject_mask,
        )
    return subject_mask


def compute_mask_coverage(mask: np.ndarray, reference_mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    intersect = int(np.logical_and(mask, reference_mask).sum())
    return intersect / area


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersect = int(np.logical_and(mask_a, mask_b).sum())
    if intersect <= 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersect / union if union > 0 else 0.0


def compute_mask_outline(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    interior = mask.copy()
    interior &= np.roll(mask, 1, axis=0)
    interior &= np.roll(mask, -1, axis=0)
    interior &= np.roll(mask, 1, axis=1)
    interior &= np.roll(mask, -1, axis=1)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return np.logical_and(mask, np.logical_not(interior))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or mask.size == 0:
        return mask
    dilated = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
            if dy > 0:
                shifted[:dy, :] = False
            elif dy < 0:
                shifted[dy:, :] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            dilated |= shifted
    return dilated


def find_label_anchor(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0
    return int(xs.min()), int(ys.min())


def resolve_label_positions(
    removed_path_stats: list[dict[str, object]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    positions: list[tuple[int, int]] = []
    step_y = 14
    step_x = 90

    for item in removed_path_stats:
        base_x, base_y = item["label_anchor"]
        x = min(max(0, base_x + 6), max(0, image_width - step_x))
        y = min(max(0, base_y + 6), max(0, image_height - step_y))
        while any(abs(x - px) < step_x and abs(y - py) < step_y for px, py in placed):
            y += step_y
            if y > image_height - step_y:
                y = min(max(0, base_y + 6), max(0, image_height - step_y))
                x += step_x
            if x > image_width - step_x:
                x = max(0, image_width - step_x)
                break
        placed.append((x, y))
        positions.append((x, y))
    return positions


def save_removed_paths_debug(
    svg_relative: Path,
    debug_removed_paths_dir: Path | None,
    subject_mask: np.ndarray,
    removed_path_stats: list[dict[str, object]],
    subject_coverage_threshold: float,
) -> None:
    if debug_removed_paths_dir is None or not removed_path_stats:
        return

    output_path = (debug_removed_paths_dir / svg_relative).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full(subject_mask.shape, 255, dtype=np.uint8)
    canvas[subject_mask] = 0
    canvas_rgb = np.repeat(canvas[..., None], 3, axis=2)
    colors = [
        np.array([255, 80, 80], dtype=np.uint8),
        np.array([80, 200, 255], dtype=np.uint8),
        np.array([255, 190, 60], dtype=np.uint8),
        np.array([140, 255, 120], dtype=np.uint8),
    ]

    for idx, item in enumerate(removed_path_stats):
        outline = dilate_mask(item["outline_mask"], radius=1)
        canvas_rgb[outline] = colors[idx % len(colors)]

    image = Image.fromarray(canvas_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label_positions = resolve_label_positions(removed_path_stats, image.width, image.height)
    for idx, (item, (x, y)) in enumerate(zip(removed_path_stats, label_positions), start=1):
        coverage = item["path_coverage"]
        label = f"#{idx} cov={coverage:.3f} th={subject_coverage_threshold:.3f}"
        draw.text((x + 1, y + 1), label, fill=(255, 255, 255), font=font)
        draw.text((x, y), label, fill=tuple(int(v) for v in colors[(idx - 1) % len(colors)]), font=font)

    image.save(output_path)


def build_depths(
    path_masks: list[np.ndarray],
    iou_threshold: float,
) -> list[int]:
    path_infos = [{"mask": mask, "area": int(mask.sum())} for mask in path_masks]

    depths = [0] * len(path_infos)
    for i, current in enumerate(path_infos):
        parent_index = None
        for j in range(i - 1, -1, -1):
            candidate = path_infos[j]
            intersect = int(np.logical_and(current["mask"], candidate["mask"]).sum())
            if intersect <= 0:
                continue

            union = current["area"] + candidate["area"] - intersect
            iou = intersect / union if union > 0 else 0.0
            coverage = intersect / current["area"] if current["area"] > 0 else 0.0
            if iou > iou_threshold or coverage > 0.85:
                parent_index = j
                break

        depths[i] = depths[parent_index] + 1 if parent_index is not None else 0

    return depths


def build_parent_map(root: ET.Element) -> dict[int, ET.Element]:
    parent_map: dict[int, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parent_map[id(child)] = parent
    return parent_map


def filter_paths_by_subject(
    root: ET.Element,
    path_elements: list[ET.Element],
    path_masks: list[np.ndarray],
    svg_relative: Path,
    debug_path_stats_dir: Path | None,
    debug_removed_paths_dir: Path | None,
    subject_mask: np.ndarray,
    subject_coverage_threshold: float,
) -> tuple[list[ET.Element], int]:
    parent_map = build_parent_map(root)
    kept_paths: list[ET.Element] = []
    removed = 0
    path_stats: list[dict[str, str | float | int]] = []
    removed_path_stats: list[dict[str, object]] = []

    for path_index, (path_el, path_mask) in enumerate(zip(path_elements, path_masks)):
        path_coverage = compute_mask_coverage(path_mask, subject_mask)
        path_iou = compute_mask_iou(path_mask, subject_mask)
        keep = path_coverage >= subject_coverage_threshold
        path_stats.append(
            {
                "path_index": path_index,
                "path_id": path_el.get("id", ""),
                "path_iou": path_iou,
                "path_coverage": path_coverage,
                "path_area": int(path_mask.sum()),
                "keep": int(keep),
            }
        )
        if keep:
            kept_paths.append(path_el)
            continue

        parent = parent_map.get(id(path_el))
        if parent is not None:
            parent.remove(path_el)
            removed += 1

        removed_path_stats.append(
            {
                "outline_mask": compute_mask_outline(path_mask),
                "label_anchor": find_label_anchor(path_mask),
                "path_coverage": path_coverage,
            }
        )

    if debug_path_stats_dir is not None:
        stats_path = (debug_path_stats_dir / svg_relative).with_suffix(".csv")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path_index", "path_id", "path_iou", "path_coverage", "path_area", "keep"],
            )
            writer.writeheader()
            writer.writerows(path_stats)

    save_removed_paths_debug(
        svg_relative=svg_relative,
        debug_removed_paths_dir=debug_removed_paths_dir,
        subject_mask=subject_mask,
        removed_path_stats=removed_path_stats,
        subject_coverage_threshold=subject_coverage_threshold,
    )

    return kept_paths, removed


def write_svg(tree: ET.ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def process_svg(
    svg_path: Path,
    input_dir: Path,
    image_dir: Path,
    output_dir: Path,
    debug_mask_dir: Path | None,
    debug_path_stats_dir: Path | None,
    debug_removed_paths_dir: Path | None,
    keep_layer_sets: list[tuple[int, ...]],
    layer_iou_threshold: float,
    subject_coverage_threshold: float,
    scale_limit: float,
    min_canvas_size: int,
    sam_model_type: str,
    sam_checkpoint: Path,
    sam_device: str,
    direct_output: bool = False,
) -> tuple[int, list[tuple[tuple[int, ...], int, int]]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    if strip_ns(root.tag) != "svg":
        raise ValueError(f"{svg_path} is not a valid SVG root.")

    relative = svg_path.relative_to(input_dir)
    path_elements = get_path_elements(root)
    if not path_elements:
        if direct_output:
            write_svg(tree, output_dir / relative)
            return 0, [(keep_layer_sets[0], 0, 0)]
        for keep_layers in keep_layer_sets:
            group_dir = output_dir / format_keep_layers_dirname(keep_layers)
            write_svg(tree, group_dir / "subject_filtered" / relative)
            write_svg(tree, group_dir / "subject_filtered_layer_filtered" / relative)
        return 0, [(keep_layers, 0, 0) for keep_layers in keep_layer_sets]

    view_box = parse_viewbox(root)
    root_attrs = dict(root.attrib)
    image_path = find_image_for_svg(svg_path, input_dir, image_dir)
    subject_mask = build_subject_mask(
        image_path,
        model_type=sam_model_type,
        checkpoint_path=sam_checkpoint,
        device=sam_device,
        input_dir=image_dir,
        debug_mask_dir=debug_mask_dir,
    )
    subject_canvas_size = (subject_mask.shape[1], subject_mask.shape[0])
    subject_path_masks = render_masks_for_paths(path_elements, root_attrs, view_box, subject_canvas_size)
    path_elements, subject_removed = filter_paths_by_subject(
        root=root,
        path_elements=path_elements,
        path_masks=subject_path_masks,
        svg_relative=relative,
        debug_path_stats_dir=debug_path_stats_dir,
        debug_removed_paths_dir=debug_removed_paths_dir,
        subject_mask=subject_mask,
        subject_coverage_threshold=subject_coverage_threshold,
    )

    if not path_elements:
        layer_results = []
        if direct_output:
            write_svg(tree, output_dir / relative)
            return subject_removed, [(keep_layer_sets[0], 0, 0)]
        for keep_layers in keep_layer_sets:
            group_dir = output_dir / format_keep_layers_dirname(keep_layers)
            write_svg(tree, group_dir / "subject_filtered" / relative)
            write_svg(tree, group_dir / "subject_filtered_layer_filtered" / relative)
            layer_results.append((keep_layers, 0, 0))
        return subject_removed, layer_results

    canvas_size = compute_canvas_size(view_box, scale_limit=scale_limit, min_canvas_size=min_canvas_size)
    layer_path_masks = render_masks_for_paths(path_elements, root_attrs, view_box, canvas_size)
    depths = build_depths(layer_path_masks, layer_iou_threshold)

    layer_results: list[tuple[tuple[int, ...], int, int]] = []
    if direct_output:
        keep_layers = keep_layer_sets[0]
        layer_root = copy.deepcopy(root)
        layer_tree = ET.ElementTree(layer_root)
        layer_path_elements = get_path_elements(layer_root)
        layer_parent_map = build_parent_map(layer_root)
        keep_layer_set = set(keep_layers)
        kept = 0
        layer_removed = 0

        for path_el, depth in zip(layer_path_elements, depths):
            if depth in keep_layer_set:
                kept += 1
                continue
            parent = layer_parent_map.get(id(path_el))
            if parent is not None:
                parent.remove(path_el)
                layer_removed += 1

        write_svg(layer_tree, output_dir / relative)
        return subject_removed, [(keep_layers, kept, layer_removed)]

    for keep_layers in keep_layer_sets:
        group_dir = output_dir / format_keep_layers_dirname(keep_layers)
        subject_output_path = group_dir / "subject_filtered" / relative
        layer_output_path = group_dir / "subject_filtered_layer_filtered" / relative

        subject_tree = ET.ElementTree(copy.deepcopy(root))
        write_svg(subject_tree, subject_output_path)

        layer_root = copy.deepcopy(root)
        layer_tree = ET.ElementTree(layer_root)
        layer_path_elements = get_path_elements(layer_root)
        layer_parent_map = build_parent_map(layer_root)
        keep_layer_set = set(keep_layers)
        kept = 0
        layer_removed = 0

        for path_el, depth in zip(layer_path_elements, depths):
            if depth in keep_layer_set:
                kept += 1
                continue
            parent = layer_parent_map.get(id(path_el))
            if parent is not None:
                parent.remove(path_el)
                layer_removed += 1

        write_svg(layer_tree, layer_output_path)
        layer_results.append((keep_layers, kept, layer_removed))

    return subject_removed, layer_results


def collect_svg_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.svg" if recursive else "*.svg"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def filter_svg_dataset(
    input_dir: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    keep_layers: list[int] | list[list[int]] | None = None,
    recursive: bool = False,
    debug_mask_dir: str | Path | None = None,
    debug_path_stats_dir: str | Path | None = None,
    debug_removed_paths_dir: str | Path | None = None,
    layer_iou_threshold: float = LAYER_IOU_THRESHOLD,
    subject_coverage_threshold: float = SUBJECT_COVERAGE_THRESHOLD,
    scale_limit: float = SCALE_LIMIT,
    min_canvas_size: int = MIN_CANVAS_SIZE,
    sam_model_type: str = SAM_MODEL_TYPE,
    sam_checkpoint: str | Path | None = None,
    sam_device: str = SAM_DEVICE,
    direct_output: bool = False,
) -> int:
    """
    Filter a directory of SVGs using the paired raster originals.

    When direct_output is False, output_dir is treated as a base directory and
    grouped results are written under `layerXYZ/...`.

    When direct_output is True, exactly one keep-layer set must be provided and
    the final filtered SVG is written directly to output_dir using the original
    relative filename.
    """
    input_dir = Path(input_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    debug_mask_dir = Path(debug_mask_dir) if debug_mask_dir else None
    debug_path_stats_dir = Path(debug_path_stats_dir) if debug_path_stats_dir else None
    debug_removed_paths_dir = Path(debug_removed_paths_dir) if debug_removed_paths_dir else None
    keep_layer_sets = normalize_keep_layer_sets(keep_layers or KEEP_LAYERS)
    sam_checkpoint = resolve_sam_checkpoint(
        model_type=sam_model_type,
        checkpoint_path=Path(sam_checkpoint) if sam_checkpoint else None,
    )
    sam_device = resolve_sam_device(sam_device)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not keep_layer_sets:
        raise ValueError("KEEP_LAYERS must contain at least one layer set.")
    if direct_output and len(keep_layer_sets) != 1:
        raise ValueError("direct_output requires exactly one keep-layer set.")

    svg_files = collect_svg_files(input_dir, recursive)
    if not svg_files:
        raise FileNotFoundError(f"No SVG files found under: {input_dir}")

    if debug_mask_dir is not None:
        print(f"Debug mask dir: {debug_mask_dir}")
    if debug_path_stats_dir is not None:
        print(f"Debug path stats dir: {debug_path_stats_dir}")
    if debug_removed_paths_dir is not None:
        print(f"Debug removed paths dir: {debug_removed_paths_dir}")

    success = 0
    for svg_path in svg_files:
        relative = svg_path.relative_to(input_dir)
        subject_removed, layer_results = process_svg(
            svg_path=svg_path,
            input_dir=input_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            debug_mask_dir=debug_mask_dir,
            debug_path_stats_dir=debug_path_stats_dir,
            debug_removed_paths_dir=debug_removed_paths_dir,
            keep_layer_sets=keep_layer_sets,
            layer_iou_threshold=layer_iou_threshold,
            subject_coverage_threshold=subject_coverage_threshold,
            scale_limit=scale_limit,
            min_canvas_size=min_canvas_size,
            sam_model_type=sam_model_type,
            sam_checkpoint=sam_checkpoint,
            sam_device=sam_device,
            direct_output=direct_output,
        )
        success += 1
        layer_summary = ", ".join(
            f"{list(layer_set)}: kept {kept}, removed {layer_removed}"
            for layer_set, kept, layer_removed in layer_results
        )
        print(f"[OK] {relative} -> removed {subject_removed} subject paths; {layer_summary}")

    print(f"Completed: {success}/{len(svg_files)} files succeeded.")
    return success


def main():
    filter_svg_dataset(
        input_dir=INPUT_DIR,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        keep_layers=KEEP_LAYERS,
        recursive=RECURSIVE,
        debug_mask_dir=DEBUG_MASK_DIR,
        debug_path_stats_dir=DEBUG_PATH_STATS_DIR,
        debug_removed_paths_dir=DEBUG_REMOVED_PATHS_DIR,
        layer_iou_threshold=LAYER_IOU_THRESHOLD,
        subject_coverage_threshold=SUBJECT_COVERAGE_THRESHOLD,
        scale_limit=SCALE_LIMIT,
        min_canvas_size=MIN_CANVAS_SIZE,
        sam_model_type=SAM_MODEL_TYPE,
        sam_checkpoint=SAM_CHECKPOINT,
        sam_device=SAM_DEVICE,
    )


if __name__ == "__main__":
    main()
