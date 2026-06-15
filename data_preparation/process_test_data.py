import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
PHASE_REFERENCE_DIRS = {
    "A-VAT": ("backbone", "proxy_svg2png"),
    "S-VAT": ("proxy_svg2png", "original"),
}


def find_stem_image(directory: Path, stem: int | str) -> Path:
    stem = str(stem)
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image '{stem}' in {directory}")


def crop_bottom_right(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop((width // 2, height // 2, width, height))


def concat_test_img(
    input_image: str | Path | Image.Image,
    style_dir: str | Path,
    phase: str,
    ref_img_idx: int = 1,
    content_scale: float = 1.0,
    panel_size: int = 512,
) -> Image.Image:
    style_dir = Path(style_dir)
    if phase not in PHASE_REFERENCE_DIRS:
        raise ValueError(f"Unsupported phase: {phase}")

    left_dir, right_dir = PHASE_REFERENCE_DIRS[phase]
    top_left_path = find_stem_image(style_dir / left_dir, ref_img_idx)
    top_right_path = find_stem_image(style_dir / right_dir, ref_img_idx)

    with Image.open(top_right_path) as reference_source:
        background_color = reference_source.convert("RGB").getpixel((10, 10))

    def prepare(image_source: str | Path | Image.Image) -> Image.Image:
        if isinstance(image_source, Image.Image):
            image = image_source.copy()
        else:
            image = Image.open(image_source)

        try:
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, background_color)
                background.paste(image, mask=image.split()[-1])
                image = background
            return image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        finally:
            if not isinstance(image_source, Image.Image):
                image.close()

    top_left = prepare(top_left_path)
    top_right = prepare(top_right_path)
    bottom_left = prepare(input_image)

    if content_scale != 1.0:
        scaled_width = int(bottom_left.width * content_scale)
        scaled_height = int(bottom_left.height * content_scale)
        scaled = bottom_left.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
        bottom_left = Image.new("RGB", (panel_size, panel_size), (255, 255, 255))
        bottom_left.paste(
            scaled,
            ((panel_size - scaled_width) // 2, (panel_size - scaled_height) // 2),
        )

    canvas = Image.new("RGB", (panel_size * 2, panel_size * 2), background_color)
    canvas.paste(top_left, (0, 0))
    canvas.paste(top_right, (panel_size, 0))
    canvas.paste(bottom_left, (0, panel_size))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a four-panel FLUX test input image.")
    parser.add_argument("--style-dir", required=True, help="Style directory under the dataset root.")
    parser.add_argument("--phase", choices=sorted(PHASE_REFERENCE_DIRS), required=True, help="Reference phase to use.")
    parser.add_argument("--ref-index", type=int, default=1, help="Reference image index.")
    parser.add_argument("--input-image", required=True, help="Input image path.")
    parser.add_argument("--output-image", required=True, help="Output image path.")
    parser.add_argument("--content-scale", type=float, default=1.0, help="Scale factor for the input panel.")
    parser.add_argument("--panel-size", type=int, default=512, help="Panel size in pixels.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_test_img(
        input_image=Path(args.input_image),
        style_dir=Path(args.style_dir),
        phase=args.phase,
        ref_img_idx=args.ref_index,
        content_scale=args.content_scale,
        panel_size=args.panel_size,
    ).save(output_path)
