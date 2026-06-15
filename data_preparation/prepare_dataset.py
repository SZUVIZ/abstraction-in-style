import argparse
import shutil
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_SRC = str((PROJECT_ROOT / "src").resolve())
sys.path = [path for path in sys.path if str(Path(path or ".").resolve()) != PROJECT_SRC]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_preparation.organize_dataset import organize
from data_preparation.svg_layer_filter import filter_svg_dataset

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ORIGINAL_BACKUP_DIRNAME = "original_backup"


def find_source_images(original_dir: Path):
    return sorted(
        [
            path
            for path in original_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def resize_original_images_in_place(image_paths, backup_dir: Path, target_size=(512, 512)):
    images_to_resize = []

    for image_path in image_paths:
        with Image.open(image_path) as img:
            if img.size != target_size:
                images_to_resize.append(image_path)

    if not images_to_resize:
        return

    backup_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        backup_path = backup_dir / image_path.name
        if not backup_path.exists():
            shutil.copyfile(image_path, backup_path)

    for image_path in images_to_resize:
        with Image.open(image_path) as img:
            resized = img.resize(target_size, Image.Resampling.LANCZOS)
            resized.save(image_path)

    print(
        f"[Prepare] backed up {len(image_paths)} originals and resized "
        f"{len(images_to_resize)} images to {target_size[0]}x{target_size[1]}"
    )


def vectorize_and_filter_images(
    image_dir: Path,
    proxy_svg_dir: Path,
    config_path: str = None,
    resize_inputs: bool = False,
    backup_dir: Path = None,
):
    from data_preparation.lv.layered_vectorization import vectorize_image

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    source_images = find_source_images(image_dir)
    if not source_images:
        raise FileNotFoundError(f"No source images found in: {image_dir}")

    if resize_inputs:
        if backup_dir is None:
            raise ValueError("backup_dir is required when resize_inputs=True")
        resize_original_images_in_place(source_images, backup_dir)

    proxy_svg_dir.mkdir(parents=True, exist_ok=True)

    print("[Prepare] vectorizing")
    for image_path in source_images:
        output_svg_path = proxy_svg_dir / f"{image_path.stem}.svg"
        vectorize_image(
            target_image=str(image_path),
            output_svg_path=str(output_svg_path),
            config_path=config_path,
        )

    print("[Prepare] filtering")
    filter_svg_dataset(
        input_dir=proxy_svg_dir,
        image_dir=image_dir,
        output_dir=proxy_svg_dir,
        keep_layers=[[0, 1, 2]],
        recursive=False,
        debug_mask_dir=None,
        debug_path_stats_dir=None,
        debug_removed_paths_dir=None,
        direct_output=True,
    )
    return source_images


def prepare_dataset(style_name: str, base_dir: str = "dataset", config_path: str = None):
    base_dir_path = Path(base_dir)
    if not base_dir_path.is_absolute():
        base_dir_path = PROJECT_ROOT / base_dir_path

    style_dir = base_dir_path / style_name
    original_dir = style_dir / "original"
    original_backup_dir = style_dir / ORIGINAL_BACKUP_DIRNAME
    proxy_svg_dir = style_dir / "proxy_svg"
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Original image directory not found: {original_dir}")

    source_images = find_source_images(original_dir)
    print(f"[Prepare] style={style_name} images={len(source_images)}")
    vectorize_and_filter_images(
        image_dir=original_dir,
        proxy_svg_dir=proxy_svg_dir,
        config_path=config_path,
        resize_inputs=True,
        backup_dir=original_backup_dir,
    )

    print("[Prepare] organizing")
    organize(style_name=style_name, base_dir=str(base_dir_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a style dataset from original raster images.")
    parser.add_argument(
        "style_name",
        nargs="?",
        default="Fluffy_Brush",
        help="Style folder name under the base directory, e.g. Fluffy_Brush",
    )
    parser.add_argument(
        "--base-dir",
        default="dataset",
        help="Dataset base directory that contains the style folder.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path for layered vectorization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_dataset(args.style_name, base_dir=args.base_dir, config_path=args.config)


if __name__ == "__main__":
    main()
