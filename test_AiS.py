import argparse
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cairosvg
import numpy as np
import torch
from PIL import Image
from skimage import io as skio
from skimage.morphology import binary_dilation, binary_erosion, disk, skeletonize

SRC_DIR = Path(__file__).resolve().parent / "src"
QUANTIZER_DIR = SRC_DIR / "diffusers" / "quantizers"
sys.path = [path for path in sys.path if Path(path or ".").resolve() != QUANTIZER_DIR]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from data_preparation.lv.layered_vectorization import vectorize_image
from data_preparation.organize_dataset import render_svg_to_binary
from data_preparation.process_test_data import SUPPORTED_IMAGE_EXTENSIONS, concat_test_img, crop_bottom_right
from data_preparation.svg_layer_filter import filter_svg_dataset


A_VAT_PROMPT = (
    "This is a four-panel image on a uniform solid-color background, hand-drawn in style, "
    "with the subject highlighted and kept as simple as possible: "
    "[TOP-LEFT]: Image of the skeleton of a subject. "
    "[TOP-RIGHT]: An edited version of the [TOP-LEFT] image, transformed to [styvec] style. "
    "[BOTTOM-LEFT]: Skeleton image of another subject. "
    "[BOTTOM-RIGHT]: An edited version of the [BOTTOM-LEFT] image, applying the same style "
    "transformation as used in [TOP-RIGHT]."
)

S_VAT_PROMPT = (
    "This is a four-panel image on a uniform solid-color background, hand-drawn in style, "
    "with the subject highlighted and kept as simple as possible: "
    "[TOP-LEFT]: Image of the structure of a subject. "
    "[TOP-RIGHT]: An edited version of the [TOP-LEFT] image, transformed to [styvec] style. "
    "[BOTTOM-LEFT]: Structural image of another subject. "
    "[BOTTOM-RIGHT]: An edited version of the [BOTTOM-LEFT] image, applying the same style "
    "transformation as used in [TOP-RIGHT]."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX fill inference for one style across test images.")
    parser.add_argument("--style", default="Fluffy_Brush", help="Style folder name under the dataset root.")
    parser.add_argument("--stage", choices=("all", "A-VAT", "S-VAT"), default="all", help="Which inference stage to run.")
    parser.add_argument("--ref-indices", nargs="+", type=int, default=[1], help="Reference image indices to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset-root", default="dataset", help="Dataset root directory.")
    parser.add_argument("--input-dir", default="test_assets/input_images", help="Directory of input test images.")
    parser.add_argument("--output-dir", default="test_assets/generated_images", help="Directory for generated outputs.")
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-Fill-dev", help="Path or model id for the base FLUX fill model.")
    parser.add_argument("--mask-path", default=None, help="Optional mask image path.")
    parser.add_argument("--vectorize-config", default=None, help="Optional YAML config path for layered vectorization.")
    parser.add_argument("--input-pattern", default="*", help="Glob pattern for selecting input images inside --input-dir.")
    parser.add_argument("--a-vat-scale", type=float, default=1.0, help="Input content scale for A-VAT inference.")
    parser.add_argument("--s-vat-scale", type=float, default=1.0, help="Input content scale for S-VAT inference.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--max-sequence-length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="Torch dtype.")
    parser.add_argument("--device-map", default="balanced", help="Diffusers device_map value.")
    parser.add_argument("--cuda-visible-devices", default="0", help="CUDA_VISIBLE_DEVICES value.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    style_dir = Path(args.dataset_root) / args.style
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_path = Path(args.mask_path) if args.mask_path else Path(args.dataset_root) / "mask_1024.png"

    if not style_dir.is_dir():
        raise FileNotFoundError(f"Style directory not found: {style_dir}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    required_weights = []
    if args.stage in ("all", "A-VAT"):
        required_weights.append(style_dir / "A-VAT_checkpoint" / "pytorch_lora_weights.safetensors")
    if args.stage in ("all", "S-VAT"):
        required_weights.append(style_dir / "S-VAT_checkpoint" / "pytorch_lora_weights.safetensors")
    for weights_path in required_weights:
        if not weights_path.is_file():
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

    input_images = sorted(
        path
        for path in input_dir.glob(args.input_pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not input_images:
        raise FileNotFoundError(f"No input images found in {input_dir} matching {args.input_pattern}")

    test_vectorized_dir = input_dir / "input_img_vectorized"
    test_backbone_dir = input_dir / "backbone"
    if args.stage == "all":
        test_vectorized_dir.mkdir(parents=True, exist_ok=True)
        test_backbone_dir.mkdir(parents=True, exist_ok=True)

        def file_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def metadata_path_for(image_path: Path) -> Path:
            return test_vectorized_dir / f"{image_path.stem}.json"

        def is_cache_current(image_path: Path, vectorized_path: Path, backbone_path: Path) -> bool:
            metadata_path = metadata_path_for(image_path)
            if not vectorized_path.is_file() or not backbone_path.is_file() or not metadata_path.is_file():
                return False
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return False
            return metadata.get("sha256") == file_sha256(image_path)

        def write_metadata(image_path: Path) -> None:
            metadata_path_for(image_path).write_text(
                json.dumps({"source_name": image_path.name, "sha256": file_sha256(image_path)}, indent=2),
                encoding="utf-8",
            )

        stale_inputs = []
        for input_image_path in input_images:
            vectorized_path = test_vectorized_dir / f"{input_image_path.stem}.svg"
            backbone_path = test_backbone_dir / f"{input_image_path.stem}.png"
            if not is_cache_current(input_image_path, vectorized_path, backbone_path):
                stale_inputs.append(input_image_path)

        if stale_inputs:
            print(f"[Preprocess] Refreshing cached vectorization/backbone for {len(stale_inputs)} input image(s)")
            filter_input_dir = Path(tempfile.mkdtemp(prefix="test_vectorized_filter_"))
            try:
                for input_image_path in stale_inputs:
                    vectorized_path = test_vectorized_dir / f"{input_image_path.stem}.svg"
                    backbone_path = test_backbone_dir / f"{input_image_path.stem}.png"
                    metadata_path = metadata_path_for(input_image_path)
                    if vectorized_path.exists():
                        vectorized_path.unlink()
                    if backbone_path.exists():
                        backbone_path.unlink()
                    if metadata_path.exists():
                        metadata_path.unlink()
                    vectorize_image(
                        target_image=str(input_image_path),
                        output_svg_path=str(filter_input_dir / f"{input_image_path.stem}.svg"),
                        config_path=args.vectorize_config,
                    )

                filter_svg_dataset(
                    input_dir=filter_input_dir,
                    image_dir=input_dir,
                    output_dir=filter_input_dir,
                    keep_layers=[[0, 1, 2]],
                    recursive=False,
                    debug_mask_dir=None,
                    debug_path_stats_dir=None,
                    debug_removed_paths_dir=None,
                    direct_output=True,
                )

                for input_image_path in stale_inputs:
                    vectorized_path = test_vectorized_dir / f"{input_image_path.stem}.svg"
                    filtered_svg_path = filter_input_dir / f"{input_image_path.stem}.svg"
                    if not filtered_svg_path.is_file():
                        raise FileNotFoundError(f"Filtered SVG not found: {filtered_svg_path}")
                    shutil.copyfile(filtered_svg_path, vectorized_path)

                    backbone_path = test_backbone_dir / f"{input_image_path.stem}.png"
                    binary = render_svg_to_binary(vectorized_path, width=512, height=512)
                    merged_eroded = binary_erosion(binary, disk(25))
                    merged_skeleton = binary_dilation(skeletonize(binary), disk(2))
                    height, width = merged_eroded.shape
                    backbone_image = np.ones((height, width, 3), dtype=np.uint8) * 255
                    backbone_image[merged_eroded] = [180, 180, 180]
                    backbone_image[merged_skeleton] = [0, 0, 0]
                    skio.imsave(backbone_path, backbone_image, check_contrast=False)
                    write_metadata(input_image_path)
            finally:
                shutil.rmtree(filter_input_dir, ignore_errors=True)
    elif args.stage == "A-VAT":
        print("[Preprocess] A-VAT mode: skipping vectorization and backbone extraction, using input_images directly.")
    else:
        print("[Preprocess] S-VAT mode: skipping vectorization and backbone extraction, using input_images directly.")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_image = load_image(str(mask_path))
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]

    print(f"Running style={args.style} stage={args.stage} refs={args.ref_indices}")

    a_vat_pipeline = None
    s_vat_pipeline = None
    if args.stage in ("all", "A-VAT"):
        a_vat_pipeline = FluxFillPipeline.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map=args.device_map)
        a_vat_pipeline.load_lora_weights(str(style_dir / "A-VAT_checkpoint" / "pytorch_lora_weights.safetensors"))
    if args.stage in ("all", "S-VAT"):
        s_vat_pipeline = FluxFillPipeline.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map=args.device_map)
        s_vat_pipeline.load_lora_weights(str(style_dir / "S-VAT_checkpoint" / "pytorch_lora_weights.safetensors"))

    try:
        for ref_index in args.ref_indices:
            for input_image_path in input_images:
                base_name = f"{input_image_path.stem}_{args.style}_ref{ref_index}_seed{args.seed}"

                a_vat_output = None
                if a_vat_pipeline is not None:
                    a_vat_input = concat_test_img(
                        input_image=input_image_path if args.stage == "A-VAT" else test_backbone_dir / f"{input_image_path.stem}.png",
                        style_dir=style_dir,
                        phase="A-VAT",
                        ref_img_idx=ref_index,
                        content_scale=args.a_vat_scale,
                    )
                    a_vat_output = a_vat_pipeline(
                        prompt=A_VAT_PROMPT,
                        image=a_vat_input,
                        mask_image=mask_image,
                        height=args.height,
                        width=args.width,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_steps,
                        max_sequence_length=args.max_sequence_length,
                        generator=torch.Generator("cpu").manual_seed(args.seed),
                    ).images[0]
                    a_vat_output.save(output_dir / f"{base_name}_A-VAT_output.png")

                if s_vat_pipeline is None:
                    continue

                s_vat_source = input_image_path
                if args.stage == "all":
                    if a_vat_output is None:
                        raise RuntimeError("Missing A-VAT output for chained S-VAT inference.")
                    s_vat_source = crop_bottom_right(a_vat_output.convert("RGB"))
                elif args.stage == "S-VAT":
                    s_vat_source = input_image_path

                s_vat_input = concat_test_img(
                    input_image=s_vat_source,
                    style_dir=style_dir,
                    phase="S-VAT",
                    ref_img_idx=ref_index,
                    content_scale=args.s_vat_scale,
                )
                s_vat_output = s_vat_pipeline(
                    prompt=S_VAT_PROMPT,
                    image=s_vat_input,
                    mask_image=mask_image,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_steps,
                    max_sequence_length=args.max_sequence_length,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                ).images[0]
                s_vat_output.save(output_dir / f"{base_name}_S-VAT_output.png")
    finally:
        if a_vat_pipeline is not None:
            del a_vat_pipeline
        if s_vat_pipeline is not None:
            del s_vat_pipeline
        gc.collect()
        torch.cuda.empty_cache()
