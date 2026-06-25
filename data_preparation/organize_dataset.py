import json
import os
import sys
import glob
import io as io2
from pathlib import Path
from PIL import Image
import cairosvg
import numpy as np
import xml.etree.ElementTree as ET
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, disk
from skimage import color, io
from skimage.filters import threshold_otsu
from typing import Optional

# ==================== User Config ====================
# Manually set the style name and base directory here.
STYLE_NAME = "ns2（hui）_origami-animal-style"
BASE_DIR = "./test_assets"
# ====================================================

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
SHAPE_TAGS = {"path", "rect", "circle", "ellipse", "polygon", "polyline", "line"}


# ==================== SVG to PNG ====================
def svg_to_png_convert(input_dir, output_dir, width=512, height=512, bg_color="#FFFFFF"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.svg"))

    for svg_file in files:
        out_file = Path(output_dir) / (svg_file.stem + ".png")
        svg_bytes = svg_file.read_bytes()
        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=width, output_height=height)

        img = Image.open(io2.BytesIO(png_bytes)).convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)

        bg = Image.new("RGBA", (width, height), bg_color)
        bg.paste(img, (0, 0), img)

        final = bg.convert("RGB")
        final.save(out_file, format="PNG", optimize=True)


# ==================== PNG to Grayscale ====================
def convert_folder_to_grayscale(folder):
    if not os.path.exists(folder):
        print(f"[Gray] Warning: directory does not exist: {folder}")
        return

    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ])

    if not files:
        print(f"[Gray] Warning: no images found in directory: {folder}")
        return

    for fname in files:
        path = os.path.join(folder, fname)
        with Image.open(path) as img:
            gray = img.convert("L")          # Grayscale
            gray = gray.convert("RGB")       # Keep three channels for downstream compatibility
            gray.save(path, optimize=True)



# ==================== Backbone Extraction ====================
def normalize_svg_bytes(svg_bytes):
    FILL_COLOR = "#FFFFFF"
    STROKE_COLOR = "#000000"
    STROKE_WIDTH = "2"
    try:
        if not svg_bytes.strip(): return svg_bytes
        tree = ET.ElementTree(ET.fromstring(svg_bytes))
        root = tree.getroot()
        for style in root.findall(f"{{{SVG_NS}}}style"): root.remove(style)
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag in SHAPE_TAGS:
                elem.set("fill", FILL_COLOR)
                elem.set("stroke", STROKE_COLOR)
                elem.set("stroke-width", STROKE_WIDTH)
                elem.set("vector-effect", "non-scaling-stroke")
                elem.attrib.pop("class", None)
                elem.attrib.pop("style", None)
        buf = io2.BytesIO()
        tree.write(buf, encoding="utf-8", xml_declaration=True)
        return buf.getvalue()
    except Exception as e:
        print(f"Failed to normalize SVG: {e}")
        return svg_bytes


def render_svg_to_binary(svg_path, width=512, height=512):
    svg_bytes = Path(svg_path).read_bytes()
    svg_bytes = normalize_svg_bytes(svg_bytes)
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=width, output_height=height)
    img = Image.open(io2.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", (width, height), "#000000")
    bg.paste(img, (0, 0), img)
    rgb_img = bg.convert("RGB")
    gray = color.rgb2gray(np.array(rgb_img))
    try:
        thresh = threshold_otsu(gray)
    except ValueError:
        thresh = 0.5
    return gray > thresh


def backbone_process(input_dir, output_dir, width=512, height=512):
    EROSION_RADIUS, EDGE_RADIUS, SKELETON_THICKNESS = 25, 3, 2
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    files = sorted(glob.glob(os.path.join(input_dir, "*.svg")))
    for svg_path in files:
        base_name = os.path.splitext(os.path.basename(svg_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}.png")
        binary = render_svg_to_binary(svg_path, width, height)
        merged_eroded = binary_erosion(binary, disk(EROSION_RADIUS))
        skel = skeletonize(binary)
        merged_skeleton = binary_dilation(skel, disk(SKELETON_THICKNESS))
        h, w = merged_eroded.shape
        final = np.ones((h, w, 3), np.uint8) * 255
        final[merged_eroded] = [180, 180, 180]
        final[merged_skeleton] = [0, 0, 0]
        io.imsave(out_path, final, check_contrast=False)


# ==================== Concatenation ====================
def resize_image_for_concat(image, target_size, fill_color=(255, 255, 255)):
    original_width, original_height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, fill_color)
    new_img.paste(image, ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2))
    return new_img


def numeric_sort_key(name: str):
    stem = os.path.splitext(os.path.basename(name))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def concat_images(folder_a, folder_b, output_folder, target_size=(512, 512), manifest_path=None):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    b_files = {os.path.splitext(f)[0]: os.path.join(folder_b, f) for f in os.listdir(folder_b) if
               os.path.splitext(f)[1].lower() in supported_extensions} if os.path.exists(folder_b) else {}
    if not b_files:
        print(f"[Concat] Warning: no supported images found in target directory {folder_b}; skipping concatenation.")
        return
    combined_images = []
    pair_metadata = []
    a_files_list = sorted(
        [f for f in os.listdir(folder_a) if os.path.splitext(f)[1].lower() in supported_extensions],
        key=numeric_sort_key,
    )
    for a_file in a_files_list:
        base_name = os.path.splitext(a_file)[0]
        b_path = b_files.get(base_name)
        if not b_path and base_name.endswith('_output'): b_path = b_files.get(base_name[:-7])
        if not b_path: continue
        with Image.open(os.path.join(folder_a, a_file)) as img_a, Image.open(b_path) as img_b:
            img_a = resize_image_for_concat(img_a.convert('RGB'), target_size)
            img_b = resize_image_for_concat(img_b.convert('RGB'), target_size)
            h_combined = Image.new('RGB', (target_size[0] * 2, target_size[1]), (255, 255, 255))
            h_combined.paste(img_a, (0, 0));
            h_combined.paste(img_b, (target_size[0], 0))
            combined_images.append(h_combined)
            pair_metadata.append(
                {
                    "left_image_stem": base_name,
                    "left_image_name": a_file,
                    "right_image_stem": os.path.splitext(os.path.basename(b_path))[0],
                    "right_image_name": os.path.basename(b_path),
                }
            )
    v_combined_count = 0
    manifest = {}
    for i in range(0, len(combined_images), 2):
        if i + 1 < len(combined_images):
            img1, img2 = combined_images[i], combined_images[i + 1]
            meta1, meta2 = pair_metadata[i], pair_metadata[i + 1]
        elif len(combined_images) > 1:
            img1, img2 = combined_images[i], combined_images[0]
            meta1, meta2 = pair_metadata[i], pair_metadata[0]
        else:
            break
        v_combined = Image.new('RGB', (target_size[0] * 2, target_size[1] * 2), (255, 255, 255))
        v_combined.paste(img1, (0, 0));
        v_combined.paste(img2, (0, target_size[1]))
        output_name = f"{v_combined_count}.jpg"
        v_combined.save(os.path.join(output_folder, output_name), quality=95)
        manifest[output_name] = {
            "top_left_stem": meta1["left_image_stem"],
            "top_left_name": meta1["left_image_name"],
            "top_right_stem": meta1["right_image_stem"],
            "top_right_name": meta1["right_image_name"],
            "bottom_left_stem": meta2["left_image_stem"],
            "bottom_left_name": meta2["left_image_name"],
            "bottom_right_source": "generated",
        }
        v_combined_count += 1
    if manifest_path is not None:
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, ensure_ascii=False)


def create_prompts_for_concat_folder(concat_folder: str, prompt_text: str) -> Optional[str]:
    if not os.path.exists(concat_folder):
        return None

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    files = sorted([f for f in os.listdir(concat_folder) if os.path.splitext(f)[1].lower() in supported_extensions])
    if not files:
        return None

    first_txt_path = None
    for fname in files:
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(concat_folder, f"{base}.txt")
        with open(txt_path, 'w', encoding='utf-8') as fp:
            fp.write(prompt_text)
        if first_txt_path is None:
            first_txt_path = txt_path
    return first_txt_path


# ==================== Main ====================
def organize(style_name=None, base_dir=None):
    style_name = style_name or STYLE_NAME
    base_dir = base_dir or BASE_DIR
    style_dir = os.path.join(base_dir, style_name)
    if not os.path.isdir(style_dir):
        print(f"Error: style directory not found: '{style_dir}'")
        sys.exit(1)

    original_dir = os.path.join(style_dir, "original")
    svg_dir = os.path.join(style_dir, "proxy_svg")
    svg2png_dir = os.path.join(style_dir, "proxy_svg2png")
    backbone_dir = os.path.join(style_dir, "backbone")
    concat_a_vat_dir = os.path.join(style_dir, "A-VAT_train_Data")
    concat_s_vat_dir = os.path.join(style_dir, "S-VAT_train_Data")

    if not os.path.exists(svg_dir):
        print(f"Error: SVG directory not found: '{svg_dir}'")
        sys.exit(1)

    print("[Organize] proxy rasters")
    svg_to_png_convert(svg_dir, svg2png_dir)
    convert_folder_to_grayscale(svg2png_dir)

    print("[Organize] backbone")
    backbone_process(svg_dir, backbone_dir)

    print("[Organize] A-VAT")
    concat_images(
        backbone_dir,
        svg2png_dir,
        concat_a_vat_dir,
        manifest_path=os.path.join(concat_a_vat_dir, "manifest.json"),
    )
    prompt_a_vat = (
        "This is a four-panel image on a uniform solid-color background, hand-drawn in style, with the subject highlighted and kept as simple as possible: "
        "[TOP-LEFT]: Image of the skeleton of a subject. "
        "[TOP-RIGHT]: An edited version of the [TOP-LEFT] image, transformed to [styvec] style. "
        "[BOTTOM-LEFT]: Skeleton image of another subject. "
        "[BOTTOM-RIGHT]: An edited version of the [BOTTOM-LEFT] image, applying the same style transformation as used in [TOP-RIGHT]."
    )
    create_prompts_for_concat_folder(concat_a_vat_dir, prompt_a_vat)

    print("[Organize] S-VAT")
    if os.path.exists(original_dir):
        concat_images(svg2png_dir, original_dir, concat_s_vat_dir)
        prompt_s_vat = (
            "This is a four-panel image on a uniform solid-color background, hand-drawn in style, with the subject highlighted and kept as simple as possible: "
            "[TOP-LEFT]: Image of the structure of a subject. "
            "[TOP-RIGHT]: An edited version of the [TOP-LEFT] image, transformed to [styvec] style. "
            "[BOTTOM-LEFT]: Structural image of another subject. "
            "[BOTTOM-RIGHT]: An edited version of the [BOTTOM-LEFT] image, applying the same style transformation as used in [TOP-RIGHT]."
        )
        create_prompts_for_concat_folder(concat_s_vat_dir, prompt_s_vat)
    else:
        print(f"Skip S-VAT concat: missing directory '{original_dir}'")


if __name__ == "__main__":
    organize(style_name="ns18（hui）_shape-style", base_dir="../dataset")
