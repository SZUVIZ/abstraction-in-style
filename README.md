# Abstraction in Style: Beyond Texture and Color

Official code for the paper [**Abstraction in Style: Beyond Texture and Color**](https://arxiv.org/abs/2603.29924).

AiS models style transfer in two stages:

- `A-VAT`: structural abstraction
- `S-VAT`: visual stylization

## Installation

Environment setup commands are provided in [install_env.sh](/mnt/d/hyf_workspace/Abstraction_in_Style/install_env.sh).

```bash
conda create -n AiS python=3.10 -y
conda activate AiS
bash install_env.sh
```

## Data Preparation

For each style, you only need to prepare:

```text
dataset/<style_name>/original/
```

Put each object cropped from the style images into this folder. Each object should be placed on a clean, centered background.

Example:

```text
dataset/Fluffy_Brush/original/
├── 1.jpg
├── 2.jpg
├── 3.jpg
└── ...
```

Then run:

```bash
python data_preparation/prepare_dataset.py <style_name>
```

This will automatically create:

- `proxy_svg/`
- `proxy_svg2png/`
- `backbone/`
- `A-VAT_train_Data/`
- `S-VAT_train_Data/`


## Training

Edit `STYLE_ORDER` in [train_AiS.sh](/mnt/d/hyf_workspace/Abstraction_in_Style/train_AiS.sh), then run:

```bash
bash train_AiS.sh
```

By default, the script trains both stages and saves weights to:

- `dataset/<style_name>/A-VAT_checkpoint/`
- `dataset/<style_name>/S-VAT_checkpoint/`

## Inference

Place test images in:

```text
test_assets/input_images/
```

Run both stages:

```bash
python test_AiS.py --style <style_name> --stage all
```

Or run a single stage:

```bash
python test_AiS.py --style <style_name> --stage A-VAT
python test_AiS.py --style <style_name> --stage S-VAT
```

Outputs are saved to:

```text
test_assets/generated_images/
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{lu2026abstraction,
  title={Abstraction in Style},
  author={Min Lu and Yuanfeng He and Anthony Chen and Jianhuang He and Pu Wang and Daniel Cohen-Or and Hui Huang},
  journal={arXiv preprint arXiv:2603.29924},
  year={2026}
}
```

Paper page: [https://arxiv.org/abs/2603.29924](https://arxiv.org/abs/2603.29924)

## License

This repository is released under the license in [LICENSE](/mnt/d/hyf_workspace/Abstraction_in_Style/LICENSE).