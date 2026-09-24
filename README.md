# CVAE-GAN PyTorch reproduction

A modern PyTorch implementation of **CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training** by Bao et al. (ICCV 2017).

The repository implements the paper's four-network design:

- conditional encoder `E` producing a Gaussian latent code;
- conditional generator `G` producing 128 x 128 RGB images;
- discriminator `D` for real/fake supervision and feature matching;
- classifier `C` for category supervision and class-specific feature matching.

The training objective combines KL regularization, pixel and pairwise feature reconstruction, discriminator mean matching, and class-conditional classifier feature matching. The default coefficients follow Algorithm 1 in the paper: `lambda_kl=3`, `lambda_recon=1`, and both mean-matching coefficients equal to `1e-3`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset layout

The loader accepts any `torchvision.datasets.ImageFolder` dataset:

```text
data/my_dataset/
  class_000/
    image_001.jpg
    image_002.jpg
  class_001/
    image_003.jpg
```

For paper-aligned experiments:

- **FaceScrub:** detect and align faces from five landmarks, then crop a 128 x 128 region around the nose.
- **Oxford 102 Flowers:** crop with the supplied segmentation mask, then resize to 128 x 128.
- **CUB-200:** use the original image and the class identity. The loader resizes and center-crops it to 128 x 128.

Dataset licenses and download terms remain with their respective publishers. The repository does not redistribute the datasets.

## Training

```bash
python train.py \
  --data data/my_dataset \
  --output runs/flowers \
  --epochs 100 \
  --batch-size 32 \
  --amp
```

Use `--max-steps 1` for a quick end-to-end smoke test before starting a long run.

Checkpoints, sample grids, and TensorBoard events are written below the selected output directory.

```bash
tensorboard --logdir runs/flowers/tensorboard
```

## Sampling

```bash
python sample.py \
  --checkpoint runs/flowers/checkpoints/epoch-0100.pt \
  --output runs/flowers/final-grid.png \
  --per-class 8
```

## Tests

```bash
pip install pytest
pytest -q
```

The tests exercise model shapes, backpropagation, and the feature-matching losses on synthetic tensors. Full paper reproduction requires the datasets and a CUDA-capable GPU.

## Reproduction status

The architecture and training objectives are implemented. On September 24, 2026, the test suite passed (`2 passed`) with PyTorch 2.14.0+cpu and torchvision 0.29.0+cpu. A one-step, two-class CPU smoke run completed the full training loop, wrote a checkpoint, and generated a sample grid from that checkpoint.

No paper-scale training run or claim of matching the published metrics is included yet. The paper reports 97.78% generated-face top-1 classification accuracy and a realism score of 19.03 for CVAE-GAN; those numbers are reference values, not results produced by this repository.

### Expected full-run targets

These targets come from the original paper and must not be interpreted as measurements from this repository:

| Metric | Expected target |
|---|---:|
| Generated-face top-1 classification accuracy | approximately 97.78% |
| Realism score used by the paper | approximately 19.03 |
| LFW accuracy with 5k synthetic identities | approximately 92.98% |

The verified local result is limited to successful tests plus an end-to-end smoke run. Actual full-dataset metrics remain pending.

To repeat the verified smoke path after preparing any two-class ImageFolder dataset:

```bash
python train.py --data data/smoke --output runs/smoke --epochs 1 \
  --batch-size 2 --workers 0 --latent-dim 32 --max-steps 1
python sample.py --checkpoint runs/smoke/checkpoints/epoch-0001.pt \
  --output runs/smoke/generated-grid.png --per-class 2
```


## Presentation

The five-slide LaTeX Beamer summary is in [`latex/presentation.tex`](latex/presentation.tex). A compiled PDF is included at [`latex/presentation.pdf`](latex/presentation.pdf).

## Reference

Jianmin Bao, Dong Chen, Fang Wen, Houqiang Li, and Gang Hua. "CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training." Proceedings of ICCV, 2017. DOI: [10.1109/ICCV.2017.299](https://doi.org/10.1109/ICCV.2017.299).

```bibtex
@inproceedings{bao2017cvaegan,
  title={CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training},
  author={Bao, Jianmin and Chen, Dong and Wen, Fang and Li, Houqiang and Hua, Gang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2017}
}
```
