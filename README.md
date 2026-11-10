
# SpikeHSINet

A fully spiking spectral-spatial network for **few-shot hyperspectral image (HSI) classification**.
It combines multi-scale spiking convolutions with spiking self-attention, and is trained with
augmentation, label smoothing and AdamW. Optional test-time augmentation (TTA) is supported.

## Model

| Stage | Description |
|---|---|
| Stem | 1x1 conv (bands to channels), BN, LIF, 3x3 conv, BN. Input is repeated for T time steps (direct encoding) |
| Spike blocks (x depth) | Multi-scale conv mix (1x1 / 3x3-DW / 5x5-DW), spiking self-attention (softmax-free spike Q, K, V), spiking MLP, all with membrane shortcuts |
| Readout | Concatenate center pixel and patch mean, average over T, LayerNorm, Linear |

Built with [SpikingJelly](https://github.com/fangwei123456/spikingjelly) (LIF neurons, ATan surrogate gradient).

## Repository structure

```text
SpikeHSINet/
├─ SpikeHSINet.py              # model
├─ hsi_data.py                 # data loading, few-shot split, patch extraction
├─ train_spikehsi.py           # train and test SpikeHSINet
├─ requirements.txt
├─ .gitignore
├─ splits/                     # auto-created: saved train/test splits (reproducibility)
└─ data/patch_data/PU/         # you add the .mat files here (not in the repo)
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/SpikeHSINet.git
cd SpikeHSINet
pip install -r requirements.txt
```

A CUDA GPU is required (the scripts use `cuda:0`). Google Colab with a T4 GPU works.

## Data

Datasets are not included. Download the public Pavia University scene (UPV/EHU
"Hyperspectral Remote Sensing Scenes") and place the files as:

```text
data/patch_data/PU/PaviaU.mat       # variable: paviaU      (610 x 340 x 103)
data/patch_data/PU/PaviaU_gt.mat    # variable: paviaU_gt   (610 x 340)
```

Indian Pines (`IP`) and Salinas (`SA`) are also supported with the standard UPV/EHU files in
`data/patch_data/IP/` and `data/patch_data/SA/`.

## Usage

Train (then quick test):

```bash
python train_spikehsi.py PU
```

Test with the saved best weights, using test-time augmentation:

```bash
python train_spikehsi.py PU --test_only --tta
```

| Option | Default | Meaning |
|---|---|---|
| `dataset` | `PU` | `PU`, `IP` or `SA` |
| `--per_class` | 10 | Training pixels per class. All other labeled pixels are the test set |
| `--seed` | 0 | Seed for the split and training |
| `--epochs` | 200 | Training epochs |
| `--T` | 4 | Spiking time steps |
| `--depth` | 2 | Number of spike blocks |
| `--hidden` | 64 | Channel width (divisible by 4) |
| `--patch` | 19 (PU) | Odd patch size |
| `--bs` | 64 | Batch size |
| `--tta` | off | 8-view test-time augmentation |
| `--test_only` | off | Skip training, load saved weights |

The best weights are chosen by **training loss only** (no test-set selection) and saved as
`best_SpikeHSINet_<dataset>_pc<per_class>_s<seed>.pth`. Testing must use the same
`--per_class` and `--seed` as training.

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/SpikeHSINet
!pip install -q spikingjelly
!python train_spikehsi.py PU
!python train_spikehsi.py PU --test_only --tta
```

## Results

Pavia University, 10 training pixels per class, all remaining labeled pixels for testing.
Fill in from your own runs (report mean ± std over several seeds).

| Model | TTA | OA (%) | AA (%) | Kappa |
|---|---|---|---|---|
| SpikeHSINet | No | | | |
| SpikeHSINet | Yes | | | |

## Reproducibility

- Python 3, PyTorch, SpikingJelly; GPU with CUDA
- Split is stored in `splits/PU_pc10_seed0.npz` and reused on later runs
- Seeds are set for NumPy and PyTorch; small run-to-run variation on GPU is normal

## Acknowledgements

Built with [SpikingJelly](https://github.com/fangwei123456/spikingjelly). Pavia University data
courtesy of Prof. Paolo Gamba, distributed by the UPV/EHU Computational Intelligence Group.

## License

MIT (see `LICENSE`).
