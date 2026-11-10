

# SpikeHSINet

A fully spiking spectral-spatial network for hyperspectral image classification.
Multi-scale spiking convolution + spiking self-attention, trained with augmentation,
label smoothing and AdamW. Test-time augmentation (TTA) is optional.

## Setup
```bash
pip install -r requirements.txt
```
Download Pavia University (`PaviaU.mat`, `PaviaU_gt.mat`) from the UPV/EHU hyperspectral
scenes page and place them in `data/patch_data/PU/`.

This repo also needs `preprocess.py` (and `MS2GCAN.py` for the baseline) from
https://github.com/fwx0314/MS2GCAN. Copy them into this folder.

## Train and test
```bash
python train_spikehsi.py PU_normal              # train, then test
python train_spikehsi.py PU_normal --test_only --tta
```
The included `PU_normal_TR.npy` / `PU_normal_TE.npy` fix the random split
(about 0.2% of pixels per class for training, 91 training pixels in total).

## Baseline
```bash
python train_baseline_ms2gcan.py                # original MS2GCAN recipe
python train_baseline_ms2gcan.py --newrecipe    # MS2GCAN with the new training recipe
```

## Results (Pavia University, PU_normal split, seed 0)
| Model | TTA | OA | AA | Kappa |
|---|---|---|---|---|
| SpikeHSINet | No | 95.73 | 92.26 | 0.9433 |
| SpikeHSINet | Yes | 96.03 | 92.44 | 0.9472 |

## Acknowledgements
Data pipeline adapted from MS2GCAN (https://github.com/fwx0314/MS2GCAN).
