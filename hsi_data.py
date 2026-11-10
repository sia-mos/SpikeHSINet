"""
hsi_data.py - self-contained data pipeline (no dependency on the MS2GCAN author's preprocess.py).

Uses the REAL public .mat files you downloaded. Expected layout (same as before):
    data/patch_data/PU/PaviaU.mat      (variable 'paviaU',    610x340x103)
    data/patch_data/PU/PaviaU_gt.mat   (variable 'paviaU_gt', 610x340)
The train/test split is made from the full ground truth: N random pixels per class for training,
ALL remaining labeled pixels for testing. It is saved to splits/*.npz so it is reproducible.
"""
import os
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# name: (image file, image key, gt file, gt key, number of classes)
DATASETS = {
    'PU': ('data/patch_data/PU/PaviaU.mat', 'paviaU',
           'data/patch_data/PU/PaviaU_gt.mat', 'paviaU_gt', 9),
    'IP': ('data/patch_data/IP/Indian_pines_corrected.mat', 'indian_pines_corrected',
           'data/patch_data/IP/Indian_pines_gt.mat', 'indian_pines_gt', 16),
    'SA': ('data/patch_data/SA/Salinas_corrected.mat', 'salinas_corrected',
           'data/patch_data/SA/Salinas_gt.mat', 'salinas_gt', 16),
}


def _load_mat(path, key):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing file: {path}')
    if os.path.getsize(path) == 0:
        raise ValueError(f'{path} is empty (failed download). Delete it and upload the real file.')
    m = loadmat(path)
    if key not in m:
        keys = [k for k in m if not k.startswith('__')]
        raise KeyError(f"{path} has no variable '{key}'. Variables found: {keys}")
    return m[key]


def load_dataset(name, root='.'):
    """Returns standardized image [H,W,S] (float32), ground truth [H,W] (0 = unlabeled), num_classes."""
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name}. Choose from {list(DATASETS)}')
    f_img, k_img, f_gt, k_gt, ncls = DATASETS[name]
    img = _load_mat(os.path.join(root, f_img), k_img).astype(np.float32)
    gt = _load_mat(os.path.join(root, f_gt), k_gt).astype(np.int64)
    assert img.shape[:2] == gt.shape, f'image {img.shape} and ground truth {gt.shape} do not match'
    H, W, S = img.shape
    counts = [int((gt == c).sum()) for c in range(1, ncls + 1)]
    print(f'[{name}] image {img.shape} | labeled pixels {sum(counts)} | per class {counts}')
    img = StandardScaler().fit_transform(img.reshape(-1, S)).reshape(H, W, S).astype(np.float32)
    return img, gt, ncls


def get_split(gt, ncls, name, per_class=10, seed=0, root='.'):
    """Fixed random split: per_class training pixels per class, everything else labeled = test."""
    path = os.path.join(root, 'splits', f'{name}_pc{per_class}_seed{seed}.npz')
    if os.path.exists(path):
        z = np.load(path)
        print(f'[split] loaded {path}')
        return z['tr_loc'], z['tr_lab'], z['te_loc'], z['te_lab']
    rng = np.random.RandomState(seed)
    tr_loc, tr_lab, te_loc, te_lab = [], [], [], []
    for c in range(1, ncls + 1):
        idx = np.argwhere(gt == c)
        k = min(per_class, len(idx) // 2)
        perm = rng.permutation(len(idx))
        tr, te = idx[perm[:k]], idx[perm[k:]]
        tr_loc.append(tr); tr_lab += [c - 1] * len(tr)
        te_loc.append(te); te_lab += [c - 1] * len(te)
    tr_loc, te_loc = np.concatenate(tr_loc), np.concatenate(te_loc)
    tr_lab, te_lab = np.array(tr_lab), np.array(te_lab)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, tr_loc=tr_loc, tr_lab=tr_lab, te_loc=te_loc, te_lab=te_lab)
    print(f'[split] created {path} | train {len(tr_loc)} | test {len(te_loc)}')
    return tr_loc, tr_lab, te_loc, te_lab


class PatchDataset(Dataset):
    """Cuts an l x l patch around each labeled pixel on demand. Item: ([l*l, S] float32, label int64)."""
    def __init__(self, img, locs, labels, l):
        assert l % 2 == 1, 'patch size must be odd'
        self.h, self.l = l // 2, l
        self.pad = np.pad(img, ((self.h, self.h), (self.h, self.h), (0, 0)), mode='reflect')
        self.locs, self.labels = locs, labels

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, k):
        i, j = self.locs[k]
        p = self.pad[i:i + self.l, j:j + self.l, :]
        return torch.from_numpy(np.ascontiguousarray(p.reshape(-1, p.shape[-1]))), int(self.labels[k])


def all_patches(ds):
    """Stack a (small) PatchDataset into tensors, e.g. the training set."""
    xs, ys = zip(*[ds[k] for k in range(len(ds))])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
