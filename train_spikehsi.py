"""
Usage:  python train_spikehsi.py PU        (or HU / WHLK)   [--cupy] [--tta] [--test_only]
Put next to preprocess.py (your repo). Uses the same fixed TR/TE splits as MS2GCAN.
Model selection uses TRAIN loss only (no test-set peeking).
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from preprocess import loadData, get_location, patch_data, set_seed, PatchDataset
from SpikeHSINet import SpikeHSINet, dihedral

CFG = {
    'PU':        dict(l=19, hidden=64, bs=64),
    'PU_normal': dict(l=19, hidden=64, bs=64),   # <-- add this line
    'HU':        dict(l=13, hidden=96, bs=32),
    'WHLK':      dict(l=15, hidden=64, bs=32),
}
ds = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'PU'
USE_CUPY, TTA, TEST_ONLY = '--cupy' in sys.argv, '--tta' in sys.argv, '--test_only' in sys.argv
cfg = CFG[ds]
T, DEPTH, EPOCHS, LR, WD, WARM, LS, SEED = 4, 2, 200, 1e-3, 5e-2, 10, 0.1, 0

device = torch.device('cuda:0')
set_seed(SEED)
torch.backends.cudnn.benchmark = True

# ---------- data ----------
data, labels_TE, labels_TR, class_num = loadData(ds)
H, W, S = data.shape
data = preprocessing.StandardScaler().fit_transform(data.reshape(H * W, -1)).reshape(H, W, -1)
tr_loc, tr_lab = get_location(labels_TR)
te_loc, te_lab = get_location(labels_TE)
l = cfg['l']
train_x = torch.from_numpy(patch_data(data, l, tr_loc)).to(device)            # small -> keep on GPU
train_y = torch.from_numpy(tr_lab.astype(np.int64)).to(device)
test_loader = DataLoader(PatchDataset(data, te_loc, te_lab, l), batch_size=cfg['bs'] * 8,
                         shuffle=False, num_workers=4)

model = SpikeHSINet(T=T, img_size=l, num_cls=class_num, input_dim=S, hidden=cfg['hidden'],
                    depth=DEPTH, use_cupy=USE_CUPY).to(device)
ckpt = f'./best_SpikeHSINet_{ds}_weights.pth'
print('params (M):', sum(p.numel() for p in model.parameters()) / 1e6)


# ---------- train ----------
def augment(x):
    x = dihedral(x, l, np.random.randint(4), bool(np.random.randint(2)))
    return x + 0.02 * torch.randn_like(x)             # light spectral noise


if not TEST_ONLY:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / WARM if e < WARM else
        0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (e - WARM) / (EPOCHS - WARM))))
    best, n, bs = 1e9, len(train_x), cfg['bs']
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        perm = torch.randperm(n, device=device)
        tot, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:                          # BatchNorm needs >1 sample
                continue
            xb, yb = augment(train_x[idx]), train_y[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=LS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        sched.step()
        tot /= n
        print(f'Ep {ep+1}/{EPOCHS} loss {tot:.4f} trainAcc {correct/n:.4f} {time.time()-t0:.1f}s')
        if tot < best:
            best = tot
            torch.save(model.state_dict(), ckpt)


# ---------- test ----------
model.load_state_dict(torch.load(ckpt))
model.eval()
tfs = [(k, f) for k in range(4) for f in (False, True)] if TTA else [(0, False)]
preds, trues = [], []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc='Testing'):
        xb = xb.to(device)
        logits = sum(F.softmax(model(dihedral(xb, l, k, f)), 1) for k, f in tfs)
        preds.append(logits.argmax(1).cpu().numpy()); trues.append(yb.numpy().astype(int))
y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
cm = confusion_matrix(y_true, y_pred)
acc_class = np.diag(cm) / cm.sum(1)
print(f'\n{ds}  OA {np.trace(cm)/cm.sum():.4f}  AA {acc_class.mean():.4f}  '
      f'Kappa {cohen_kappa_score(y_true, y_pred):.4f}')
print('per-class acc:', np.round(acc_class, 4))
