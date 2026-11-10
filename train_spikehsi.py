"""
SpikeHSINet - train / test on the REAL Pavia University data (no author files needed).

  python train_spikehsi.py PU                 # train, then quick test
  python train_spikehsi.py PU --test_only --tta
Options: --per_class 10  --seed 0  --epochs 200  --T 4  --depth 2  --hidden 64  --patch 19  --bs 64  --cupy
Needs in the same folder: hsi_data.py, SpikeHSINet.py  (preprocess.py is NOT needed)
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tqdm import tqdm

from hsi_data import load_dataset, get_split, PatchDataset, all_patches
from SpikeHSINet import SpikeHSINet, dihedral

ap = argparse.ArgumentParser()
ap.add_argument('dataset', nargs='?', default='PU', help='PU / IP / SA')
ap.add_argument('--per_class', type=int, default=10, help='training pixels per class')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--T', type=int, default=4)
ap.add_argument('--depth', type=int, default=2)
ap.add_argument('--hidden', type=int, default=64)
ap.add_argument('--patch', type=int, default=0, help='odd patch size (0 = dataset default)')
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--tta', action='store_true')
ap.add_argument('--test_only', action='store_true')
ap.add_argument('--cupy', action='store_true')
args = ap.parse_args()

l = args.patch or {'PU': 19, 'IP': 15, 'SA': 15}[args.dataset]
LR, WD, WARM, LS = 1e-3, 5e-2, 10, 0.1
device = torch.device('cuda:0')
np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

# ---------- data ----------
img, gt, class_num = load_dataset(args.dataset)
tr_loc, tr_lab, te_loc, te_lab = get_split(gt, class_num, args.dataset, args.per_class, args.seed)
train_x, train_y = all_patches(PatchDataset(img, tr_loc, tr_lab, l))
train_x, train_y = train_x.to(device), train_y.to(device)
test_loader = DataLoader(PatchDataset(img, te_loc, te_lab, l), batch_size=256, shuffle=False, num_workers=2)

model = SpikeHSINet(T=args.T, img_size=l, num_cls=class_num, input_dim=img.shape[-1],
                    hidden=args.hidden, depth=args.depth, use_cupy=args.cupy).to(device)
ckpt = f'./best_SpikeHSINet_{args.dataset}_pc{args.per_class}_s{args.seed}.pth'
print('params (M):', sum(p.numel() for p in model.parameters()) / 1e6)


def augment(x):
    x = dihedral(x, l, np.random.randint(4), bool(np.random.randint(2)))
    return x + 0.02 * torch.randn_like(x)


# ---------- train ----------
if not args.test_only:
    EPOCHS, bs, n = args.epochs, args.bs, len(train_x)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / WARM if e < WARM else
        0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (e - WARM) / max(1, EPOCHS - WARM))))
    best = 1e9
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        perm = torch.randperm(n, device=device)
        tot, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:                       # BatchNorm needs > 1 sample
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
        if tot < best:                             # selected on TRAIN loss only (no test peeking)
            best = tot
            torch.save(model.state_dict(), ckpt)

# ---------- test ----------
try:
    model.load_state_dict(torch.load(ckpt))
except FileNotFoundError:
    raise SystemExit(f'No weights at {ckpt}. Train first (run without --test_only) with the same --per_class/--seed.')
model.eval()
tfs = [(k, f) for k in range(4) for f in (False, True)] if args.tta else [(0, False)]
preds, trues = [], []
with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc='Testing'):
        xb = xb.to(device)
        p = sum(F.softmax(model(dihedral(xb, l, k, f)), 1) for k, f in tfs)
        preds.append(p.argmax(1).cpu().numpy()); trues.append(yb.numpy())
y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
cm = confusion_matrix(y_true, y_pred)
acc_class = np.diag(cm) / cm.sum(1)
print(f'\nSpikeHSINet [{args.dataset}, {args.per_class}/class{", TTA" if args.tta else ""}]  '
      f'OA {np.trace(cm)/cm.sum():.4f}  AA {acc_class.mean():.4f}  Kappa {cohen_kappa_score(y_true, y_pred):.4f}')
print('per-class acc:', np.round(acc_class, 4))
