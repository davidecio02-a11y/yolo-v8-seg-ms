# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:11:18 2026

@author: david
"""

import argparse
import copy
import gc
import itertools
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
 
# =====================================================================
# CONFIGURAZIONE
# =====================================================================
ROOT = Path(r"C:\spalling_corrosion_patches_v2_new")
WORK = Path(r"D:\yolo_seg\final_runs")
YOLO_WEIGHTS, YOLO_YAML = "yolov8s-seg.pt", "yolov8s-seg.yaml"
SEEDS = [0, 1, 2]
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
GHOST_K = 5
BCE_WEIGHT, NEG_WEIGHT = 0.4, 1.0
LR_WARMUP, LR_NEW, LR_PRETRAINED, LR_SCRATCH = 1e-3, 5e-4, 1e-4, 1e-3
EPOCHS = dict(warm=5, ft=45, scratch=120, yolo_pre=50, yolo_scratch=120)
VAL_THRS = torch.arange(0.20, 0.75, 0.05)             # soglie provate a ogni epoca (scelta del checkpoint)
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)       # soglie provate nella valutazione finale (validation)
MIN_FRACS = [0.0, 0.0025, 0.005, 0.01, 0.02]          # filtro componenti piccole (frazione dell'area della patch)
BINS = [0, 0.02, 0.10, 0.30, 1.0]
BIN_LABELS = ["<=2%", "2-10%", "10-30%", ">30%"]
N_BOOT = 2000
 
_ms = dict(kind="ms", pretrained=True, ghost=0, rep=0, simam=0, loss="iou")
CONFIGS = {
    "yolo_pre": dict(kind="yolo", pretrained=True),
    "yolo_scratch": dict(kind="yolo", pretrained=False),
    "ms_base": dict(_ms),
    "ms_ghost": dict(_ms, ghost=1),
    "ms_rep": dict(_ms, rep=1),
    "ms_simam": dict(_ms, simam=1),
    "ms_dsiou": dict(_ms, loss="dsiou"),
    "ms_full": dict(_ms, ghost=1, rep=1, simam=1, loss="dsiou"),
    "ms_base_scratch": dict(_ms, pretrained=False),
    "ms_full_scratch": dict(_ms, pretrained=False, ghost=1, rep=1, simam=1, loss="dsiou"),
}
ABLATION = ["ms_ghost", "ms_rep", "ms_simam", "ms_dsiou", "ms_full"]          # ciascuno confrontato con ms_base
PAIRS = [("ms_full", "yolo_pre"), ("ms_base", "yolo_pre"), ("ms_full", "ms_base"),
         ("ms_full_scratch", "yolo_scratch"), ("ms_base_scratch", "yolo_scratch"),
         ("ms_full_scratch", "ms_base_scratch"), ("ms_full", "ms_full_scratch"),
         ("ms_base", "ms_base_scratch"), ("yolo_pre", "yolo_scratch")]
QUICK_CONFIGS = ["yolo_pre", "ms_base", "ms_full"]
ARCH_OF = {"yolo_scratch": "yolo_pre", "ms_dsiou": "ms_base", "ms_base_scratch": "ms_base",
           "ms_full_scratch": "ms_full"}                                          # stessa architettura = stesso costo
EFF_CONFIGS = ["yolo_pre", "ms_base", "ms_ghost", "ms_rep", "ms_simam", "ms_full"]
PRESETS = {
    "confirm": dict(configs=["yolo_pre", "ms_base", "ms_ghost", "ms_full"], seeds=[1, 2]),   # conferma il confronto principale
    "pretrain": dict(configs=["yolo_scratch", "ms_base_scratch"], seeds=[1, 2]),             # effetto del transfer learning
    "pretrain1": dict(configs=["yolo_scratch", "ms_base_scratch"], seeds=[1]),               # versione ridotta
    "ablation": dict(configs=["ms_rep", "ms_simam", "ms_dsiou"], seeds=[1, 2]),              # moduli senza effetto: bassa priorita'
    "rest": dict(configs=list(CONFIGS), seeds=[1, 2]),                                       # tutto cio' che manca
}
DEFAULT_MIN = {"pre": 6.0, "scratch": 13.0}        # stime iniziali (minuti per run, valutazione inclusa)
CUDNN_BENCHMARK = True                              # ingressi a dimensione fissa: scelta piu' veloce degli algoritmi
YOLO_CACHE = "disk"                                  # immagini in RAM: stessi dati e stesso protocollo, meno lettura da disco
 
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
# =====================================================================
# 1. MODELLO MS CON INTERRUTTORI (ghost / rep / simam)
# =====================================================================
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
 
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
 
    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        return x * torch.sigmoid(d / (4 * (v + self.e_lambda)) + 0.5)
 
 
class GhostConv(nn.Module):
    """Primaria (k, s) su meta' canali + cheap-op depthwise (GHOST_K) sull'altra meta'."""
 
    def __init__(self, c1, c2, k=1, s=1, dw_k=None):
        super().__init__()
        assert c2 % 2 == 0, "GhostConv richiede canali di uscita pari"
        c_half = c2 // 2
        self.cv1 = Conv(c1, c_half, k, s)
        self.cv2 = Conv(c_half, c_half, dw_k or GHOST_K, 1, g=c_half)
 
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)
 
 
class GhostDown(GhostConv):
    def __init__(self, c1, c2):
        super().__init__(c1, c2, k=3, s=2)
 
 
class GhostBottleneck(nn.Module):
    """Sostituisce Bottleneck di YOLOv8 dentro C2f -> GC-C2f."""
 
    def __init__(self, c, shortcut):
        super().__init__()
        self.cv = nn.Sequential(GhostConv(c, c), GhostConv(c, c))
        self.add = shortcut
        last = self.cv[1]
        nn.init.zeros_(last.cv1.bn.weight)
        nn.init.zeros_(last.cv2.bn.weight)
 
    def forward(self, x):
        y = self.cv(x)
        return x + y if self.add else y
 
 
def _fuse_conv_bn_pair(w, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return w * t, bn.bias - bn.running_mean * bn.weight / std
 
 
class RepBlock(nn.Module):
    """Training: 3x3+BN, 1x1+BN, (identity+BN). Inferenza: singola conv 3x3."""
 
    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.deploy = False
        self.conv3 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.bn_id = nn.BatchNorm2d(c1) if c1 == c2 else None
        self.act = nn.SiLU()
 
    def forward(self, x):
        if self.deploy:
            return self.act(self.fused(x))
        out = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
        if self.bn_id is not None:
            out = out + self.bn_id(x)
        return self.act(out)
 
    @torch.no_grad()
    def reparameterize(self):
        if self.deploy:
            return
        w3, b3 = _fuse_conv_bn_pair(self.conv3.weight, self.bn3)
        w1, b1 = _fuse_conv_bn_pair(F.pad(self.conv1.weight, [1, 1, 1, 1]), self.bn1)
        w, b = w3 + w1, b3 + b1
        if self.bn_id is not None:
            ident = torch.zeros_like(w3)
            for i in range(self.c1):
                ident[i, i, 1, 1] = 1.0
            wi, bi = _fuse_conv_bn_pair(ident, self.bn_id)
            w, b = w + wi, b + bi
        fused = nn.Conv2d(self.c1, self.c2, 3, 1, 1, bias=True).to(w.device)
        fused.weight.copy_(w)
        fused.bias.copy_(b)
        self.fused = fused
        del self.conv3, self.bn3, self.conv1, self.bn1
        if self.bn_id is not None:
            del self.bn_id
        self.deploy = True
 
 
def _to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
 
 
class SemHead(nn.Module):
    """Proto pretrained + P4 + P5 + skip s4 -> fusione -> (SimAM) -> up + skip s2 -> logit.
    rep=True: blocchi RepBlock; rep=False: Conv 3x3 + BN + SiLU. simam=False: nessuna attenzione."""
 
    def __init__(self, nm, c_s4, c_s2, c4, c5, mid=32, rep=True, simam=True):
        super().__init__()
        blk = (lambda a, b: RepBlock(a, b)) if rep else (lambda a, b: Conv(a, b, 3))
        self.b4 = blk(c4, mid)
        self.b5 = blk(c5, mid)
        self.skip4 = Conv(c_s4, mid, 1)
        self.fuse4 = blk(nm + 3 * mid, mid)
        self.simam = SimAM() if simam else nn.Identity()
        self.up = nn.ConvTranspose2d(mid, mid, 2, 2, bias=False)
        self.bn_up = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()
        self.skip2 = Conv(c_s2, mid, 1)
        self.fuse2 = blk(2 * mid, mid)
        self.out = nn.Conv2d(mid, 1, 1)
        nn.init.constant_(self.out.bias, -2.0)
 
    def forward(self, proto, p4, p5, f_s4, f_s2):
        x = torch.cat([proto, _to(self.b4(p4), proto), _to(self.b5(p5), proto),
                       _to(self.skip4(f_s4), proto)], 1)
        x = self.simam(self.fuse4(x))
        x = self.act(self.bn_up(self.up(x)))
        x = self.fuse2(torch.cat([x, _to(self.skip2(f_s2), x)], 1))
        return self.out(x)
 
 
def ghostify(layers, sppf_idx):
    """Sostituisce in-place Bottleneck dei C2f con GhostBottleneck e i Conv stride 2 del neck con GhostDown."""
    for i, m in enumerate(layers):
        name = type(m).__name__
        if name == "C2f":
            shortcut = bool(getattr(m.m[0], "add", False))
            m.m = nn.ModuleList(GhostBottleneck(m.c, shortcut) for _ in range(len(m.m)))
        elif name == "Conv" and i > sppf_idx and m.conv.stride == (2, 2):
            g = GhostDown(m.conv.in_channels, m.conv.out_channels)
            ch = m.conv.out_channels // 2          # slicing dei pesi YOLO nella parte primaria
            with torch.no_grad():
                g.cv1.conv.weight.copy_(m.conv.weight[:ch])
                for n in ("weight", "bias", "running_mean", "running_var"):
                    getattr(g.cv1.bn, n).copy_(getattr(m.bn, n)[:ch])
            g.f, g.i = m.f, m.i
            layers[i] = g
 
 
class MSYOLOv8SemSeg(nn.Module):
    def __init__(self, weights=YOLO_WEIGHTS, ghost=True, rep=True, simam=True):
        super().__init__()
        from ultralytics import YOLO
        det = YOLO(weights).model.float()
        layers = list(det.model)
        head = layers[-1]
        assert hasattr(head, "proto"), "Il checkpoint deve essere *-seg"
        sppf_idx = next(i for i, m in enumerate(layers) if type(m).__name__ == "SPPF")
        body = layers[:-1]
        if ghost:
            ghostify(body, sppf_idx)
        self.layers = nn.ModuleList(body)
        self.proto = head.proto
        self.p3, self.p4, self.p5 = head.f
        with torch.no_grad():
            self.eval()
            y = self._run(torch.zeros(1, 3, 256, 256))
            nm = self.proto(y[self.p3]).shape[1]
            c_s2, c_s4 = y[0].shape[1], y[2].shape[1]
            c4, c5 = y[self.p4].shape[1], y[self.p5].shape[1]
        self.head = SemHead(nm, c_s4, c_s2, c4, c5, rep=rep, simam=simam)
        for p in self.parameters():
            p.requires_grad_(True)
        new_mods = [m for m in self.layers.modules() if isinstance(m, GhostBottleneck)]
        new_mods += [m.cv2 for m in self.layers.modules() if isinstance(m, GhostDown)]
        new_mods.append(self.head)
        new_ids = {id(p) for m in new_mods for p in m.parameters()}
        self._new = [p for p in self.parameters() if id(p) in new_ids]
        self._pre = [p for p in self.parameters() if id(p) not in new_ids]
 
    def _run(self, x):
        y = []
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return y
 
    def forward(self, x):
        y = self._run(x)
        logits = self.head(self.proto(y[self.p3]), y[self.p4], y[self.p5], y[2], y[0])
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits.squeeze(1)
 
    def pretrained_params(self):
        return self._pre
 
    def new_params(self):
        return self._new
 
    def reparameterize(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.reparameterize()
        return self
 
 
def reset_all(model):
    for m in model.modules():
        if m is not model and hasattr(m, "reset_parameters"):
            m.reset_parameters()
 
 
def set_frozen(model, frozen):
    for p in model.pretrained_params():
        p.requires_grad_(not frozen)
 
 
def set_train_mode(model, frozen):
    model.train()
    if frozen:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                m.eval()
 
 
@torch.no_grad()
def fuse_conv_bn(model):
    """Fonde BatchNorm in Conv (Conv di Ultralytics e Conv del modello MS): forward = act(bn(conv(x)))."""
    n = 0
    for m in list(model.modules()):
        conv, bn = getattr(m, "conv", None), getattr(m, "bn", None)
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            scale = bn.weight / (bn.running_var + bn.eps).sqrt()
            fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, conv.groups, bias=True).to(conv.weight.device)
            fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
            b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            fused.bias.copy_((b - bn.running_mean) * scale + bn.bias)
            m.conv, m.bn = fused, nn.Identity()
            n += 1
    return n
 
 
# =====================================================================
# 2. DATASET, LOSS, ALLENAMENTO DEL MODELLO MS
# =====================================================================
def load_gt_mask(label_dir, stem, h, w):
    gt = np.zeros((h, w), np.uint8)
    txt, png = label_dir / f"{stem}.txt", label_dir / f"{stem}.png"
    if txt.exists():
        for line in txt.read_text().splitlines():
            p = list(map(float, line.split()))
            if len(p) > 5:
                c = np.array(p[1:]).reshape(-1, 2)
                c[:, 0] *= w
                c[:, 1] *= h
                cv2.fillPoly(gt, [np.round(c).astype(np.int32)], 1)
    elif png.exists():
        m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        gt = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return gt
 
 
class SpallingDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=256, alpha=1.0, tau=0.5, is_train=True, edt=True, cache=True):
        self.images_dir, self.labels_dir = Path(images_dir), Path(labels_dir)
        self.img_size, self.alpha, self.tau, self.is_train, self.edt = img_size, alpha, tau, is_train, edt
        self.files = sorted(p.name for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp") for p in self.images_dir.glob(ext))
        self._imgs = self._masks = None
        if cache:                      # legge dal disco una sola volta: utile se il dataset sta su un'unita' lenta
            self._imgs, self._masks = [], []
            for name in self.files:
                img, mask = self._read(name)
                self._imgs.append(img)
                self._masks.append(mask)
 
    def __len__(self):
        return len(self.files)
 
    def _edt_weight_map(self, mask):
        h, w = mask.shape
        small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST) > 0.5
        if not small.any() or small.all():
            return np.ones((h, w), np.float32)
        dist = distance_transform_edt(small) + distance_transform_edt(~small)     # distanza dal bordo
        ws = 1.0 + self.alpha * np.exp(-self.tau * dist)
        return cv2.resize(ws.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
 
    def _read(self, name):
        img = cv2.imread(str(self.images_dir / name))
        if img is None:
            raise IOError(f"Impossibile leggere {name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        # GT: poligoni normalizzati delle label ridisegnati direttamente alla dimensione di training
        mask = load_gt_mask(self.labels_dir, Path(name).stem, self.img_size, self.img_size)
        return img, mask
 
    def _arrays(self, idx):
        if self._imgs is not None:
            return self._imgs[idx], self._masks[idx]
        return self._read(self.files[idx])
 
    def __getitem__(self, idx):
        img, mask = self._arrays(idx)
        mask = mask.astype(np.float32)            # copia: la cache non viene mai modificata dalle augmentation
        if self.is_train:
            if random.random() > 0.5:
                img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
            if random.random() > 0.5:
                img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
            if random.random() > 0.5:
                a, b = random.uniform(0.8, 1.2), random.randint(-15, 15)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        w = self._edt_weight_map(mask) if self.edt else np.ones_like(mask, np.float32)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img_t, torch.from_numpy(mask).float(), torch.from_numpy(w).float()
 
 
class SegLoss(nn.Module):
    """BCE + soft-IoU. boundary=True: soft-IoU e BCE pesati sulla distanza dal bordo (DSIoU).
    Sulle immagini senza difetto il soft-IoU e' inerte: si aggiunge la probabilita' media predetta (neg_weight)."""
 
    def __init__(self, bce_weight=0.4, neg_weight=1.0, boundary=True, eps=1e-6):
        super().__init__()
        self.bce_weight, self.neg_weight, self.boundary, self.eps = bce_weight, neg_weight, boundary, eps
 
    def forward(self, logits, target, weight):
        logits = logits.float()
        if not self.boundary:
            weight = torch.ones_like(weight)
        probs = torch.sigmoid(logits)
        inter = (weight * probs * target).sum(dim=[1, 2])
        union = (weight * (probs + target - probs * target)).sum(dim=[1, 2])
        iou_loss = 1.0 - (inter + self.eps) / (union + self.eps)
        has_obj = target.flatten(1).sum(dim=1) > 0
        zero = logits.new_zeros(())
        pos_term = iou_loss[has_obj].mean() if has_obj.any() else zero
        neg_term = (weight * probs).mean(dim=[1, 2])[~has_obj].mean() if (~has_obj).any() else zero
        loss = pos_term + self.neg_weight * neg_term
        bce = (weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * loss
 
 
@torch.no_grad()
def validate(model, loader, criterion, device, thrs):
    model.eval()
    thrs = thrs.to(device)
    tp = torch.zeros(len(thrs), device=device)
    fp, fn = torch.zeros_like(tp), torch.zeros_like(tp)
    loss_sum, n = 0.0, 0
    use_amp = device.type == "cuda"
    for images, gt, w in loader:
        images, gt, w = images.to(device), gt.to(device), w.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        loss_sum += criterion(logits, gt, w).item()
        n += 1
        pred = torch.sigmoid(logits.float()).unsqueeze(0) >= thrs.view(-1, 1, 1, 1)
        g = (gt > 0.5).unsqueeze(0)
        tp += (pred & g).sum(dim=(1, 2, 3))
        fp += (pred & ~g).sum(dim=(1, 2, 3))
        fn += (~pred & g).sum(dim=(1, 2, 3))
    e = 1e-7
    iou = tp / (tp + fp + fn + e)
    b = int(iou.argmax())
    return dict(loss=loss_sum / max(n, 1), thr=float(thrs[b]), iou=float(iou[b]))
 
 
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, frozen):
    set_train_mode(model, frozen)
    use_amp = device.type == "cuda"
    total = 0.0
    for images, gt, w in loader:
        images, gt, w = images.to(device), gt.to(device), w.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        loss = criterion(logits, gt, w)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)
 
 
def ms_epochs(pretrained, quick):
    if pretrained:
        return (1, 3) if quick else (EPOCHS["warm"], EPOCHS["ft"])
    return (0, 4) if quick else (0, EPOCHS["scratch"])
 
 
def train_ms(name, cfg, seed, ckpt_path, quick):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    set_seed(seed)
    va = "valid" if (ROOT / "valid").exists() else "val"
    edt = cfg["loss"] == "dsiou"
    train_ds = SpallingDataset(ROOT / "train" / "images", ROOT / "train" / "labels", IMG_SIZE, is_train=True, edt=edt)
    val_ds = SpallingDataset(ROOT / va / "images", ROOT / va / "labels", IMG_SIZE, is_train=False, edt=edt)
    kw = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=dev.type == "cuda")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, **kw)
    arch = dict(ghost=bool(cfg["ghost"]), rep=bool(cfg["rep"]), simam=bool(cfg["simam"]))
    model = MSYOLOv8SemSeg(YOLO_WEIGHTS, **arch).to(dev)
    pretrained = cfg["pretrained"]
    if not pretrained:
        reset_all(model)
    n_pre = sum(p.numel() for p in model.pretrained_params())
    n_new = sum(p.numel() for p in model.new_params())
    print(f"   parametri: {(n_pre + n_new) / 1e6:.2f}M (di cui con init YOLO {n_pre / 1e6:.2f}M, pretrained={pretrained})")
    criterion = SegLoss(BCE_WEIGHT, NEG_WEIGHT, boundary=edt)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    warm, ft = ms_epochs(pretrained, quick)
 
    if warm:
        set_frozen(model, True)
        opt = torch.optim.AdamW(model.new_params(), lr=LR_WARMUP, weight_decay=0.01)
        for _ in range(warm):
            train_one_epoch(model, train_loader, criterion, opt, scaler, dev, frozen=True)
        set_frozen(model, False)
    if pretrained:
        groups = [{"params": model.pretrained_params(), "lr": LR_PRETRAINED},
                  {"params": model.new_params(), "lr": LR_NEW}]
    else:
        groups = [{"params": list(model.parameters()), "lr": LR_SCRATCH}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: max(1.0 - e / ft, 0.01))
 
    tmp = Path(str(ckpt_path) + ".tmp")
    best = -1.0
    for ep in range(1, ft + 1):
        t_ep = time.time()
        tl = train_one_epoch(model, train_loader, criterion, opt, scaler, dev, frozen=False)
        if ep == 1:
            sec = time.time() - t_ep
            print(f"   (1a epoca: {sec:.0f} s -> circa {sec * ft / 60:.1f} min per {ft} epoche)", flush=True)
        sched.step()
        v = validate(model, val_loader, criterion, dev, VAL_THRS)
        improved = v["iou"] > best
        if improved:
            best = v["iou"]
            torch.save({"state_dict": model.state_dict(), "arch": arch, "cfg": cfg, "seed": seed,
                        "thr": v["thr"], "val": v}, str(tmp))
        if improved or ep % 10 == 0 or ep == ft:
            print(f"   ep {ep:03d}/{ft} | train {tl:.4f} | val {v['loss']:.4f} | IoU val {v['iou'] * 100:.2f}% "
                  f"(thr {v['thr']:.2f}){' *' if improved else ''}", flush=True)
    os.replace(str(tmp), str(ckpt_path))          # solo ora la run risulta completa
    del model, opt, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
 
def write_runtime_yaml():
    va = "valid" if (ROOT / "valid").exists() else "val"
    y = WORK / "data_runtime.yaml"
    y.write_text(f"path: {ROOT.resolve().as_posix()}\ntrain: train/images\nval: {va}/images\ntest: test/images\n\n"
                 "names:\n  0: corrosion_spalling\n", encoding="utf-8")
    return y
 
 
def train_yolo(name, cfg, seed, best_path, quick):
    from ultralytics import YOLO
    scratch = not cfg["pretrained"]
    epochs = 3 if quick else EPOCHS["yolo_scratch" if scratch else "yolo_pre"]
    model = YOLO(YOLO_YAML if scratch else YOLO_WEIGHTS)
    model.train(data=str(write_runtime_yaml()), epochs=epochs, imgsz=IMG_SIZE, batch=32, workers=NUM_WORKERS,
                device=0 if torch.cuda.is_available() else "cpu", seed=seed, pretrained=not scratch,
                project=str(WORK / "yolo_runs"), name=f"{name}_s{seed}", exist_ok=True, plots=False,
                cache=False if quick else YOLO_CACHE)
    tmp = Path(str(best_path) + ".tmp")
    shutil.copy2(Path(model.trainer.save_dir) / "weights" / "best.pt", tmp)
    os.replace(str(tmp), str(best_path))
    del model
    gc.collect()
 
 
# =====================================================================
# 3. VALUTAZIONE (protocollo identico per tutti i modelli)
# =====================================================================
def make_ms_predictor(ckpt_path):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(ckpt_path), map_location="cpu")
    model = MSYOLOv8SemSeg(YOLO_WEIGHTS, **ck["arch"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(dev).eval()
    model.reparameterize()
 
    @torch.no_grad()
    def predict(img_bgr, path=None):
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255).to(dev)
        up = F.interpolate(model(x).unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(up)[0, 0].cpu().numpy()
 
    return predict
 
 
def make_yolo_predictor(best_path):
    from ultralytics import YOLO
    model = YOLO(str(best_path))
    dev = 0 if torch.cuda.is_available() else "cpu"
 
    def predict(img_bgr, path=None):
        h, w = img_bgr.shape[:2]
        r = model.predict(img_bgr, imgsz=IMG_SIZE, conf=0.001, retina_masks=True, verbose=False, device=dev)[0]
        cm = np.zeros((h, w), np.float32)      # confidenza massima per pixel: unione delle istanze con conf >= t
        if r.masks is not None and len(r.masks):
            masks = r.masks.data.cpu().numpy() > 0.5
            for m, c in zip(masks, r.boxes.conf.cpu().numpy()):
                np.maximum(cm, m * np.float32(c), out=cm)
        return cm
 
    return predict
 
 
def collect_maps(predict, img_dir, lbl_dir):
    paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp") for p in img_dir.glob(ext))
    assert paths, f"Nessuna immagine in {img_dir}"
    data = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt_mask(lbl_dir, p.stem, h, w).astype(bool)
        data.append(dict(name=p.stem, gt=gt, pm=np.asarray(predict(img, p), np.float32)))
    return data
 
 
def micro(tp, fp, fn):
    e = 1e-9
    return dict(iou=tp / (tp + fp + fn + e), dice=2 * tp / (2 * tp + fp + fn + e),
                precision=tp / (tp + fp + e), recall=tp / (tp + fn + e))
 
 
def filter_small(pred, min_px):
    """Rimuove le componenti connesse con meno di min_px pixel."""
    if min_px <= 0 or not pred.any():
        return pred
    n, lab, st, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]
 
 
def counts(data, thr, min_frac=0.0):
    tp = fp = fn = 0
    for d in data:
        pred = filter_small(d["pm"] >= thr, int(min_frac * d["gt"].size))
        tp += (pred & d["gt"]).sum()
        fp += (pred & ~d["gt"]).sum()
        fn += (~pred & d["gt"]).sum()
    return tp, fp, fn
 
 
def evaluate_run(predict, root):
    """Soglia e filtro scelti sul validation (IoU micro, negativi inclusi); risultati per immagine sul test."""
    root = Path(root)
    val_dir = root / "valid" if (root / "valid").exists() else root / "val"
    val = collect_maps(predict, val_dir / "images", val_dir / "labels")
    test = collect_maps(predict, root / "test" / "images", root / "test" / "labels")
    ious = [(float(t), micro(*counts(val, t))["iou"]) for t in GRID]
    thr = max(ious, key=lambda x: x[1])[0]
    best_mf, best_iou = 0.0, -1.0
    for mf in MIN_FRACS:                                           # a parita' si preferisce il filtro piu' leggero
        iou = micro(*counts(val, thr, mf))["iou"]
        if iou > best_iou + 1e-9:
            best_mf, best_iou = mf, iou
    rows = []
    for d in test:
        pm, gt = d["pm"], d["gt"]
        pred = pm >= thr
        fpred = filter_small(pred, int(best_mf * gt.size))
        rows.append(dict(name=d["name"], tp=int((pred & gt).sum()), fp=int((pred & ~gt).sum()),
                         fn=int((~pred & gt).sum()), ftp=int((fpred & gt).sum()), ffp=int((fpred & ~gt).sum()),
                         ffn=int((~fpred & gt).sum()), npix=int(gt.size), has_gt=bool(gt.any()),
                         area_frac=float(pred.mean()), farea_frac=float(fpred.mean()), maxp=float(pm.max()),
                         thr=thr, min_frac=best_mf))
    return pd.DataFrame(rows)
 
 
# =====================================================================
# 4. ORCHESTRAZIONE: ALLENAMENTO + VALUTAZIONE
# =====================================================================
def ckpt_path(name, seed):
    return WORK / "ckpt" / f"{name}_s{seed}{'_best.pt' if CONFIGS[name]['kind'] == 'yolo' else '.pt'}"
 
 
def csv_path(name, seed):
    return WORK / "csv" / f"{name}_s{seed}.csv"
 
 
def marker_path(name, seed):
    return ckpt_path(name, seed).with_suffix(".done")       # creato solo a fine allenamento
 
 
def is_trained(name, seed):
    return ckpt_path(name, seed).exists() and marker_path(name, seed).exists()
 
 
def adopt_existing(trust_all=False):
    """I checkpoint che hanno gia' il CSV sono completi (erano stati valutati): crea il marcatore.
    Con trust_all lo crea per tutti i checkpoint esistenti."""
    d = WORK / "ckpt"
    n = 0
    if not d.exists():
        return n
    for p in d.iterdir():
        m = re.match(r"(.+)_s(\d+)(_best)?\.pt$", p.name)
        if not m or m.group(1) not in CONFIGS:
            continue
        name, seed = m.group(1), int(m.group(2))
        if ckpt_path(name, seed) == p and not marker_path(name, seed).exists() and (trust_all or csv_path(name, seed).exists()):
            marker_path(name, seed).touch()
            n += 1
    return n
 
 
def status(name, seed):
    if csv_path(name, seed).exists():
        return "OK"
    return "val" if is_trained(name, seed) else "-"
 
 
def pending(name, seed, stage):
    cp_ok = csv_path(name, seed).exists()
    if stage == "train":
        return not is_trained(name, seed) and not cp_ok
    if stage == "eval":
        return is_trained(name, seed) and not cp_ok
    return not cp_ok
 
 
def timing_file():
    return WORK / "timing.csv"
 
 
def log_time(name, seed, stage, minutes):
    f = timing_file()
    new = not f.exists()
    with open(f, "a", encoding="utf-8") as fh:
        if new:
            fh.write("config,seed,stage,minutes\n")
        fh.write(f"{name},{seed},{stage},{minutes:.2f}\n")
 
 
def est_run_minutes(name, seed=None):
    """Minuti previsti per una run: tempi misurati sulle tue run se ci sono, altrimenti le stime iniziali."""
    pre = CONFIGS[name]["pretrained"]
    t = pd.read_csv(timing_file()) if timing_file().exists() else pd.DataFrame(columns=["config", "seed", "stage", "minutes"])
 
    def med(stage, default):
        d = t[(t.stage == stage) & (t.config == name)]
        if not len(d):
            same = [n for n in CONFIGS if CONFIGS[n]["kind"] == CONFIGS[name]["kind"] and CONFIGS[n]["pretrained"] == pre]
            d = t[(t.stage == stage) & (t.config.isin(same))]
        return float(d.minutes.median()) if len(d) else default
 
    ev = med("eval", 1.0)
    if seed is not None and is_trained(name, seed):
        return ev
    return med("train", DEFAULT_MIN["pre" if pre else "scratch"] - 1.0) + ev
 
 
def run_one(name, seed, stage, quick):
    cfg = CONFIGS[name]
    ck, cp, mk = ckpt_path(name, seed), csv_path(name, seed), marker_path(name, seed)
    ck.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    if stage in ("all", "train") and not is_trained(name, seed) and not cp.exists():
        for stale in (ck, Path(str(ck) + ".tmp"), mk):          # resti di una run interrotta: non sono affidabili
            if stale.exists():
                stale.unlink()
        t0 = time.time()
        print(f"[allena] {name} seed {seed} ...", flush=True)
        (train_yolo if cfg["kind"] == "yolo" else train_ms)(name, cfg, seed, ck, quick)
        mk.touch()                                              # solo ora la run risulta completa
        minutes = (time.time() - t0) / 60
        log_time(name, seed, "train", minutes)
        print(f"   fatto in {minutes:.1f} min", flush=True)
    if stage in ("all", "eval") and is_trained(name, seed) and not cp.exists():
        t0 = time.time()
        print(f"[valuta] {name} seed {seed} ...", flush=True)
        pred = (make_yolo_predictor if cfg["kind"] == "yolo" else make_ms_predictor)(ck)
        evaluate_run(pred, ROOT).to_csv(cp, index=False)
        gc.collect()
        minutes = (time.time() - t0) / 60
        log_time(name, seed, "eval", minutes)
        print(f"   fatto in {minutes:.1f} min", flush=True)
 
 
def print_plan(configs, seeds, stage, quick):
    """Stato di ogni (configurazione, seed), run ancora da fare e minuti previsti."""
    print("\nStato (OK = completato, val = allenato e da valutare, - = da fare):")
    w = max(len(c) for c in configs)
    print(f"   {'config':<{w}}  " + "  ".join(f"s{s:<3}" for s in seeds))
    for c in configs:
        print(f"   {c:<{w}}  " + "  ".join(f"{status(c, s):<4}" for s in seeds))
    todo = [(s, n) for s in seeds for n in configs if pending(n, s, stage)]
    minutes = sum(est_run_minutes(n, s) for s, n in todo) * (0.15 if quick else 1.0)
    tempo = f"{minutes / 60:.1f} h" if minutes >= 90 else f"{minutes:.0f} min"
    src = ("tempi misurati sulle tue run, stime per il resto" if timing_file().exists()
           else "stime: ~5 min di allenamento con pesi pre-addestrati, ~12 da zero, ~1 min di valutazione")
    print(f"Da eseguire: {len(todo)} run, circa {tempo} ({src}).")
    return todo
 
 
# =====================================================================
# 5. EFFICIENZA
# =====================================================================
def build_for_efficiency(name):
    cfg = CONFIGS[name]
    if cfg["kind"] == "ms":
        m = MSYOLOv8SemSeg(YOLO_WEIGHTS, ghost=bool(cfg["ghost"]), rep=bool(cfg["rep"]), simam=bool(cfg["simam"]))
        m.eval()
        m.reparameterize()
    else:
        from ultralytics import YOLO
        m = copy.deepcopy(YOLO(YOLO_WEIGHTS).model).float().eval()
    fuse_conv_bn(m)
    return m
 
 
def time_forward(model, x, iters, warmup, sync):
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    sync()
    ts = []
    for _ in range(iters):
        s = time.perf_counter()
        with torch.no_grad():
            model(x)
        sync()
        ts.append((time.perf_counter() - s) * 1000)
    return float(np.median(ts))
 
 
def measure_efficiency(repeats=2):
    from torch.utils.flop_counter import FlopCounterMode
    has_cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if has_cuda else (lambda: None)
    torch.backends.cudnn.benchmark = True
    models = {n: build_for_efficiency(n) for n in EFF_CONFIGS}
    rows = {n: dict(config=n, params_M=sum(p.numel() for p in m.parameters()) / 1e6) for n, m in models.items()}
    for n, m in models.items():
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(torch.rand(1, 3, IMG_SIZE, IMG_SIZE))
        rows[n]["gflops"] = fc.get_total_flops() / 1e9
    lat = {n: dict(gpu1=[], gpu16=[], cpu1=[]) for n in models}
    for r in range(repeats):
        print(f"   misura latenze, giro {r + 1}/{repeats}", flush=True)
        for n, m in models.items():
            if has_cuda:
                g = copy.deepcopy(m).cuda().eval()
                lat[n]["gpu1"].append(time_forward(g, torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device="cuda"), 150, 20, sync))
                lat[n]["gpu16"].append(time_forward(g, torch.rand(16, 3, IMG_SIZE, IMG_SIZE, device="cuda"), 100, 15, sync))
                del g
            torch.set_num_threads(1)
            lat[n]["cpu1"].append(time_forward(m, torch.rand(1, 3, IMG_SIZE, IMG_SIZE), 40, 5, lambda: None))
            torch.set_num_threads(max(os.cpu_count() or 1, 1))
    for n in models:
        for k, v in lat[n].items():
            rows[n][f"lat_{k}_ms"] = float(np.median(v)) if v else np.nan
        rows[n]["gpu16_imgs"] = 16000 / rows[n]["lat_gpu16_ms"] if has_cuda else np.nan
    df = pd.DataFrame(list(rows.values()))
    df.to_csv(WORK / "efficiency.csv", index=False)
    print(df.round(2).to_string(index=False))
    return df
 
 
# =====================================================================
# 6. ANALISI E REPORT
# =====================================================================
def auc(pos_scores, neg_scores):
    n1, n0 = len(pos_scores), len(neg_scores)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.concat([pd.Series(list(pos_scores)), pd.Series(list(neg_scores))], ignore_index=True).rank(method="average")
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
 
 
def run_metrics(df):
    pos, neg = df[df.has_gt], df[~df.has_gt]
    a = micro(df.tp.sum(), df.fp.sum(), df.fn.sum())
    out = dict(iou=a["iou"], dice=a["dice"], precision=a["precision"], recall=a["recall"],
               pos_iou=micro(pos.tp.sum(), pos.fp.sum(), pos.fn.sum())["iou"],
               pos_macro_iou=float((pos.tp / (pos.tp + pos.fp + pos.fn)).mean()),
               f_iou=micro(df.ftp.sum(), df.ffp.sum(), df.ffn.sum())["iou"],
               f_pos_iou=micro(pos.ftp.sum(), pos.ffp.sum(), pos.ffn.sum())["iou"],
               thr=float(df.thr.iloc[0]), min_frac=float(df.min_frac.iloc[0]))
    if len(neg):
        out.update(fp1=float((neg.area_frac > 0.01).mean()), fpany=float((neg.fp > 0).mean()),
                   fp_area=float((neg.fp / neg.npix).mean()), auc=auc(pos.area_frac, neg.area_frac),
                   bacc=0.5 * float((pos.area_frac > 0.01).mean() + (neg.area_frac <= 0.01).mean()),
                   f_fp1=float((neg.farea_frac > 0.01).mean()),
                   f_bacc=0.5 * float((pos.farea_frac > 0.01).mean() + (neg.farea_frac <= 0.01).mean()))
    return out
 
 
def load_runs(configs, seeds):
    rows = []
    for n in configs:
        for s in seeds:
            if csv_path(n, s).exists():
                rows.append(dict(config=n, seed=s, **run_metrics(pd.read_csv(csv_path(n, s)))))
    return pd.DataFrame(rows)
 
 
def pooled_boot(name_a, name_b, seeds, n_boot=N_BOOT):
    """Bootstrap appaiato per immagine sorgente sui conteggi sommati tra i seed disponibili per entrambi."""
    seeds = [s for s in seeds if csv_path(name_a, s).exists() and csv_path(name_b, s).exists()]
    if not seeds:
        return None
 
    def stack(name):
        out = None
        for s in seeds:
            d = pd.read_csv(csv_path(name, s)).set_index("name").sort_index()
            pos = d.has_gt
            cur = pd.DataFrame({"tp": d.tp, "fp": d.fp, "fn": d.fn, "ptp": d.tp.where(pos, 0), "pfp": d.fp.where(pos, 0),
                                "pfn": d.fn.where(pos, 0), "negn": (~pos).astype(int),
                                "negfp": ((~pos) & (d.area_frac > 0.01)).astype(int)})
            out = cur if out is None else out.add(cur, fill_value=0)
        return out
 
    a, b = stack(name_a), stack(name_b)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    grp = pd.Series(common).map(lambda n: str(n).split("_")[0]).to_numpy()
    cols = ["tp", "fp", "fn", "ptp", "pfp", "pfn", "negn", "negfp"]
    G = np.hstack([a[cols].groupby(grp).sum().to_numpy(float), b[cols].groupby(grp).sum().to_numpy(float)])
    e = 1e-9
    iou = lambda t, f, n: t / (t + f + n + e)
 
    def stats(s):
        x, y = s[:8], s[8:]
        return np.array([iou(x[0], x[1], x[2]) - iou(y[0], y[1], y[2]),
                         iou(x[3], x[4], x[5]) - iou(y[3], y[4], y[5]),
                         (x[7] - y[7]) / x[6] if x[6] > 0 else np.nan])
 
    obs = stats(G.sum(0))
    rng = np.random.default_rng(0)
    boots = np.array([stats(G[rng.integers(0, len(G), len(G))].sum(0)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return dict(n_groups=len(G), n_seeds=len(seeds), obs=obs, lo=lo, hi=hi)
 
 
def verdict(lo, hi, higher_better=True):
    if lo > 0:
        return "A meglio" if higher_better else "B meglio"
    if hi < 0:
        return "B meglio" if higher_better else "A meglio"
    return "non distinguibile"
 
 
def pm(values, scale=100, d=1):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return "-"
    if len(v) == 1:
        return f"{v[0] * scale:.{d}f}"
    return f"{v.mean() * scale:.{d}f} ± {v.std(ddof=1) * scale:.{d}f}"
 
 
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)
 
 
def make_plots(runs, eff, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   (matplotlib non disponibile: salto i grafici)")
        return
    base = runs[runs.config == "ms_base"].set_index("seed")
    names = [n for n in ABLATION if n in set(runs.config)]
    if len(base) and names:
        fig, ax = plt.subplots(figsize=(7, 4))
        means, sds = [], []
        for n in names:
            d = runs[runs.config == n].set_index("seed")
            delta = ((d.pos_iou - base.pos_iou).dropna() * 100).to_numpy()
            means.append(delta.mean() if len(delta) else 0)
            sds.append(delta.std(ddof=1) if len(delta) > 1 else 0)
        ax.bar(names, means, yerr=sds, capsize=4, color="#4C78A8")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Δ IoU sui difetti rispetto a ms_base (punti)")
        ax.set_title("Ablation: un modulo alla volta (media ± std tra seed)")
        plt.xticks(rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_ablation.png", dpi=150)
        plt.close(fig)
    if eff is not None and len(runs):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for n in eff.config:
            r = runs[runs.config == n]
            if len(r):
                e = eff[eff.config == n].iloc[0]
                ax.scatter(e.gflops, r.pos_iou.mean() * 100, s=50)
                ax.annotate(n, (e.gflops, r.pos_iou.mean() * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("GFLOPs @256")
        ax.set_ylabel("IoU sui difetti (%)")
        ax.set_title("Accuratezza e costo")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_accuratezza_vs_costo.png", dpi=150)
        plt.close(fig)
 
 
def build_report(configs, seeds, n_boot):
    runs = load_runs(configs, seeds)
    if runs.empty:
        print("Nessun risultato in", WORK / "csv")
        return
    out_dir = WORK / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    eff = pd.read_csv(WORK / "efficiency.csv") if (WORK / "efficiency.csv").exists() else None
    md = ["# Risultati finali", "",
          f"Dataset: `{ROOT}` | seed disponibili: {sorted(runs.seed.unique().tolist())} | "
          f"configurazioni: {runs.config.nunique()}", ""]
 
    ex = pd.read_csv(csv_path(runs.config.iloc[0], int(runs.seed.iloc[0])))
    n_src = ex.name.map(lambda n: str(n).split("_")[0]).nunique()
    md += [f"Test: {len(ex)} patch ({int(ex.has_gt.sum())} con difetto, {int((~ex.has_gt).sum())} senza) da {n_src} immagini sorgente.", ""]
 
    cols = [("iou", "IoU tutte"), ("pos_iou", "IoU difetti"), ("fp1", "FP neg >1%"), ("auc", "AUC patch"),
            ("bacc", "Acc.bil. patch"), ("f_fp1", "FP neg >1% (filtro)"), ("f_pos_iou", "IoU difetti (filtro)")]
    t1 = []
    for n in configs:
        d = runs[runs.config == n]
        if len(d):
            t1.append({"config": n, "seed": len(d), **{h: pm(d[k]) for k, h in cols if k in d}})
    md += ["## 1. Riepilogo (%, media ± std tra seed; con un solo seed e' il valore singolo, senza incertezza di training)", "", md_table(pd.DataFrame(t1)), ""]
 
    base = runs[runs.config == "ms_base"].set_index("seed")
    t2 = []
    for n in ABLATION:
        d = runs[runs.config == n].set_index("seed")
        if not len(d) or not len(base):
            continue
        row = {"modulo aggiunto a ms_base": n}
        for k, h in (("pos_iou", "Δ IoU difetti (pt)"), ("fp1", "Δ FP neg >1% (pt)"), ("auc", "Δ AUC (pt)")):
            if k in d and k in base:
                delta = (d[k] - base[k]).dropna()
                row[h] = f"{pm(delta.to_numpy(), d=2)}  [{int((delta > 0).sum())}/{len(delta)} seed >0]" if len(delta) else "-"
        b = pooled_boot(n, "ms_base", seeds, n_boot)
        if b:
            row["ΔIoU difetti, IC95% bootstrap"] = (f"{b['obs'][1] * 100:+.2f} [{b['lo'][1] * 100:+.2f}, {b['hi'][1] * 100:+.2f}] "
                                                    f"→ {verdict(b['lo'][1], b['hi'][1])}")
        t2.append(row)
    if t2:
        md += ["## 2. Ablation causale (stessa pipeline, un modulo alla volta)", "",
               "Δ = configurazione - ms_base. Per i falsi positivi un valore negativo e' meglio. 'A meglio' = la configurazione con il modulo.", "",
               md_table(pd.DataFrame(t2)), ""]
 
    t3 = []
    for a_, b_ in PAIRS:
        b = pooled_boot(a_, b_, seeds, n_boot)
        if b:
            t3.append({"A": a_, "B": b_, "seed": b["n_seeds"],
                       "ΔIoU tutte [IC95%]": f"{b['obs'][0] * 100:+.2f} [{b['lo'][0] * 100:+.2f}, {b['hi'][0] * 100:+.2f}]",
                       "ΔIoU difetti [IC95%]": f"{b['obs'][1] * 100:+.2f} [{b['lo'][1] * 100:+.2f}, {b['hi'][1] * 100:+.2f}]",
                       "esito (IoU difetti)": verdict(b["lo"][1], b["hi"][1]),
                       "Δ FP neg >1% [IC95%]": (f"{b['obs'][2] * 100:+.2f} [{b['lo'][2] * 100:+.2f}, {b['hi'][2] * 100:+.2f}]"
                                                if not np.isnan(b['obs'][2]) else "-"),
                       "esito (FP)": (verdict(b["lo"][2], b["hi"][2], higher_better=False) if not np.isnan(b['obs'][2]) else "-")})
    if t3:
        md += [f"## 3. Confronti diretti (bootstrap per immagine sorgente, {t3[0].get('seed', '?')}+ seed sommati)", "",
               "Δ = A - B. 'A meglio' o 'B meglio' solo se l'intervallo esclude lo zero.", "", md_table(pd.DataFrame(t3)), ""]
 
    if eff is not None:
        t4 = eff.round(2).rename(columns={"params_M": "Parametri (M)", "gflops": "GFLOPs@256", "lat_gpu1_ms": "GPU b=1 (ms)",
                                          "lat_gpu16_ms": "GPU b=16 (ms)", "gpu16_imgs": "GPU b=16 (img/s)",
                                          "lat_cpu1_ms": "CPU 1 thread (ms)"})
        md += ["## 4. Efficienza (forma deploy: RepBlock e Conv+BN fusi)", "", md_table(t4),
               "", "Le varianti da zero e `ms_dsiou` hanno la stessa architettura di: " +
               ", ".join(f"{k}→{v}" for k, v in ARCH_OF.items()) + ".", ""]
 
    md += ["## 5. Come leggere e limiti", "",
           "- Soglia, filtro delle componenti piccole e checkpoint sono scelti solo sul validation; il test non entra in nessuna scelta.",
           "- Gli intervalli del bootstrap tengono conto del campione di immagini di test, non della variabilita' del training: per quella servono i seed.",
           "- Differenze sotto circa 1 punto di IoU tra configurazioni non si interpretano; con pochi seed la deviazione standard e' grossolana.",
           "- Il test contiene negativi in rapporto 1:1: in una facciata reale la prevalenza dei negativi e' molto maggiore e i falsi positivi pesano di piu'.",
           "- Un solo dataset (S2DS, immagini di calcestruzzo) e un solo hardware per l'efficienza.",
           "- 'FP neg >1%' = frazione di patch senza difetto con piu' dell'1% dell'area predetta; AUC e accuratezza bilanciata usano l'area predetta come punteggio della patch.",
           "- S2DS: solo uso accademico, non ridistribuibile (Benz & Rodehorst, DAGM 2022)."]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    make_plots(runs, eff, out_dir)
    print("\n".join(md))
    print(f"\nReport salvato in {out_dir}")
 
 
# =====================================================================
# 7. MAIN
# =====================================================================
def main():
    global ROOT, WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval", "efficiency", "report"])
    ap.add_argument("--quick", action="store_true", help="prova rapida: 1 seed, 3 configurazioni, poche epoche")
    ap.add_argument("--preset", default=None, help=f"insieme predefinito di run: {', '.join(PRESETS)}")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--plan", action="store_true", help="mostra cosa e' fatto e cosa manca, senza eseguire")
    ap.add_argument("--budget-min", type=float, default=None, help="si ferma (tra due run) dopo questo numero di minuti")
    ap.add_argument("--force-efficiency", action="store_true", help="rifa la misura di efficienza anche se esiste")
    ap.add_argument("--trust-existing", action="store_true", help="considera completi tutti i checkpoint esistenti")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--work", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a, _ = ap.parse_known_args()
    ROOT = Path(a.root)
    WORK = Path(a.work) if a.work else (Path(str(WORK) + "_quick") if a.quick else WORK)
    preset = None
    if a.preset:
        preset = PRESETS.get(a.preset)
        if preset is None:
            sys.exit(f"Preset sconosciuto: {a.preset}. Validi: {list(PRESETS)}")
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else (preset["seeds"] if preset else ([0] if a.quick else SEEDS))
    configs = a.configs.split(",") if a.configs else (preset["configs"] if preset else (QUICK_CONFIGS if a.quick else list(CONFIGS)))
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        sys.exit(f"Configurazioni sconosciute: {bad}. Valide: {list(CONFIGS)}")
    WORK.mkdir(parents=True, exist_ok=True)
    adopted = adopt_existing(a.trust_existing)
    print(f"Dataset: {ROOT}\nCartella lavoro: {WORK}\nConfigurazioni: {configs}\nSeed: {seeds}")
    if adopted:
        print(f"Checkpoint esistenti riconosciuti come completi: {adopted}")
 
    failures, budget_hit = [], False
    train_stages = ("all", "train", "eval")
    if a.stage in train_stages or a.plan:
        todo = print_plan(configs, seeds, a.stage if a.stage in train_stages else "all", a.quick)
        if a.plan:
            return
        t_start = time.time()
        for i, (s, n) in enumerate(todo if a.stage in train_stages else []):
            elapsed = (time.time() - t_start) / 60
            if a.budget_min and i > 0 and elapsed + est_run_minutes(n, s) > a.budget_min:
                budget_hit = True
                print(f"\n[budget] mi fermo dopo {elapsed:.0f} min: restano {len(todo) - i} run. "
                      "Rilancia lo stesso comando per continuare (riprende da qui).")
                break
            try:
                run_one(n, s, a.stage, a.quick)
            except Exception:
                failures.append((n, s))
                with open(WORK / "errors.log", "a", encoding="utf-8") as f:
                    f.write(f"--- {n} seed {s}\n{traceback.format_exc()}\n")
                print(f"[ERRORE] {n} seed {s}: vedi {WORK / 'errors.log'}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    eff_exists = (WORK / "efficiency.csv").exists()
    if a.stage == "efficiency" or (a.stage == "all" and not budget_hit and (a.force_efficiency or not eff_exists)):
        print("\n[efficienza] misura parametri, GFLOPs e latenze ...")
        try:
            measure_efficiency()
        except Exception:
            failures.append(("efficiency", -1))
            with open(WORK / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"--- efficiency\n{traceback.format_exc()}\n")
            print(f"[ERRORE] efficienza: vedi {WORK / 'errors.log'}")
    elif a.stage == "all" and eff_exists:
        print("\n[efficienza] gia' misurata (efficiency.csv): salto (usa --force-efficiency per rifarla)")
    if a.stage in ("all", "report"):
        # il report usa tutti i seed e le configurazioni con dati, non solo quelli di questa esecuzione
        build_report(configs if a.configs else list(CONFIGS), sorted(set(seeds) | set(SEEDS)), a.n_boot)
    if failures:
        print(f"\nRun falite: {failures}. Dettagli in {WORK / 'errors.log'}; rilancia per riprovare solo quelle.")
 
 
if __name__ == "__main__":
    main()