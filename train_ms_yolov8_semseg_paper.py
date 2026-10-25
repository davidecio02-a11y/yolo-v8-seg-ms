
import os
import random
 
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
 
# =====================================================================
# CONFIG
# =====================================================================
WEIGHTS = "yolov8s-seg.pt"
PRETRAINED = True               # False = regime del paper (da zero)
GHOST_K = 5                     # filtro cheap-op Ghost (ablation del paper: 3/5/7/9)
DATASET_ROOT = "D:/yolo_seg/spalling_corrosion_patches_v2_new"   # positivi + negativi
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS_WARMUP = 5               # solo parti nuove (se PRETRAINED)
EPOCHS_FINETUNE = 45
EPOCHS_SCRATCH = 120
LR_WARMUP = 1e-3
LR_NEW = 5e-4
LR_PRETRAINED = 1e-4
LR_SCRATCH = 1e-3               # il paper usa 6e-5 con 120 epoche
BCE_WEIGHT = 0.4                # 0.0 => solo DSIoU come nel paper
NEG_WEIGHT = 1.0                # peso della penalita' sui falsi positivi nelle immagini SENZA difetto
                                # (None = comportamento precedente: il soft-IoU non da gradiente sui negativi)
SEED = 42
THR_GRID = torch.arange(0.20, 0.75, 0.05)
SAVE_PATH = "ms_yolov8_semseg_paper_best.pt"
 
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
# =====================================================================
# 1. MODULI
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
    """Primaria (k, s) su metà canali + cheap-op depthwise (GHOST_K) sull'altra metà."""
 
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
    """Downsample 3x3 stride 2 del neck (sostituisce il Conv di YOLOv8)."""
 
    def __init__(self, c1, c2):
        super().__init__(c1, c2, k=3, s=2)
 
 
class GhostBottleneck(nn.Module):
    """Sostituisce Bottleneck di YOLOv8 dentro C2f -> GC-C2f."""
 
    def __init__(self, c, shortcut):
        super().__init__()
        self.cv = nn.Sequential(GhostConv(c, c), GhostConv(c, c))
        self.add = shortcut
        last = self.cv[1]  # avvio "neutro": il ramo Ghost parte da 0 e cresce col training
        nn.init.zeros_(last.cv1.bn.weight)
        nn.init.zeros_(last.cv2.bn.weight)
 
    def forward(self, x):
        y = self.cv(x)
        return x + y if self.add else y
 
 
def _fuse_conv_bn(w, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return w * t, bn.bias - bn.running_mean * bn.weight / std
 
 
class RepBlock(nn.Module):
    """Training: 3x3+BN, 1x1+BN, (identity+BN). Inference: singola conv 3x3 (Eq. 3-6 del paper)."""
 
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
        """Chiamare con il modello in eval() (usa le statistiche BN)."""
        if self.deploy:
            return
        w3, b3 = _fuse_conv_bn(self.conv3.weight, self.bn3)
        w1, b1 = _fuse_conv_bn(F.pad(self.conv1.weight, [1, 1, 1, 1]), self.bn1)
        w, b = w3 + w1, b3 + b1
        if self.bn_id is not None:
            ident = torch.zeros_like(w3)
            for i in range(self.c1):
                ident[i, i, 1, 1] = 1.0
            wi, bi = _fuse_conv_bn(ident, self.bn_id)
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
 
 
class SimAMHead(nn.Module):
    """P3(proto pretrained) + P4 + P5 + skip s4 -> RepBlock -> SimAM -> up + skip s2 -> logit."""
 
    def __init__(self, nm, c_s4, c_s2, c4, c5, mid=32):
        super().__init__()
        self.b4 = RepBlock(c4, mid)
        self.b5 = RepBlock(c5, mid)
        self.skip4 = Conv(c_s4, mid, 1)
        self.fuse4 = RepBlock(nm + 3 * mid, mid)
        self.simam = SimAM()
        self.up = nn.ConvTranspose2d(mid, mid, 2, 2, bias=False)
        self.bn_up = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()
        self.skip2 = Conv(c_s2, mid, 1)
        self.fuse2 = RepBlock(2 * mid, mid)
        self.out = nn.Conv2d(mid, 1, 1)
        nn.init.constant_(self.out.bias, -2.0)
 
    def forward(self, proto, p4, p5, f_s4, f_s2):
        x = torch.cat([proto, _to(self.b4(p4), proto), _to(self.b5(p5), proto),
                       _to(self.skip4(f_s4), proto)], 1)
        x = self.simam(self.fuse4(x))
        x = self.act(self.bn_up(self.up(x)))
        x = self.fuse2(torch.cat([x, _to(self.skip2(f_s2), x)], 1))
        return self.out(x)
 
 
# =====================================================================
# 2. MODELLO: YOLOv8-seg "ghostificato"
# =====================================================================
def ghostify(layers, sppf_idx):
    """Modifica in-place la lista di layer YOLOv8. cv1/cv2 dei C2f restano pretrained."""
    n_c2f = n_down = 0
    for i, m in enumerate(layers):
        name = type(m).__name__
        if name == "C2f":
            shortcut = bool(getattr(m.m[0], "add", False))
            m.m = nn.ModuleList(GhostBottleneck(m.c, shortcut) for _ in range(len(m.m)))
            n_c2f += 1
        elif name == "Conv" and i > sppf_idx and m.conv.stride == (2, 2):
            g = GhostDown(m.conv.in_channels, m.conv.out_channels)
            ch = m.conv.out_channels // 2  # slicing: prime ch canali dei pesi YOLO -> primaria Ghost
            with torch.no_grad():
                g.cv1.conv.weight.copy_(m.conv.weight[:ch])
                for n in ("weight", "bias", "running_mean", "running_var"):
                    getattr(g.cv1.bn, n).copy_(getattr(m.bn, n)[:ch])
            g.f, g.i = m.f, m.i
            layers[i] = g
            n_down += 1
    assert n_c2f > 0, "Nessun C2f trovato: serve un checkpoint YOLOv8-seg"
    return n_c2f, n_down
 
 
class MSYOLOv8SemSeg(nn.Module):
    def __init__(self, weights=WEIGHTS):
        super().__init__()
        det = YOLO(weights).model.float()
        layers = list(det.model)
        head = layers[-1]
        assert hasattr(head, "proto"), "Il checkpoint deve essere *-seg"
        sppf_idx = next(i for i, m in enumerate(layers) if type(m).__name__ == "SPPF")
        body = layers[:-1]
        self.n_c2f, self.n_down = ghostify(body, sppf_idx)
        self.layers = nn.ModuleList(body)
        self.proto = head.proto
        self.p3, self.p4, self.p5 = head.f  # 15, 18, 21 in YOLOv8
 
        with torch.no_grad():  # dry run per i canali (dipendono dalla taglia n/s/m/...)
            self.eval()
            y = self._run(torch.zeros(1, 3, 256, 256))
            nm = self.proto(y[self.p3]).shape[1]
            c_s2, c_s4 = y[0].shape[1], y[2].shape[1]
            c4, c5 = y[self.p4].shape[1], y[self.p5].shape[1]
        self.head = SimAMHead(nm, c_s4, c_s2, c4, c5)
 
        for p in self.parameters():
            p.requires_grad_(True)
 
        # parti NUOVE (random): bottleneck Ghost, cheap-op dei downsample, testa
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
        proto = self.proto(y[self.p3])
        logits = self.head(proto, y[self.p4], y[self.p5], y[2], y[0])
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits.squeeze(1)  # (B, H, W)
 
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
    if frozen:  # i BN con parametri congelati non devono aggiornare le statistiche
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                m.eval()
 
 
# =====================================================================
# 3. DATASET
# =====================================================================
class SpallingDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=256, alpha=1.0, tau=0.5, is_train=True):
        self.images_dir, self.labels_dir = images_dir, labels_dir
        self.img_size, self.alpha, self.tau, self.is_train = img_size, alpha, tau, is_train
        exts = (".jpg", ".png", ".jpeg", ".bmp")
        self.files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(exts))
        missing = sum(
            not (os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt"))
                 or os.path.exists(os.path.join(labels_dir, os.path.splitext(f)[0] + ".png")))
            for f in self.files)
        empty = sum(
            os.path.exists(t := os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt")) and os.path.getsize(t) == 0
            for f in self.files)
        print(f"[dataset] {images_dir}: {len(self.files)} immagini, {empty} con label vuota (senza difetto), "
              f"{missing} senza file label (sfondo)")
 
    def __len__(self):
        return len(self.files)
 
    def _edt_weight_map(self, mask):
        h, w = mask.shape
        small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST) > 0.5
        if not small.any() or small.all():
            return np.ones((h, w), np.float32)
        # distanza dal bordo = dist_inside + dist_outside (ciascuna è nulla nella regione opposta)
        dist = distance_transform_edt(small) + distance_transform_edt(~small)
        ws = 1.0 + self.alpha * np.exp(-self.tau * dist)
        return cv2.resize(ws.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
 
    def _load_mask(self, base):
        s = self.img_size
        mask = np.zeros((s, s), np.float32)
        txt = os.path.join(self.labels_dir, base + ".txt")
        png = os.path.join(self.labels_dir, base + ".png")
        if os.path.exists(txt):
            with open(txt) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 5:
                        c = np.array(parts[1:], np.float32).reshape(-1, 2)
                        c[:, 0] *= s
                        c[:, 1] *= s
                        cv2.fillPoly(mask, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], 1.0)
        elif os.path.exists(png):
            m = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
            mask = (m > 0).astype(np.float32)
        return mask
 
    def __getitem__(self, idx):
        name = self.files[idx]
        base = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(self.images_dir, name))
        if img is None:
            raise IOError(f"Impossibile leggere {name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = self._load_mask(base)
 
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
 
        w = self._edt_weight_map(mask)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # RGB/255 come Ultralytics
        return img_t, torch.from_numpy(mask).float(), torch.from_numpy(w).float()
 
 
# =====================================================================
# 4. LOSS (DSIoU + BCE pesata opzionale) E VALIDAZIONE
# =====================================================================
class DSIoULoss(nn.Module):
    """Soft-IoU pesato sul bordo (DSIoU) + BCE pesata opzionale.
 
    Con target vuoto il soft-IoU vale ~1 qualunque sia la predizione e il suo gradiente e' ~0
    (verificato numericamente): sui negativi supervisiona solo la BCE. Con neg_weight (numero) l'IoU
    viene mediato sulle sole immagini con difetto e sulle immagini senza difetto si aggiunge la
    probabilita' media predetta (area di falso positivo). neg_weight=None = comportamento precedente.
    """
 
    def __init__(self, bce_weight=0.4, neg_weight=None, eps=1e-6):
        super().__init__()
        self.bce_weight, self.neg_weight, self.eps = bce_weight, neg_weight, eps
 
    def forward(self, logits, target, weight):
        logits = logits.float()
        probs = torch.sigmoid(logits)
        inter = (weight * probs * target).sum(dim=[1, 2])
        union = (weight * (probs + target - probs * target)).sum(dim=[1, 2])
        iou_loss = 1.0 - (inter + self.eps) / (union + self.eps)          # (B,)
        if self.neg_weight is None:
            loss = iou_loss.mean()
        else:
            has_obj = target.flatten(1).sum(dim=1) > 0                    # (B,)
            zero = logits.new_zeros(())
            pos_term = iou_loss[has_obj].mean() if has_obj.any() else zero
            neg_term = (weight * probs).mean(dim=[1, 2])[~has_obj].mean() if (~has_obj).any() else zero
            loss = pos_term + self.neg_weight * neg_term
        if self.bce_weight > 0:
            bce = (weight * F.binary_cross_entropy_with_logits(logits, target, reduction="none")).mean()
            loss = self.bce_weight * bce + (1.0 - self.bce_weight) * loss
        return loss
 
 
@torch.no_grad()
def validate(model, loader, criterion, device, thrs):
    model.eval()
    thrs = thrs.to(device)
    tp = torch.zeros(len(thrs), device=device)
    fp, fn = torch.zeros_like(tp), torch.zeros_like(tp)
    neg_fp, neg_n = torch.zeros_like(tp), 0          # immagini senza difetto con >1% di pixel predetti
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
        is_neg = gt.flatten(1).sum(dim=1) == 0                              # (B,)
        if is_neg.any():
            area = pred.float().mean(dim=(2, 3))                            # (T, B)
            neg_fp += ((area > 0.01) & is_neg.unsqueeze(0)).sum(dim=1)
            neg_n += int(is_neg.sum())
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    prec, rec = tp / (tp + fp + eps), tp / (tp + fn + eps)
    b = int(iou.argmax())
    return dict(loss=loss_sum / max(n, 1), thr=float(thrs[b]), iou=float(iou[b]),
                dice=float(dice[b]), precision=float(prec[b]), recall=float(rec[b]),
                neg_fp_rate=(float(neg_fp[b]) / neg_n) if neg_n else None)
 
 
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
 
 
# =====================================================================
# 5. INFERENZA / DEPLOY
# =====================================================================
def load_for_inference(path, device="cpu", deploy=True):
    global GHOST_K
    ck = torch.load(path, map_location=device)
    GHOST_K = ck.get("ghost_k", GHOST_K)   # il filtro Ghost deve coincidere con quello del training
    model = MSYOLOv8SemSeg(ck["weights"])
    model.load_state_dict(ck["state_dict"])  # carica PRIMA di fondere i RepBlock
    model = model.to(device).eval()
    if deploy:
        model.reparameterize()
    return model, ck["thr"]
 
 
@torch.no_grad()
def check_reparam(path, sample, device):
    """Verifica che il modello fuso dia lo stesso output di quello multi-ramo."""
    ref_model, _ = load_for_inference(path, device, deploy=False)
    ref = ref_model(sample.to(device))
    fused_model, _ = load_for_inference(path, device, deploy=True)
    out = fused_model(sample.to(device))
    print(f"[reparam] max |diff| logit = {(ref - out).abs().max().item():.2e}")
 
 
# =====================================================================
# 6. MAIN
# =====================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--pretrained", type=int, default=int(PRETRAINED), help="1 = pesi YOLO, 0 = da zero")
    ap.add_argument("--dataset", default=DATASET_ROOT)
    ap.add_argument("--save", default=None, help="file del checkpoint (default: ms_semseg_neg_<pre|scratch>_s<seed>.pt)")
    args, _ = ap.parse_known_args()      # tollera gli argomenti extra di Spyder
    SEED, PRETRAINED, DATASET_ROOT = args.seed, bool(args.pretrained), args.dataset
    SAVE_PATH = args.save or f"ms_semseg_neg_{'pre' if PRETRAINED else 'scratch'}_s{SEED}.pt"
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Esecuzione su: {device} | pretrained={PRETRAINED} | ghost_k={GHOST_K} | seed={SEED} | "
          f"neg_weight={NEG_WEIGHT} | checkpoint: {SAVE_PATH}")
 
    tr_img = os.path.join(DATASET_ROOT, "train", "images")
    tr_lbl = os.path.join(DATASET_ROOT, "train", "labels")
    va_img = os.path.join(DATASET_ROOT, "valid", "images")
    va_lbl = os.path.join(DATASET_ROOT, "valid", "labels")
    if not os.path.exists(va_img):
        va_img = os.path.join(DATASET_ROOT, "val", "images")
        va_lbl = os.path.join(DATASET_ROOT, "val", "labels")
 
    train_ds = SpallingDataset(tr_img, tr_lbl, IMG_SIZE, is_train=True)
    val_ds = SpallingDataset(va_img, va_lbl, IMG_SIZE, is_train=False)
    kw = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
              pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, **kw)
 
    model = MSYOLOv8SemSeg(WEIGHTS).to(device)
    n_pre = sum(p.numel() for p in model.pretrained_params())
    n_new = sum(p.numel() for p in model.new_params())
    print(f"[+] C2f->GC-C2f: {model.n_c2f} | downsample->GhostConv: {model.n_down}")
    print(f"[+] Parametri con init YOLOv8-seg: {n_pre/1e6:.2f}M ({100*n_pre/(n_pre+n_new):.1f}%) | "
          f"nuovi: {n_new/1e6:.2f}M")
 
    with torch.no_grad():
        model.eval()
        out = model(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=device))
        assert out.shape == (2, IMG_SIZE, IMG_SIZE), out.shape
 
    if not PRETRAINED:
        reset_all(model)
 
    criterion = DSIoULoss(bce_weight=BCE_WEIGHT, neg_weight=NEG_WEIGHT)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
 
    if PRETRAINED:
        print(f"\n[+] Fase 1: warmup parti nuove ({EPOCHS_WARMUP} epoche), pesi YOLO congelati")
        set_frozen(model, True)
        opt = torch.optim.AdamW(model.new_params(), lr=LR_WARMUP, weight_decay=0.01)
        for ep in range(1, EPOCHS_WARMUP + 1):
            tl = train_one_epoch(model, train_loader, criterion, opt, scaler, device, frozen=True)
            v = validate(model, val_loader, criterion, device, THR_GRID)
            print(f"    Warmup {ep}/{EPOCHS_WARMUP} | train {tl:.4f} | val {v['loss']:.4f} | "
                  f"IoU {v['iou']*100:.2f}% (thr {v['thr']:.2f})")
        set_frozen(model, False)
        groups = [{"params": model.pretrained_params(), "lr": LR_PRETRAINED},
                  {"params": model.new_params(), "lr": LR_NEW}]
        epochs = EPOCHS_FINETUNE
    else:
        groups = [{"params": list(model.parameters()), "lr": LR_SCRATCH}]
        epochs = EPOCHS_SCRATCH
 
    print(f"\n[+] Fase 2: training completo ({epochs} epoche), PolyLR potenza 1.0 come nel paper")
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: max(1.0 - e / epochs, 0.01))
 
    best_iou = 0.0
    for ep in range(1, epochs + 1):
        tl = train_one_epoch(model, train_loader, criterion, opt, scaler, device, frozen=False)
        sched.step()
        v = validate(model, val_loader, criterion, device, THR_GRID)
        print(f"    Epoca {ep:03d}/{epochs} | train {tl:.4f} | val {v['loss']:.4f} | "
              f"IoU {v['iou']*100:.2f}% Dice {v['dice']*100:.2f}% "
              f"P {v['precision']*100:.1f}% R {v['recall']*100:.1f}% (thr {v['thr']:.2f})"
              + (f" | FP su negativi {v['neg_fp_rate']*100:.1f}%" if v["neg_fp_rate"] is not None else ""))
        if v["iou"] > best_iou:
            best_iou = v["iou"]
            torch.save({"state_dict": model.state_dict(), "weights": WEIGHTS,
                        "thr": v["thr"], "img_size": IMG_SIZE, "val": v, "pretrained": PRETRAINED, "seed": SEED,
                        "ghost_k": GHOST_K, "bce_weight": BCE_WEIGHT, "neg_weight": NEG_WEIGHT,
                        "dataset": DATASET_ROOT}, SAVE_PATH)
            print(f"      [*] Salvato best (IoU {best_iou*100:.2f}%, soglia {v['thr']:.2f})")
 
    print(f"\n[OK] Miglior Val IoU (micro): {best_iou*100:.2f}%")
    sample, _, _ = val_ds[0]
    check_reparam(SAVE_PATH, sample.unsqueeze(0), device)
    print("     Valuta sul TEST con la soglia salvata (ck['thr']); load_for_inference() fonde i RepBlock.")