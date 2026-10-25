# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:01:19 2026

@author: david
"""
import os
import tempfile
import time
from pathlib import Path
 
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
 
st.set_page_config(page_title="Ispezione calcestruzzo: spalling e corrosione", page_icon="🏗️",
                   layout="wide", initial_sidebar_state="expanded")
 
TILE = 256
CONFRONTO = "Confronto affiancato (solo immagini)"
PROFILES = {
    "standard": dict(
        titolo="Segmentazione standard: YOLOv8s-seg", kind="yolo", colore=(0, 140, 255),
        percorso=r"D:\yolo_seg\final_runs\ckpt\yolo_pre_s0_best.pt",
        soglia_label="Soglia di confidenza", soglia_default=0.35, area_min_default=0.0,
        nota="Segmentazione di istanze. Nei test i falsi allarmi sono piu' rari, ma quando avvengono sono molto estesi."),
    "efficient": dict(
        titolo="Segmentazione efficiente: MS-YOLOv8-Seg (dispositivi mobili / edge)", kind="ms", colore=(0, 0, 255),
        percorso=r"D:\yolo_seg\final_runs\ckpt\ms_full_s0.pt",
        soglia_label="Soglia di probabilita'", soglia_default=0.30, area_min_default=1.0,
        nota="Modello semantico compatto. Nei test i falsi allarmi sono piu' frequenti ma di piccole dimensioni."),
}
# Valori misurati nella tesi: test S2DS (374 tessere, 3 seed) e latenze su RTX 5050 Laptop / CPU a un thread.
SCHEDA = {
    "standard": [("Parametri", "11,81 M"), ("GFLOPs a 256 px", "6,41"), ("Latenza CPU (1 thread)", "96 ms"),
                 ("Latenza GPU (batch 1)", "8,9 ms"), ("IoU sui difetti", "86,5 ± 0,7 %"),
                 ("Tessere pulite con >1% di area segnata", "10,9 ± 1,5 %")],
    "efficient": [("Parametri", "5,76 M"), ("GFLOPs a 256 px", "4,50"), ("Latenza CPU (1 thread)", "85 ms"),
                  ("Latenza GPU (batch 1)", "8,8 ms"), ("IoU sui difetti", "85,2 ± 0,2 %"),
                  ("Tessere pulite con >1% di area segnata", "22,5 ± 3,2 %")],
}
 
 
# =====================================================================
# 1. ARCHITETTURA MS-YOLOv8-Seg (identica a ms_full di final_project.py)
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
    def __init__(self, c1, c2, k=1, s=1, dw_k=5):
        super().__init__()
        assert c2 % 2 == 0, "GhostConv richiede canali di uscita pari"
        c_half = c2 // 2
        self.cv1 = Conv(c1, c_half, k, s)
        self.cv2 = Conv(c_half, c_half, dw_k, 1, g=c_half)
 
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)
 
 
class GhostDown(GhostConv):
    def __init__(self, c1, c2):
        super().__init__(c1, c2, k=3, s=2)
 
 
class GhostBottleneck(nn.Module):
    def __init__(self, c, shortcut):
        super().__init__()
        self.cv = nn.Sequential(GhostConv(c, c), GhostConv(c, c))
        self.add = shortcut
 
    def forward(self, x):
        y = self.cv(x)
        return x + y if self.add else y
 
 
def _fuse_conv_bn(w, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return w * t, bn.bias - bn.running_mean * bn.weight / std
 
 
class RepBlock(nn.Module):
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
 
    def forward(self, proto, p4, p5, f_s4, f_s2):
        x = torch.cat([proto, _to(self.b4(p4), proto), _to(self.b5(p5), proto), _to(self.skip4(f_s4), proto)], 1)
        x = self.simam(self.fuse4(x))
        x = self.act(self.bn_up(self.up(x)))
        x = self.fuse2(torch.cat([x, _to(self.skip2(f_s2), x)], 1))
        return self.out(x)
 
 
def ghostify(layers, sppf_idx):
    for i, m in enumerate(layers):
        name = type(m).__name__
        if name == "C2f":
            shortcut = bool(getattr(m.m[0], "add", False))
            m.m = nn.ModuleList(GhostBottleneck(m.c, shortcut) for _ in range(len(m.m)))
        elif name == "Conv" and i > sppf_idx and m.conv.stride == (2, 2):
            g = GhostDown(m.conv.in_channels, m.conv.out_channels)
            g.f, g.i = m.f, m.i                       # mantiene il routing dei tensori di Ultralytics
            layers[i] = g
 
 
class MSYOLOv8SemSeg(nn.Module):
    def __init__(self, weights="yolov8s-seg.pt"):
        super().__init__()
        from ultralytics import YOLO
        det = YOLO(weights).model.float()
        layers = list(det.model)
        head = layers[-1]
        sppf_idx = next(i for i, m in enumerate(layers) if type(m).__name__ == "SPPF")
        body = layers[:-1]
        ghostify(body, sppf_idx)
        self.layers = nn.ModuleList(body)
        self.proto = head.proto
        self.p3, self.p4, self.p5 = head.f
        with torch.no_grad():
            self.eval()
            y = self._run(torch.zeros(1, 3, TILE, TILE))
            nm = self.proto(y[self.p3]).shape[1]
            c_s2, c_s4 = y[0].shape[1], y[2].shape[1]
            c4, c5 = y[self.p4].shape[1], y[self.p5].shape[1]
        self.head = SimAMHead(nm, c_s4, c_s2, c4, c5)
 
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
        return logits.squeeze(1)                      # (B, H, W)
 
    def reparameterize(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.reparameterize()
        return self
 
 
# =====================================================================
# 2. CARICAMENTO DEI MODELLI (con cache di Streamlit)
# =====================================================================
@st.cache_resource(show_spinner="Caricamento di MS-YOLOv8-Seg...")
def load_ms_model(path, device):
    if not os.path.exists(path):
        return None, 0.30, f"File dei pesi non trovato: {path}"
    try:
        ck = torch.load(path, map_location="cpu")
        is_dict = isinstance(ck, dict)
        state = ck["state_dict"] if is_dict and "state_dict" in ck else ck
        model = MSYOLOv8SemSeg(ck.get("weights", "yolov8s-seg.pt") if is_dict else "yolov8s-seg.pt")
        model.load_state_dict(state)
        model = model.to(device).eval().reparameterize()          # fonde i RepBlock per l'inferenza
        with torch.no_grad():                                      # riscaldamento: la prima chiamata e' piu' lenta
            model(torch.zeros(1, 3, TILE, TILE, device=device))
        return model, float(ck.get("thr", 0.30)) if is_dict else 0.30, None
    except Exception as e:
        return None, 0.30, f"Errore nel caricamento di MS-YOLOv8-Seg: {e}"
 
 
@st.cache_resource(show_spinner="Caricamento di YOLOv8s-seg...")
def load_yolo_model(path, device):
    if not os.path.exists(path):
        return None, f"File dei pesi non trovato: {path}"
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        model.predict(np.zeros((TILE, TILE, 3), np.uint8), imgsz=TILE, device=0 if device == "cuda" else "cpu", verbose=False)
        return model, None
    except Exception as e:
        return None, f"Errore nel caricamento di YOLOv8s-seg: {e}"
 
 
# =====================================================================
# 3. PIPELINE: TESSERE -> INFERENZA -> RICOMPOSIZIONE -> PULIZIA -> OVERLAY
# =====================================================================
def tile_frame(frame_bgr, max_side):
    """Ridimensiona senza deformare (lato lungo <= max_side, mai ingrandendo) e divide in tessere TILExTILE."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    img = frame_bgr if scale >= 1.0 else cv2.resize(
        frame_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    hh, ww = img.shape[:2]
    rows, cols = -(-hh // TILE), -(-ww // TILE)
    padded = cv2.copyMakeBorder(img, 0, rows * TILE - hh, 0, cols * TILE - ww, cv2.BORDER_REPLICATE)
    tiles = padded.reshape(rows, TILE, cols, TILE, 3).swapaxes(1, 2).reshape(rows * cols, TILE, TILE, 3)
    return img, tiles, rows, cols
 
 
def stitch(tile_maps, rows, cols, hh, ww):
    """Inverso di tile_frame per mappe (N, TILE, TILE): ricompone e ritaglia alla dimensione utile."""
    full = tile_maps.reshape(rows, cols, TILE, TILE).swapaxes(1, 2).reshape(rows * TILE, cols * TILE)
    return full[:hh, :ww]
 
 
def predict_ms(model, tiles_bgr, device, batch=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(tiles_bgr), batch):
            rgb = np.ascontiguousarray(tiles_bgr[i:i + batch][..., ::-1])
            x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255).to(device)
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)                                   # (N, TILE, TILE)
 
 
def predict_yolo(model, tiles_bgr, device, conf):
    results = model.predict(list(tiles_bgr), imgsz=TILE, conf=conf, retina_masks=True,
                            device=0 if device == "cuda" else "cpu", verbose=False)
    maps = np.zeros((len(tiles_bgr), TILE, TILE), np.uint8)
    best = 0.0
    for i, r in enumerate(results):
        if r.masks is not None and len(r.masks):
            m = (r.masks.data.cpu().numpy() > 0.5).any(axis=0)
            if m.shape != (TILE, TILE):
                m = cv2.resize(m.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST) > 0
            maps[i] = m
            best = max(best, float(r.boxes.conf.max().item()))
    return maps, best
 
 
def clean_mask(mask, min_px):
    """Elimina le componenti connesse con meno di min_px pixel; restituisce maschera e riquadri (x, y, w, h, area)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(min_px, 1)
    boxes = [(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
              int(stats[i, cv2.CC_STAT_HEIGHT]), int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n) if keep[i]]
    return keep[lab].astype(np.uint8), boxes
 
 
def draw_overlay(img_bgr, mask, boxes, color, text):
    vis = img_bgr.copy()
    if mask.any():
        fill = vis.copy()
        fill[mask == 1] = color
        vis = cv2.addWeighted(fill, 0.45, vis, 0.55, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        for x, y, w, h, _a in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
 
 
def estimate_area_cm2(damage_pct, w_px, h_px, width_cm):
    """Stima valida solo per superficie piana ripresa frontalmente e larghezza reale nota."""
    if width_cm <= 0:
        return None
    return damage_pct / 100.0 * width_cm * (width_cm * h_px / w_px)
 
 
def analyze_frame(frame_bgr, profile, model, device, max_side, threshold, min_area_frac):
    """Pipeline completa. threshold: confidenza (YOLO) o probabilita' (MS); min_area_frac: frazione di una tessera."""
    t0 = time.perf_counter()
    img, tiles, rows, cols = tile_frame(frame_bgr, max_side)
    hh, ww = img.shape[:2]
    t1 = time.perf_counter()
    if profile["kind"] == "ms":
        probs = stitch(predict_ms(model, tiles, device), rows, cols, hh, ww)
        raw, score = probs >= threshold, float(probs.max())
    else:
        maps, score = predict_yolo(model, tiles, device, threshold)
        raw = stitch(maps, rows, cols, hh, ww) > 0
    t2 = time.perf_counter()
    mask, boxes = clean_mask(raw, int(round(min_area_frac * TILE * TILE)))
    damage_pct = 100.0 * float(mask.sum()) / (hh * ww)
    vis = draw_overlay(img, mask, boxes, profile["colore"], f"Area sospetta {damage_pct:.2f}%  |  {len(boxes)} regioni")
    t3 = time.perf_counter()
    return dict(vis=vis, damage_pct=damage_pct, n_regions=len(boxes), score=score, n_tiles=len(tiles), size=(ww, hh),
                lat_net_ms=(t2 - t1) * 1000, lat_total_ms=(t3 - t0) * 1000)
 
 
# =====================================================================
# 4. INTERFACCIA
# =====================================================================
def show_image(target, img_bgr, caption=None):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        target.image(rgb, caption=caption, use_container_width=True)
    except TypeError:                                              # versioni di Streamlit piu' vecchie
        target.image(rgb, caption=caption, use_column_width=True)
 
 
def update_dashboard(res, ph, alert_ph, alert_pct, width_cm):
    alarm = res["damage_pct"] > alert_pct and res["n_regions"] > 0
    ph["lat"].metric("Latenza totale", f"{res['lat_total_ms']:.0f} ms")
    ph["fps"].metric("Fotogrammi al secondo", f"{1000.0 / max(res['lat_total_ms'], 1e-6):.1f}")
    ph["stato"].metric("Stato", "Area sospetta" if alarm else "Nessuna area sospetta")
    ph["area"].metric("Area sospetta", f"{res['damage_pct']:.2f}% dell'immagine")
    ph["score"].metric("Punteggio massimo", f"{res['score'] * 100:.0f}%")
    ph["info"].caption(f"Rete: {res['lat_net_ms']:.0f} ms · {res['n_tiles']} tessere · risoluzione di lavoro "
                       f"{res['size'][0]}×{res['size'][1]} px · il punteggio non e' una probabilita' calibrata")
    if alarm:
        cm2 = estimate_area_cm2(res["damage_pct"], res["size"][0], res["size"][1], width_cm)
        extra = f" (circa {cm2:.0f} cm², stima su superficie piana ripresa frontalmente)" if cm2 else ""
        alert_ph.error(f"🚨 Area sospetta pari al {res['damage_pct']:.2f}% dell'immagine{extra}. Verificare sul posto.")
    else:
        alert_ph.success("✅ Nessuna area sospetta oltre la soglia impostata. Non esclude difetti non rilevati.")
 
 
def main():
    st.title("🏗️ Ispezione di difetti sul calcestruzzo")
    st.caption("Spalling e corrosione · strumento di supporto: segnala aree sospette, la valutazione resta al tecnico")
 
    st.sidebar.header("⚙️ Configurazione")
    choice = st.sidebar.radio("Modello", [PROFILES["standard"]["titolo"], PROFILES["efficient"]["titolo"], CONFRONTO])
    if choice == CONFRONTO:
        keys = ["standard", "efficient"]
    else:
        keys = ["standard"] if choice == PROFILES["standard"]["titolo"] else ["efficient"]
 
    dev_choice = st.sidebar.radio("Dispositivo di calcolo", ["GPU (CUDA)", "CPU"])
    device = "cuda" if dev_choice.startswith("GPU") and torch.cuda.is_available() else "cpu"
    if dev_choice.startswith("GPU") and device == "cpu":
        st.sidebar.warning("GPU non disponibile: uso la CPU.")
 
    paths = {}
    with st.sidebar.expander("📂 File dei pesi"):
        for k in ("standard", "efficient"):
            paths[k] = st.text_input(f"Pesi: {PROFILES[k]['titolo']}", PROFILES[k]["percorso"])
 
    models, thr0, errors = {}, {}, []
    for k in keys:
        if PROFILES[k]["kind"] == "ms":
            models[k], thr0[k], err = load_ms_model(paths[k], device)
        else:
            models[k], err = load_yolo_model(paths[k], device)
            thr0[k] = PROFILES[k]["soglia_default"]
        if err:
            errors.append(err)
    for err in errors:
        st.error(err)
    if errors:
        return
 
    st.sidebar.subheader("🎛️ Parametri")
    max_side = st.sidebar.select_slider("Risoluzione di lavoro (lato lungo, px)", options=[256, 512, 768, 1024, 1536], value=1024,
                                        help="I modelli sono addestrati su tessere 256×256 ricavate da immagini da 1024 px: "
                                             "il fotogramma viene ridimensionato a questo lato lungo e diviso in tessere.")
    thr, area_min = {}, {}
    for k in keys:
        p = PROFILES[k]
        suffix = f" ({'standard' if k == 'standard' else 'efficiente'})" if len(keys) > 1 else ""
        thr[k] = st.sidebar.slider(p["soglia_label"] + suffix, 0.05, 0.95, float(np.clip(round(thr0[k], 2), 0.05, 0.95)), 0.05)
        area_min[k] = st.sidebar.slider("Area minima di una regione (% di una tessera)" + suffix, 0.0, 5.0,
                                        float(p["area_min_default"]), 0.25,
                                        help="Elimina le regioni piu' piccole. Per il modello efficiente riduce i falsi allarmi minuti.") / 100.0
    alert_pct = st.sidebar.slider("Segnala se l'area sospetta supera (% dell'immagine)", 0.0, 10.0, 1.0, 0.25)
    width_cm = st.sidebar.number_input("Larghezza reale inquadrata (cm), opzionale", min_value=0.0, value=0.0, step=10.0,
                                       help="Se indicata, stima l'area in cm². Valida solo per superficie piana ripresa frontalmente.")
 
    with st.sidebar.expander("ℹ️ Schede dei modelli"):
        for k in ("standard", "efficient"):
            st.markdown(f"**{PROFILES[k]['titolo']}**")
            st.markdown("\n".join(f"- {a}: {b}" for a, b in SCHEDA[k]))
            st.caption(PROFILES[k]["nota"])
        st.caption("Accuratezza: test S2DS, 374 tessere, 3 seed. Latenze: RTX 5050 Laptop e CPU a un thread. "
                   "Prestazioni su dispositivi mobili o embedded: non misurate.")
 
    names = {"lat": "Latenza totale", "fps": "Fotogrammi al secondo", "stato": "Stato", "area": "Area sospetta", "score": "Punteggio massimo"}
    ph = {}
    for col, (key, label) in zip(st.columns(5), names.items()):
        ph[key] = col.empty()
        ph[key].metric(label, "–")
    ph["info"] = st.empty()
    alert_ph = st.empty()
    st.divider()
 
    source = st.radio("Sorgente", ["Scatta una foto", "Webcam (tempo reale)", "Carica immagine o video"], horizontal=True)
    compare = choice == CONFRONTO
 
    def run(frame, k):
        return analyze_frame(frame, PROFILES[k], models[k], device, max_side, thr[k], area_min[k])
 
    def process_image(frame):
        if compare:
            for col, k in zip(st.columns(2), keys):
                res = run(frame, k)
                with col:
                    st.subheader(PROFILES[k]["titolo"])
                    show_image(st, res["vis"])
                    st.markdown(f"Latenza totale **{res['lat_total_ms']:.0f} ms** (rete {res['lat_net_ms']:.0f} ms) · "
                                f"area sospetta **{res['damage_pct']:.2f}%** · regioni **{res['n_regions']}** · "
                                f"punteggio massimo {res['score'] * 100:.0f}%")
        else:
            res = run(frame, keys[0])
            update_dashboard(res, ph, alert_ph, alert_pct, width_cm)
            show_image(st, res["vis"], "Maschera, contorni e regioni sospette")
 
    if compare and source == "Webcam (tempo reale)":
        st.warning("Il confronto affiancato è disponibile solo per immagini: scegli un singolo modello per la webcam.")
        return
 
    if source == "Scatta una foto":
        shot = st.camera_input("Inquadra la superficie in calcestruzzo")
        if shot is not None:
            frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                process_image(frame)
 
    elif source == "Webcam (tempo reale)":
        if st.checkbox("Avvia il flusso video"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Impossibile accedere alla webcam.")
            else:
                img_ph = st.empty()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    res = run(frame, keys[0])
                    update_dashboard(res, ph, alert_ph, alert_pct, width_cm)
                    show_image(img_ph, res["vis"])
                cap.release()
 
    else:
        up = st.file_uploader("Immagine o video", type=["jpg", "jpeg", "png", "mp4", "avi"])
        if up is not None:
            ext = Path(up.name).suffix.lower()
            if ext in (".jpg", ".jpeg", ".png"):
                frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Immagine non leggibile.")
                else:
                    process_image(frame)
            elif compare:
                st.warning("Il confronto affiancato è disponibile solo per immagini.")
            else:
                skip = st.slider("Elabora un fotogramma ogni N", 1, 30, 5)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tfile.write(up.read())
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                img_ph = st.empty()
                idx = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % skip == 0:
                        res = run(frame, keys[0])
                        update_dashboard(res, ph, alert_ph, alert_pct, width_cm)
                        show_image(img_ph, res["vis"])
                    idx += 1
                cap.release()
                os.unlink(tfile.name)
 
 
if __name__ == "__main__":
    main()
