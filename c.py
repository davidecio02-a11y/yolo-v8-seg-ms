# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:02:20 2026

@author: david
"""

import re
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
# ------------------------- CONFIGURAZIONE (modifica qui) -------------------------
WORK = Path(r"D:\yolo_seg\final_runs")
SEEDS = None                  # None = tutti i seed trovati; oppure [0] / [0, 1, 2]
N_BOOT = 2000
NOISE_IOU_PT = 2.5            # IoU sui difetti (micro)
NOISE_IMG_PT = 5.0            # IoU medio per immagine sui difetti
NOISE_SMALL_PT = 10.0         # IoU sui difetti piccoli (area <= 10% della patch)
NOISE_FP_PT = 10.0            # % di patch pulite con >1% di area predetta
 
CONFIG_ORDER = ["yolo_pre", "yolo_scratch", "ms_base", "ms_ghost", "ms_rep", "ms_simam", "ms_dsiou", "ms_full",
                "ms_base_scratch", "ms_full_scratch"]
ABLATION = ["ms_ghost", "ms_rep", "ms_simam", "ms_dsiou", "ms_full"]
LABEL = {"ms_ghost": "blocchi Ghost", "ms_rep": "RepBlock", "ms_simam": "SimAM", "ms_dsiou": "pesi sul bordo (DSIoU)",
         "ms_full": "tutti i moduli insieme"}
PAIRS = [("ms_full", "yolo_pre"), ("ms_base", "yolo_pre"), ("ms_full", "ms_base"),
         ("ms_full_scratch", "yolo_scratch"), ("ms_base_scratch", "yolo_scratch"), ("ms_full_scratch", "ms_base_scratch"),
         ("ms_full", "ms_full_scratch"), ("ms_base", "ms_base_scratch"), ("yolo_pre", "yolo_scratch")]
PRE_PAIRS = [("yolo_pre", "yolo_scratch"), ("ms_base", "ms_base_scratch"), ("ms_full", "ms_full_scratch")]
ARCH_OF = {"yolo_scratch": "yolo_pre", "ms_dsiou": "ms_base", "ms_base_scratch": "ms_base", "ms_full_scratch": "ms_full"}
BINS = [0, 0.02, 0.10, 0.30, 1.0]
BIN_LABELS = ["<=2%", "2-10%", "10-30%", ">30%"]
SMALL_MAX = 0.10
REQUIRED = {"name", "tp", "fp", "fn", "ftp", "ffp", "ffn", "npix", "has_gt", "area_frac", "farea_frac", "thr", "min_frac"}
KEYS = {"iou": 1, "fp": 2, "img": 3, "small": 4}      # posizioni nel vettore delle statistiche del bootstrap
 
 
class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout
 
    def write(self, s):
        self.out.write(s)
        self.f.write(s)
 
    def flush(self):
        self.out.flush()
        self.f.flush()
 
 
# ------------------------- DATI E METRICHE -------------------------
def load_all(work, seeds_filter):
    runs = {}
    for p in sorted((work / "csv").glob("*_s*.csv")):
        m = re.match(r"(.+)_s(\d+)\.csv$", p.name)
        if not m:
            continue
        cfg, seed = m.group(1), int(m.group(2))
        if seeds_filter is not None and seed not in seeds_filter:
            continue
        df = pd.read_csv(p)
        if not REQUIRED.issubset(df.columns):
            print(f"[!] {p.name}: colonne mancanti ({sorted(REQUIRED - set(df.columns))}), file ignorato")
            continue
        runs[(cfg, seed)] = df
    return runs
 
 
def micro(tp, fp, fn):
    e = 1e-9
    return dict(iou=tp / (tp + fp + fn + e), precision=tp / (tp + fp + e), recall=tp / (tp + fn + e))
 
 
def auc(pos_scores, neg_scores):
    n1, n0 = len(pos_scores), len(neg_scores)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.concat([pd.Series(list(pos_scores)), pd.Series(list(neg_scores))], ignore_index=True).rank(method="average")
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
 
 
def metrics(df):
    pos, neg = df[df.has_gt].copy(), df[~df.has_gt]
    small = pos[(pos.tp + pos.fn) / pos.npix <= SMALL_MAX]
    out = dict(iou=micro(df.tp.sum(), df.fp.sum(), df.fn.sum())["iou"],
               pos_iou=micro(pos.tp.sum(), pos.fp.sum(), pos.fn.sum())["iou"],
               pos_macro=float((pos.tp / (pos.tp + pos.fp + pos.fn)).mean()),
               small_iou=micro(small.tp.sum(), small.fp.sum(), small.fn.sum())["iou"] if len(small) else np.nan,
               f_pos_iou=micro(pos.ftp.sum(), pos.ffp.sum(), pos.ffn.sum())["iou"],
               thr=float(df.thr.iloc[0]), min_frac=float(df.min_frac.iloc[0]))
    if len(neg):
        a = (neg.fp / neg.npix)
        fpi = a[neg.fp > 0]
        out.update(fp1=float((neg.area_frac > 0.01).mean()), fpany=float((neg.fp > 0).mean()),
                   f_fp1=float((neg.farea_frac > 0.01).mean()),
                   fp_med=float(fpi.median()) if len(fpi) else np.nan,
                   fp_small=float((fpi < 0.01).mean()) if len(fpi) else np.nan,
                   auc=auc(pos.area_frac, neg.area_frac),
                   auc_maxp=auc(pos.maxp, neg.maxp) if "maxp" in df else np.nan,
                   bacc=0.5 * float((pos.area_frac > 0.01).mean() + (neg.area_frac <= 0.01).mean()))
    return out
 
 
def per_config(runs):
    return pd.DataFrame([dict(config=c, seed=s, **metrics(d)) for (c, s), d in runs.items()])
 
 
def bin_table(runs, cfgs):
    out = []
    for c in cfgs:
        per_seed = []
        for (cc, s), d in runs.items():
            if cc != c:
                continue
            pos = d[d.has_gt].copy()
            pos["bin"] = pd.cut((pos.tp + pos.fn) / pos.npix, BINS, labels=BIN_LABELS)
            per_seed.append({b: micro(g.tp.sum(), g.fp.sum(), g.fn.sum())["iou"] for b, g in pos.groupby("bin", observed=True)})
        if per_seed:
            m = pd.DataFrame(per_seed).mean()
            out.append({"config": c, **{b: f"{m[b] * 100:.1f}" for b in m.index}})
    return pd.DataFrame(out)
 
 
def pooled_boot(runs, a, b, n_boot):
    """Bootstrap appaiato per immagine sorgente sui conteggi sommati tra i seed disponibili per entrambe.
    Statistiche (punti): [IoU tutte, IoU difetti, FP>1%, IoU medio per immagine (difetti), IoU difetti piccoli]."""
    seeds = sorted({s for (c, s) in runs if c == a} & {s for (c, s) in runs if c == b})
    if not seeds:
        return None
 
    def stack(name):
        out = None
        for s in seeds:
            d = runs[(name, s)].set_index("name").sort_index()
            pos = d.has_gt
            sm = pos & ((d.tp + d.fn) / d.npix <= SMALL_MAX)
            cur = pd.DataFrame({"tp": d.tp, "fp": d.fp, "fn": d.fn, "ptp": d.tp.where(pos, 0), "pfp": d.fp.where(pos, 0),
                                "pfn": d.fn.where(pos, 0), "negn": (~pos).astype(int),
                                "negfp": ((~pos) & (d.area_frac > 0.01)).astype(int),
                                "isum": (d.tp / (d.tp + d.fp + d.fn).replace(0, np.nan)).where(pos, 0).fillna(0),
                                "npos": pos.astype(int), "stp": d.tp.where(sm, 0), "sfp": d.fp.where(sm, 0),
                                "sfn": d.fn.where(sm, 0)})
            out = cur if out is None else out.add(cur, fill_value=0)
        return out
 
    A, B = stack(a), stack(b)
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    grp = pd.Series(common).map(lambda n: str(n).split("_")[0]).to_numpy()
    cols = ["tp", "fp", "fn", "ptp", "pfp", "pfn", "negn", "negfp", "isum", "npos", "stp", "sfp", "sfn"]
    G = np.hstack([A[cols].groupby(grp).sum().to_numpy(float), B[cols].groupby(grp).sum().to_numpy(float)])
    k = len(cols)
    e = 1e-9
    iou = lambda t, f, n: t / (t + f + n + e)
 
    def stats(s):
        x, y = s[:k], s[k:]
        return np.array([iou(x[0], x[1], x[2]) - iou(y[0], y[1], y[2]),
                         iou(x[3], x[4], x[5]) - iou(y[3], y[4], y[5]),
                         (x[7] - y[7]) / x[6] if x[6] > 0 else np.nan,
                         x[8] / max(x[9], 1) - y[8] / max(y[9], 1),
                         iou(x[10], x[11], x[12]) - iou(y[10], y[11], y[12])]) * 100
 
    obs = stats(G.sum(0))
    rng = np.random.default_rng(0)
    boots = np.array([stats(G[rng.integers(0, len(G), len(G))].sum(0)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return dict(n_groups=len(G), n_seeds=len(seeds), obs=obs, lo=lo, hi=hi)
 
 
def level(obs, lo, hi, noise):
    if np.isnan(obs):
        return "-"
    excl = lo > 0 or hi < 0
    big = abs(obs) > noise
    return "SEGNALE" if (excl and big) else ("debole" if (excl or big) else "nessuno")
 
 
def noise_levels(cm):
    base = dict(iou=NOISE_IOU_PT, img=NOISE_IMG_PT, small=NOISE_SMALL_PT, fp=NOISE_FP_PT)
    multi = cm.groupby("config").filter(lambda g: len(g) >= 2)
    if multi.empty:
        return base, "assunta (1 seed): soglie prudenziali tratte dal confronto con run precedenti"
    sd = lambda col: multi.groupby("config")[col].std().median() * 100
    return dict(iou=max(1.0, 2 * sd("pos_iou")), img=max(2.0, 2 * sd("pos_macro")), small=max(4.0, 2 * sd("small_iou")),
                fp=max(3.0, 2 * sd("fp1"))), "stimata dalla std tra seed (2 x mediana)"
 
 
def cell(r, which, noise):
    i = KEYS[which]
    if np.isnan(r["obs"][i]):
        return "-"
    return f"{r['obs'][i]:+.1f} [{r['lo'][i]:+.1f},{r['hi'][i]:+.1f}] {level(r['obs'][i], r['lo'][i], r['hi'][i], noise[which])}"
 
 
def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}"
 
 
def main():
    out_dir = WORK / "analisi"
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(out_dir / "analisi.txt")
    sys.stdout = tee
    try:
        _run(out_dir)
    finally:
        sys.stdout = tee.out
        tee.f.close()
    print(f"\n[salvato in {out_dir}]")
 
 
def _run(out_dir):
    runs = load_all(WORK, SEEDS)
    if not runs:
        print(f"Nessun CSV trovato in {WORK / 'csv'}")
        return
    cm = per_config(runs)
    seeds = sorted(cm.seed.unique().tolist())
    cfgs = [c for c in CONFIG_ORDER if c in set(cm.config)] + [c for c in sorted(set(cm.config)) if c not in CONFIG_ORDER]
    noise, how = noise_levels(cm)
    have = set(cm.config)
 
    ex = next(iter(runs.values()))
    n_src = ex.name.map(lambda n: str(n).split("_")[0]).nunique()
    print("=" * 110)
    print("ANALISI DEI RISULTATI")
    print(f"Seed: {seeds} | configurazioni: {len(cfgs)} | test: {len(ex)} patch ({int(ex.has_gt.sum())} con difetto, "
          f"{int((~ex.has_gt).sum())} senza) da {n_src} immagini sorgente")
    print(f"Rumore ({how}): IoU difetti {noise['iou']:.1f} pt | IoU per immagine {noise['img']:.1f} pt | "
          f"IoU difetti piccoli {noise['small']:.1f} pt | FP>1% {noise['fp']:.1f} pt")
    print("Regole: SEGNALE = |Δ| > rumore E intervallo bootstrap che esclude lo zero | debole = una sola delle due | nessuno = nessuna")
    print("=" * 110)
 
    # ---- 1. tabella principale ----
    t1 = []
    for c in cfgs:
        d = cm[cm.config == c]
        row = {"config": c, "n": len(d)}
        for k, h in (("iou", "IoU tutte"), ("pos_iou", "IoU difetti"), ("pos_macro", "IoU/img"), ("small_iou", "IoU piccoli"),
                     ("fp1", "FP>1%"), ("fpany", "FP>=1px"), ("auc", "AUC area"), ("auc_maxp", "AUC maxp"), ("bacc", "AccBil"),
                     ("f_fp1", "FP>1% filtro")):
            if k in d:
                row[h] = pct(d[k].mean()) + (f"±{d[k].std() * 100:.1f}" if len(d) > 1 else "")
        row["thr"] = f"{d.thr.mean():.2f}"
        row["filtro"] = f"{d.min_frac.mean() * 100:.2f}%"
        t1.append(row)
    t1 = pd.DataFrame(t1)
    print("\n1) METRICHE PER CONFIGURAZIONE (%). IoU tutte = tutte le patch | IoU difetti = micro sui soli difetti | IoU/img = media per immagine")
    print("   IoU piccoli = difetti con area <= 10% della patch | FP>1% = patch pulite con >1% di area predetta | AUC/AccBil = 'c'e' un difetto?' per patch")
    print(t1.to_string(index=False))
    t1.to_csv(out_dir / "tabella_metriche.csv", index=False)
 
    # ---- 2. ablation ----
    print("\n2) ABLATION: un modulo alla volta rispetto a ms_base (Δ in punti [IC95%] livello; per FP negativo = meglio)")
    ab_rows, ab_res = [], {}
    if "ms_base" in have:
        for n in ABLATION:
            r = pooled_boot(runs, n, "ms_base", N_BOOT)
            if r:
                ab_res[n] = r
                ab_rows.append({"modulo": f"{n} ({LABEL[n]})", "Δ IoU difetti": cell(r, "iou", noise), "Δ IoU/img": cell(r, "img", noise),
                                "Δ IoU piccoli": cell(r, "small", noise), "Δ FP>1%": cell(r, "fp", noise)})
        print(pd.DataFrame(ab_rows).to_string(index=False) if ab_rows else "   (nessuna configurazione dell'ablation disponibile)")
        pd.DataFrame(ab_rows).to_csv(out_dir / "ablation.csv", index=False)
    else:
        print("   ms_base non disponibile")
 
    # ---- 3. confronti diretti ----
    print("\n3) CONFRONTI DIRETTI (Δ = A - B)")
    pr, pr_res = [], {}
    for a, b in PAIRS:
        if a in have and b in have:
            r = pooled_boot(runs, a, b, N_BOOT)
            if r:
                pr_res[(a, b)] = r
                pr.append({"A": a, "B": b, "Δ IoU difetti": cell(r, "iou", noise), "Δ IoU/img": cell(r, "img", noise),
                           "Δ IoU piccoli": cell(r, "small", noise), "Δ FP>1%": cell(r, "fp", noise)})
    print(pd.DataFrame(pr).to_string(index=False) if pr else "   (nessuna coppia disponibile)")
    pd.DataFrame(pr).to_csv(out_dir / "confronti.csv", index=False)
 
    # ---- 4-6. dimensione, stile FP, costo ----
    print("\n4) IoU SUI DIFETTI PER DIMENSIONE (area del difetto / area della patch, %)")
    print(bin_table(runs, cfgs).to_string(index=False))
 
    print("\n5) STILE DEI FALSI POSITIVI sulle patch senza difetto")
    fp_rows = []
    for c in cfgs:
        d = cm[cm.config == c]
        if "fpany" in d:
            fp_rows.append({"config": c, "FP>=1px": pct(d.fpany.mean()), "FP>1%": pct(d.fp1.mean()),
                            "area mediana quando c'e' FP": pct(d.fp_med.mean(), 1) + "%",
                            "FP piccoli (<1%) tra i FP": pct(d.fp_small.mean())})
    print(pd.DataFrame(fp_rows).to_string(index=False))
 
    eff = pd.read_csv(WORK / "efficiency.csv") if (WORK / "efficiency.csv").exists() else None
    if eff is not None:
        print("\n6) COSTO (forma deploy; le varianti con la stessa architettura condividono i valori)")
        rows = []
        for c in cfgs:
            e = eff[eff.config == ARCH_OF.get(c, c)]
            if len(e):
                e = e.iloc[0]
                rows.append({"config": c, "param (M)": f"{e.params_M:.2f}", "GFLOPs": f"{e.gflops:.2f}", "GPU b=1 (ms)": f"{e.lat_gpu1_ms:.1f}",
                             "GPU b=16 (img/s)": f"{e.gpu16_imgs:.0f}", "CPU 1 thread (ms)": f"{e.lat_cpu1_ms:.0f}"})
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("\n6) COSTO: efficiency.csv non trovato (esegui: python final_project.py --stage efficiency, circa 5 minuti)")
 
    # ---- 7. conclusioni ----
    print("\n" + "=" * 110)
    print("CONCLUSIONI (a regole fisse; con un solo seed sono indicative e da confermare)")
    print("=" * 110)
 
    def phrase(a, b, r, which, higher_better=True):
        i = KEYS[which]
        obs, lo, hi = r["obs"][i], r["lo"][i], r["hi"][i]
        if np.isnan(obs):
            return "n/d"
        lv = level(obs, lo, hi, noise[which])
        if lv == "nessuno":
            return f"nessuna differenza dimostrabile ({obs:+.1f} pt, IC [{lo:+.1f},{hi:+.1f}])"
        who = a if (obs > 0) == higher_better else b
        return f"{'' if lv == 'SEGNALE' else 'indizio debole: '}{who} meglio ({obs:+.1f} pt, IC [{lo:+.1f},{hi:+.1f}])"
 
    mean_iou = cm.groupby("config").pos_iou.mean() * 100
    best = mean_iou.idxmax()
    tie = [c for c in mean_iou.index if mean_iou[best] - mean_iou[c] <= noise["iou"] and c != best]
    print(f"\n[Segmentazione dei difetti] Valore piu' alto: {best} ({mean_iou[best]:.1f}%). "
          + (f"Entro il rumore ({noise['iou']:.1f} pt): {', '.join(tie)}. Tra queste non e' dimostrata alcuna differenza." if tie
             else "Le altre sono sotto di oltre il rumore."))
 
    if ab_res:
        print("\n[Ablation: cosa aggiunge ciascun modulo a ms_base]")
        for n, r in ab_res.items():
            print(f"   - {LABEL[n]} ({n}): difetti: {phrase(n, 'ms_base', r, 'iou')}; piccoli: {phrase(n, 'ms_base', r, 'small')}; "
                  f"falsi positivi: {phrase(n, 'ms_base', r, 'fp', False)}")
        sig = lambda n, w, sgn: level(ab_res[n]["obs"][KEYS[w]], ab_res[n]["lo"][KEYS[w]], ab_res[n]["hi"][KEYS[w]], noise[w]) == "SEGNALE" \
            and ab_res[n]["obs"][KEYS[w]] * sgn > 0
        helpful = [LABEL[n] for n in ab_res if sig(n, "iou", 1)]
        harmful = [LABEL[n] for n in ab_res if sig(n, "iou", -1)]
        weak = [LABEL[n] for n, r in ab_res.items() if level(r["obs"][1], r["lo"][1], r["hi"][1], noise["iou"]) == "debole" and r["obs"][1] > 0]
        print("   Sintesi: " + (f"aiutano sui difetti (segnale): {', '.join(helpful)}. " if helpful else "nessun modulo migliora i difetti con un segnale netto. ")
              + (f"Indizi deboli a favore: {', '.join(weak)}. " if weak else "") + (f"Peggiorano (segnale): {', '.join(harmful)}." if harmful else ""))
        grp_g = [c for c in ("ms_ghost", "ms_full") if c in have]
        grp_n = [c for c in ("ms_base", "ms_rep", "ms_simam", "ms_dsiou") if c in have]
        if len(grp_g) == 2 and len(grp_n) >= 3:
            mg, mn = mean_iou[grp_g].mean(), mean_iou[grp_n].mean()
            print(f"   Pattern (non e' un test): le configurazioni con blocchi Ghost ({', '.join(grp_g)}) hanno IoU difetti medio {mg:.1f}% contro "
                  f"{mn:.1f}% delle altre con pesi pre-addestrati ({', '.join(grp_n)}): {mg - mn:+.1f} pt. "
                  "I Ghost sostituiscono blocchi pre-addestrati: puo' essere un costo del trasferimento parziale, da confermare con altri seed.")
 
    print("\n[Modello MS contro YOLO, entrambi con pesi COCO]")
    for a, b, lab in (("ms_full", "yolo_pre", "ms_full vs yolo_pre"),
                      ("ms_base", "yolo_pre", "ms_base (pipeline semantica senza moduli) vs yolo_pre")):
        if (a, b) in pr_res:
            r = pr_res[(a, b)]
            print(f"   - {lab}:\n       difetti (micro): {phrase(a, b, r, 'iou')}\n       difetti (media per immagine): {phrase(a, b, r, 'img')}"
                  f"\n       difetti piccoli: {phrase(a, b, r, 'small')}\n       falsi positivi: {phrase(a, b, r, 'fp', False)}")
    if ("ms_base", "yolo_pre") in pr_res and ("ms_full", "yolo_pre") in pr_res:
        print("     (se ms_base e ms_full danno lo stesso esito contro YOLO, il vantaggio viene dalla pipeline semantica, non dai moduli)")
 
    pre_pairs = [p for p in PRE_PAIRS if p in pr_res]
    if pre_pairs:
        print("\n[Effetto del transfer learning: con pesi COCO contro da zero]")
        for a, b in pre_pairs:
            print(f"   - {a} vs {b}: difetti: {phrase(a, b, pr_res[(a, b)], 'iou')}; falsi positivi: {phrase(a, b, pr_res[(a, b)], 'fp', False)}")
        n_up = sum(pr_res[p]["obs"][1] > 0 for p in pre_pairs)
        n_fp = sum(pr_res[p]["obs"][2] < 0 for p in pre_pairs)
        print(f"   Coerenza: i pesi pre-addestrati danno IoU difetti piu' alto in {n_up}/{len(pre_pairs)} coppie e meno falsi positivi in {n_fp}/{len(pre_pairs)}.")
    else:
        print("\n[Effetto del transfer learning] non valutabile: mancano le run da zero")
 
    if "fpany" in cm:
        print("\n[Come sbagliano sulle patch pulite]")
        for c in [x for x in ("yolo_pre", "ms_base", "ms_full") if x in have]:
            d = cm[cm.config == c]
            any_, med, small = d.fpany.mean(), d.fp_med.mean(), d.fp_small.mean()
            style = ("rari ma di grandi dimensioni" if any_ < 0.12 and med >= 0.15 else
                     "frequenti e piccoli" if any_ >= 0.15 and med < 0.01 else "frequenti e di dimensione moderata" if any_ >= 0.15 else "misti")
            print(f"   - {c}: {any_ * 100:.0f}% delle patch pulite ha almeno un pixel predetto; quando succede l'area mediana e' "
                  f"{med * 100:.1f}% della patch ({small * 100:.0f}% dei casi sotto l'1%) -> {style}")
        for c in [x for x in ("yolo_pre", "ms_full") if x in have]:
            d = cm[cm.config == c]
            print(f"   - filtro delle componenti piccole su {c} (scelto sul validation): FP>1% da {d.fp1.mean() * 100:.1f}% a "
                  f"{d.f_fp1.mean() * 100:.1f}%, IoU difetti da {d.pos_iou.mean() * 100:.1f}% a {d.f_pos_iou.mean() * 100:.1f}%")
        low = cm[cm.thr <= 0.15]
        if len(low):
            print(f"   - Attenzione: soglia scelta sul validation <= 0.15 per {', '.join(sorted(set(low.config)))}: e' agli estremi della griglia, "
                  "segno di probabilita' poco calibrate o di un validation rumoroso (36 immagini sorgente).")
 
    if "auc" in cm:
        a, am = cm.groupby("config").auc.mean(), cm.groupby("config").auc_maxp.mean()
        print(f"\n[Decisione a livello di patch ('c'e' un difetto?')] AUC (area predetta): massima {a.idxmax()} ({a.max():.3f}), minima {a.idxmin()} ({a.min():.3f}). "
              f"AUC (confidenza massima): massima {am.idxmax()} ({am.max():.3f}), minima {am.idxmin()} ({am.min():.3f}). "
              "L'AUC dell'area dipende dalla soglia scelta (un modello che manca i difetti piccoli ha area 0); se i due AUC discordano, fidati di meno del primo.")
 
    if eff is not None and "ms_full" in set(eff.config) and "yolo_pre" in set(eff.config):
        m, y = eff[eff.config == "ms_full"].iloc[0], eff[eff.config == "yolo_pre"].iloc[0]
        print(f"\n[Costo] ms_full usa {m.params_M / y.params_M:.2f}x i parametri e {m.gflops / y.gflops:.2f}x i FLOPs di yolo_pre; latenza GPU batch 1 "
              f"{m.lat_gpu1_ms:.1f} contro {y.lat_gpu1_ms:.1f} ms, CPU 1 thread {m.lat_cpu1_ms:.0f} contro {y.lat_cpu1_ms:.0f} ms. "
              "(YOLO include anche la testa di rilevamento, il modello MS no.)")
 
    print("\n[Limiti da dichiarare]")
    print(f"   - {len(seeds)} seed: " + ("la variabilita' del training non e' misurata; due run della stessa configurazione con seed diversi possono differire di diversi punti."
                                         if len(seeds) == 1 else "deviazione standard tra pochi seed: stima grossolana."))
    print(f"   - Test di {n_src} immagini sorgente, rapporto 1:1 tra patch con e senza difetto (in una facciata reale i negativi sono molti di piu').")
    print("   - Un solo dataset (S2DS) e, per l'efficienza, un solo hardware.")
    print("   - Soglia, filtro delle componenti piccole e checkpoint sono scelti solo sul validation.")
 
    # ---- grafici ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if ab_res:
            fig, ax = plt.subplots(figsize=(7, 4))
            names = list(ab_res)
            obs = [ab_res[n]["obs"][1] for n in names]
            err = [[ab_res[n]["obs"][1] - ab_res[n]["lo"][1] for n in names], [ab_res[n]["hi"][1] - ab_res[n]["obs"][1] for n in names]]
            ax.bar(names, obs, yerr=err, capsize=4, color="#4C78A8")
            ax.axhline(0, color="k", lw=0.8)
            ax.axhspan(-noise["iou"], noise["iou"], color="orange", alpha=0.15, label="zona di rumore assunta")
            ax.set_ylabel("Δ IoU difetti vs ms_base (pt, IC95%)")
            ax.legend()
            plt.xticks(rotation=20)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_ablation_ic.png", dpi=150)
            plt.close(fig)
        if "fp1" in cm:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            m = cm.groupby("config")[["pos_iou", "fp1"]].mean() * 100
            ax.scatter(m.fp1, m.pos_iou, s=50)
            for c, r in m.iterrows():
                ax.annotate(c, (r.fp1, r.pos_iou), fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax.set_xlabel("patch pulite con >1% di area predetta (%)  <- meglio")
            ax.set_ylabel("IoU sui difetti (%)  ^ meglio")
            ax.set_title("Compromesso tra segmentazione e falsi positivi")
            fig.tight_layout()
            fig.savefig(out_dir / "fig_compromesso.png", dpi=150)
            plt.close(fig)
    except ImportError:
        print("\n(matplotlib non disponibile: grafici saltati)")
 
 
if __name__ == "__main__":
    main()