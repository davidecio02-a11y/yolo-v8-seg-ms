# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:50:40 2026

@author: david
"""

import argparse
from pathlib import Path
 
from ultralytics import YOLO
 
DATA = r"D:\yolo_seg\spalling_corrosion_patches_v2_new\data.yaml"
PROJECT = r"D:\yolo_seg\runs"
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="default: corrosion_seg_neg_s<seed>")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=int, default=50)
    a, _ = ap.parse_known_args()          # tollera gli argomenti extra di Spyder
    name = a.name or f"corrosion_seg_neg_s{a.seed}"
 
    model = YOLO("yolov8s-seg.pt")       # pesi COCO standard (si scaricano da soli se mancano)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=256, batch=32, workers=4, device=0,
        seed=a.seed, name=name, project=PROJECT,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"      # percorso reale (non quello ipotizzato)
    print("\n[+] Training completato!")
    print(f"[+] Pesi migliori: {best}")
    print(f"[+] Confronto: python compare_models.py --yolo-model \"{best}\"")
 
 
if __name__ == "__main__":
    main()
 