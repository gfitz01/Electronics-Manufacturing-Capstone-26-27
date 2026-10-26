"""Train the part detector on the frames make_detector_data.py made.

    pip install ultralytics                   once
    python train_detector.py --quick          2-minute check that everything works
    python train_detector.py                  the real run, roughly 40-60 min on the 3060

What it trains
--------------
A YOLO detector (Ultralytics YOLOv8, the smallest 'n' size), starting from
weights already trained on everyday photos -- the same head start AlexNet got
from ImageNet. It learns one class, 'impeller', and to draw a box round each
part's rim. It runs on the GPU.

It stops early once the check frames stop improving for 15 epochs, the same
idea as stage one's early stopping.

What you get
------------
    detector\\impeller.pt       the trained detector, for multi_inspect.py
    detector\\detector.json     its scores on the check frames, and the settings
    results\\detector\\<time>\\   Ultralytics' own run folder: curves, sample
                                 predictions, confusion matrix

How to read the score
---------------------
    precision   of the boxes it drew, the share that were real parts
    recall      of the real parts, the share it found
    mAP50       one number for both, 0 to 1; above 0.95 is very good here

These are measured on fake frames built from validation photos. The real test
is your camera -- checkpoint two hooks this into the scanner.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="detect_data/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="starting weights; downloaded automatically the first time")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=832,
                    help="training picture size; bigger finds small parts better "
                         "but is slower")
    ap.add_argument("--batch", type=int, default=16,
                    help="lower to 8 if the GPU runs out of memory")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true",
                    help="1 epoch on 5%% of the frames, to check the setup works")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Ultralytics is not installed. Run:  pip install ultralytics")
    import torch

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"Missing {data}\nRun first:  python make_detector_data.py")
    meta_path = data.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    if torch.cuda.is_available():
        device = 0
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("No GPU found -- this will be very slow on the CPU.")

    epochs, fraction = (1, 0.05) if args.quick else (args.epochs, 1.0)
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    project = Path("results/detector").resolve()

    model = YOLO(args.model)
    model.train(data=str(data.resolve()), epochs=epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, patience=args.patience,
                workers=args.workers, fraction=fraction, seed=0,
                project=str(project), name=stamp, exist_ok=False, plots=True)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training finished but {best} is missing.")

    print("\nScoring the best epoch on the check frames ...")
    m = YOLO(str(best)).val(data=str(data.resolve()), imgsz=args.imgsz,
                            device=device, split="val", plots=False)
    scores = {"precision": round(float(m.box.mp), 4),
              "recall": round(float(m.box.mr), 4),
              "mAP50": round(float(m.box.map50), 4),
              "mAP50_95": round(float(m.box.map), 4)}

    print()
    print("=" * 60)
    print("  PART DETECTOR, on check frames built from validation photos")
    print("=" * 60)
    print(f"  precision   {scores['precision']:.3f}   boxes that were real parts")
    print(f"  recall      {scores['recall']:.3f}   parts it found")
    print(f"  mAP50       {scores['mAP50']:.3f}")
    print("=" * 60)

    if args.quick:
        print("\nSetup works. Now the real run:  python train_detector.py")
        return 0

    out = Path("detector")
    out.mkdir(exist_ok=True)
    shutil.copy(best, out / "impeller.pt")
    (out / "detector.json").write_text(json.dumps({
        "weights": "detector/impeller.pt",
        "run": str(run_dir),
        "start_weights": args.model, "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "scores_on_check_frames": scores,
        "box": meta.get("box", "outer rim of the casting"),
        "crop_per_rim_diameter": meta.get("crop_per_rim_diameter", 1.236),
    }, indent=1), encoding="utf-8")
    print(f"\nSaved {out / 'impeller.pt'} and {out / 'detector.json'}")
    print(f"Curves and sample predictions: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
