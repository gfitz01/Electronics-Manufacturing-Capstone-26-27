# Quickstart: image classification without Google Drive

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9-3.12. `gdown` is optional now — install it only if you want to
retry the Drive download path (`pip install gdown`).

## 2. Get the data once, by hand

Do **not** rely on the Drive download in `ManufacturingNet/datasets/`. Those
file IDs are shared by everyone who ever installed the library, so they hit
Google's per-file download quota, and Google's virus-scan interstitial changes
without warning.

Download the archive once in a browser, then:

```python
from ManufacturingNet.datasets import use_local_archive, prepare_image_folder

# unzip an archive you downloaded yourself
use_local_archive("~/Downloads/casting_data.zip", destination="raw")

# split a flat class-folder layout into the train/val layout the models want
train_dir, val_dir = prepare_image_folder("raw/casting_data",
                                          output="data",
                                          val_split=0.2)
```

`prepare_image_folder` turns this:

```
raw/casting_data/ok/*.jpeg
raw/casting_data/defective/*.jpeg
```

into this:

```
data/train/ok/...        data/val/ok/...
data/train/defective/... data/val/defective/...
```

If your data is already in the train/val layout, skip this step and pass the
two directories straight in.

Sanity-check what you have before training:

```python
from ManufacturingNet.datasets import count_images
print(count_images("data/train"))   # {'defective': 3758, 'ok': 2875}
```

Primary sources, if a Drive link is dead:

| Dataset | Where it actually comes from |
| --- | --- |
| CastingData | Kaggle: *casting product image data for quality inspection* |
| CWRUBearingData | Case Western Reserve University Bearing Data Center |
| PaderbornBearingData | Paderborn University KAt-DataCenter |

## 3. Train

```python
from ManufacturingNet.models import AlexNet

AlexNet(train_dir, val_dir)
```

The constructor runs the interactive prompt sequence and then trains.

## 4. Read the output

Training now writes a machine-readable run report alongside the existing
`loss.png` / `accuracy.png`:

```
results/alexnet-20260913-193427/
├── results.json      full detail — the contract for anything downstream
├── summary.csv       one row per epoch
└── predictions.csv   one row per validation image
```

`results.json` looks like this (abbreviated):

```json
{
  "schema_version": "1.0",
  "run": { "model": "AlexNet", "duration_seconds": 512.4, "...": "..." },
  "hyperparameters": { "epochs": 10, "batch_size": 32, "...": "..." },
  "epochs": [
    { "epoch": 0, "train_loss": 0.4896, "train_accuracy_pct": 78.81,
      "val_loss": 0.5915, "val_accuracy_pct": 69.49, "seconds": 46.1 }
  ],
  "final_metrics": {
    "total_samples": 1300,
    "accuracy_pct": 97.31,
    "macro_f1": 0.9714,
    "class_names": ["ok", "defective"],
    "confusion_matrix": [[788, 16], [19, 477]],
    "confusion_matrix_orientation": "rows=true, cols=predicted",
    "per_class": [ { "class_name": "defective", "precision": 0.9675,
                     "recall": 0.9617, "f1": 0.9646, "...": "..." } ],
    "inspection_summary": {
      "pieces_inspected": 1300,
      "actual_damaged_count": 496,
      "actual_damaged_pct": 38.15,
      "model_flagged_damaged_count": 493,
      "model_flagged_damaged_pct": 37.92,
      "false_negative": 19,
      "missed_damaged_pct": 3.83,
      "false_positive": 16,
      "false_alarm_pct": 1.99
    }
  },
  "predictions": [
    { "file": "data/val/ok/img_00000.jpeg", "true": "ok",
      "predicted": "ok", "correct": true, "confidence": 0.9525 }
  ]
}
```

`inspection_summary` is the block to build a dashboard on:

- **`actual_damaged_pct`** — how much of the batch really is damaged.
- **`model_flagged_damaged_pct`** — how much the model says is damaged.
- **`missed_damaged_pct`** — share of genuinely damaged pieces the model let
  through. On a scrap line this is the number that costs money.
- **`false_alarm_pct`** — share of good pieces wrongly rejected.

`defect_classes` is auto-detected from class names containing `defect`,
`damaged`, `reject`, `crack`, `ng`, etc. Override it explicitly if your names
don't match:

```python
RunReport(model_name="AlexNet",
          class_names=["classA", "classB"],
          defect_classes=["classB"])
```

## 5. See the output format without training

```bash
python examples/demo_report.py
```

This writes a `results.json` from simulated numbers so you can build against
the format before the training run works. **Every number it produces is
synthetic** — the metric computation is real, the inputs are not.
