"""Training run.

    python run.py                  small subset, no augmentation  (~1 minute)
    python run.py full             full dataset, no augmentation
    python run.py full augment     full dataset, with augmentation
    python run.py clean            leak-free dataset, no augmentation
    python run.py clean augment    leak-free dataset, with augmentation

    python run.py --data defect_type_front/fray augment
                                   any directory holding train/ and val/,
                                   which is how the stage-two defect-type
                                   folders are trained

The 'clean' dataset is data_clean/, built by make_clean_split.py: the
published test split contains 64 images byte-identical to training images,
so those are removed from TRAINING and the 715-image validation set is left
whole. Report results from the clean runs.

Use the small one to prove the pipeline works. Use the full ones for results
worth reporting -- and run BOTH full variants if you want the augmentation
comparison, since that is the experiment.

Each run writes its own timestamped directory under results/, named for the
model and whether augmentation was on, so runs accumulate rather than
overwriting each other.

AlexNet() interviews you in the terminal. Answers, in the order asked:

    default values for all training parameters ...... n
    image size (height, width, channels) ............ 224, 224, 3
    pretrained model (y/n) .......................... y
    batch size ...................................... 16
    optimizer ....................................... <press Enter>  (Adam)
    learning rate ................................... 0.0001
    scheduler ....................................... 2        (StepLR)
      step value .................................... 5
      multiplying factor ............................ 0.1
    number of epochs ................................ 2 small / 15 full
    save the model weights (y/n) .................... y

USE THE SAME ANSWERS FOR BOTH FULL RUNS. The comparison is only meaningful if
augmentation is the single thing that differs between them.

Why those values:

* Learning rate 0.0001, not the 0.001 default. 1e-3 with Adam is a rate for
  training from scratch. Applied to a pretrained network it can wash the
  ImageNet features out of the early layers within the first few hundred
  steps - you keep the word "pretrained" and lose most of the benefit. 1e-4
  is the standard fine-tuning rate.

* Batch 16, not 32 or larger. ~6,633 training images means 414 weight
  updates per epoch at batch 16 versus 207 at 32 and 25 at 256. On a dataset
  this size the shortage is updates, not gradient quality.

* StepLR, step 5, gamma 0.1. Drops the rate 10x at epochs 5 and 10, so the
  model takes big steps early and small ones late. This also matters because
  the library always saves the FINAL epoch's weights, not the best epoch's -
  decaying the rate makes the last epoch settle rather than bounce around.

* 15 epochs. With augmentation, consider more: augmented training is harder
  per epoch, so it converges more slowly. If summary.csv shows val_loss still
  falling at epoch 14, the augmented run wants 20-25.
"""

import sys
from pathlib import Path

from ManufacturingNet.models import AlexNet

argv = sys.argv[1:]
data_dir, rest, i = None, [], 0
while i < len(argv):
    a = argv[i]
    if a in ("--data", "-d"):
        i += 1
        if i >= len(argv):
            raise SystemExit("--data needs a directory containing train/ and val/")
        data_dir = argv[i]
    elif a.startswith("--data="):
        data_dir = a.split("=", 1)[1]
    else:
        rest.append(a)
    i += 1

flags = [a.lower().lstrip("-") for a in rest]
full = "full" in flags
clean = "clean" in flags
augment = "augment" in flags or "aug" in flags

HERE = Path(__file__).resolve().parent
if data_dir:
    dataset = Path(data_dir)
    if not dataset.is_absolute():
        dataset = HERE / dataset
elif clean:
    dataset = HERE / "data_clean"     # duplicates removed from training
elif full:
    dataset = HERE / "data"
else:
    dataset = HERE / "data_small"

train_dir = dataset / "train"
val_dir = dataset / "val"

for directory in (train_dir, val_dir):
    if not directory.is_dir():
        raise SystemExit(
            f"Missing {directory}\n" + (
                "A --data directory must contain train/ and val/ subfolders."
                if data_dir else
                "Run:  python prepare_data.py casting_data")
        )

print("=" * 60)
print(f"Training on: {dataset.name}")
print(f"  train:      {train_dir}")
print(f"  val:        {val_dir}")
print(f"  augment:    {'ON  (training images only)' if augment else 'off'}")
print("=" * 60)
print()

AlexNet(str(train_dir), str(val_dir), augment=augment)
