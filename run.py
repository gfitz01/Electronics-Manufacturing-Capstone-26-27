"""Training run.

    python run.py          small subset - 120 train / 30 val, about a minute
    python run.py full     full dataset - ~7,300 images, roughly 20-40 minutes

Use the small one to prove the pipeline works. Use the full one for a result
worth reporting.

AlexNet() interviews you in the terminal. Answers for this first pass, in the
order they are actually asked:

SMOKE TEST (python run.py) - just proving the pipeline runs:

    default values for all training parameters ...... n
    image size (height, width, channels) ............ 224, 224, 3
    pretrained model (y/n) .......................... y
    batch size ...................................... 16
    optimizer ....................................... <press Enter>  (Adam)
    learning rate ................................... <press Enter>  (0.001)
    scheduler ....................................... <press Enter>  (none)
    number of epochs ................................ 2
    save the model weights (y/n) .................... y

REAL RUN (python run.py full) - tuned for the best result, not the fastest:

    default values for all training parameters ...... n
    image size (height, width, channels) ............ 224, 224, 3
    pretrained model (y/n) .......................... y
    batch size ...................................... 16
    optimizer ....................................... <press Enter>  (Adam)
    learning rate ................................... 0.0001
    scheduler ....................................... 2        (StepLR)
      step value .................................... 5
      multiplying factor ............................ 0.1
    number of epochs ................................ 15
    save the model weights (y/n) .................... y

Why those values:

* Learning rate 0.0001, not the 0.001 default. 1e-3 with Adam is a rate for
  training from scratch. Applied to a pretrained network it can wash the
  ImageNet features out of the early layers within the first few hundred
  steps - you keep the word "pretrained" and lose most of the benefit. 1e-4
  is the standard fine-tuning rate.

* Batch 16, not 32 or larger. ~6,633 training images means 414 weight
  updates per epoch at batch 16 versus 207 at 32 and 25 at 256. On a dataset
  this size the shortage is updates, not gradient quality. Bigger batches
  finish sooner and learn less unless you scale the learning rate to match.

* StepLR, step 5, gamma 0.1. Drops the rate 10x at epochs 5 and 10, so the
  model takes big steps early and small ones late. This also matters because
  the library always saves the FINAL epoch's weights, not the best epoch's -
  decaying the rate makes the last epoch settle rather than bounce around.

* 15 epochs. Enough to converge with the decay schedule above. Read
  summary.csv afterwards: if val_loss bottoms out early and then climbs,
  that is overfitting and fewer epochs would have been better.

Notes:

* The image size takes all THREE numbers on one line, comma separated.
  Channels must be 1 or 3.

* Use 3 channels, not 1. The casting images are greyscale, but the transform
  replicates them across three channels, and three channels is what lets the
  pretrained AlexNet stem be kept intact instead of rebuilt. With 1 channel
  the first conv layer has to be re-seeded and you lose some of the benefit
  of `pretrained = y`.

* There is no loss-function prompt - it is hard-coded to CrossEntropy.

* Where it says "press Enter without any input" you can just hit Enter and
  take the default.

Two epochs on 120 images will NOT produce a good model. That is fine and
expected - this run exists to prove the plumbing works, not to classify
anything well. What matters is that it finishes and writes:

    results/alexnet-<timestamp>/results.json
    results/alexnet-<timestamp>/summary.csv
    results/alexnet-<timestamp>/predictions.csv

Each run writes its own timestamped directory, so runs accumulate rather than
overwriting each other and you can compare them later.
"""

import sys
from pathlib import Path

from ManufacturingNet.models import AlexNet

#   python run.py          -> data_small  (120 train / 30 val, ~1 minute)
#   python run.py full     -> data        (full dataset, ~20-40 minutes)
full = "full" in [a.lower().lstrip("-") for a in sys.argv[1:]]

HERE = Path(__file__).resolve().parent
dataset = HERE / ("data" if full else "data_small")

train_dir = dataset / "train"
val_dir = dataset / "val"

for directory in (train_dir, val_dir):
    if not directory.is_dir():
        raise SystemExit(
            f"Missing {directory}\n"
            "Run:  python prepare_data.py casting_data"
        )

print("=" * 60)
print(f"Training on: {dataset.name}")
print(f"  train: {train_dir}")
print(f"  val:   {val_dir}")
print("=" * 60)
print()

AlexNet(str(train_dir), str(val_dir))
