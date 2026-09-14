"""Show the exact shape of the output a training run now produces.

This does NOT train anything. It drives ManufacturingNet.reporting.RunReport
with simulated epoch losses and simulated validation predictions so you can
see - and start building against - the results.json format before the real
training run is working.

Every number it prints is synthetic. The metric computation is real: the
accuracy, confusion matrix, per-class precision/recall/F1 and the damaged-rate
summary are all computed by the same code path a real run uses.

Run with:  python examples/demo_report.py
"""

import numpy as np

from ManufacturingNet.reporting import RunReport

SEED = 7
N_VAL = 1300          # validation images
TRUE_DEFECT_RATE = 0.38
EPOCHS = 10


def simulated_epochs(report):
    """A plausible-looking loss/accuracy curve."""
    rng = np.random.default_rng(SEED)
    train_loss, val_loss = 0.68, 0.72
    train_acc, val_acc = 58.0, 55.0

    for epoch in range(EPOCHS):
        train_loss *= rng.uniform(0.62, 0.78)
        val_loss *= rng.uniform(0.66, 0.84)
        train_acc += (99.2 - train_acc) * rng.uniform(0.35, 0.55)
        val_acc += (97.0 - val_acc) * rng.uniform(0.30, 0.50)

        report.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            seconds=float(rng.uniform(41, 58)),
            learning_rate=0.001 * (0.9 ** epoch),
        )


def simulated_predictions():
    """Validation labels/predictions for a ~97%-accurate defect classifier."""
    rng = np.random.default_rng(SEED)

    # class 0 = ok, class 1 = defective
    y_true = (rng.random(N_VAL) < TRUE_DEFECT_RATE).astype(int)
    y_pred = y_true.copy()

    # Miss ~4% of genuine defects, and false-alarm on ~2% of good pieces.
    defect_idx = np.flatnonzero(y_true == 1)
    ok_idx = np.flatnonzero(y_true == 0)
    missed = rng.choice(defect_idx, size=int(0.04 * defect_idx.size),
                        replace=False)
    false_alarm = rng.choice(ok_idx, size=int(0.02 * ok_idx.size),
                             replace=False)
    y_pred[missed] = 0
    y_pred[false_alarm] = 1

    # Wrong predictions get lower confidence, as they usually do.
    confidences = rng.uniform(0.90, 0.999, size=N_VAL)
    wrong = y_true != y_pred
    confidences[wrong] = rng.uniform(0.51, 0.78, size=int(wrong.sum()))

    files = [f"data/val/{'defective' if t else 'ok'}/img_{i:05d}.jpeg"
             for i, t in enumerate(y_true)]

    return y_true, y_pred, confidences, files


def main():
    report = RunReport(
        model_name="AlexNet",
        class_names=["ok", "defective"],
        hyperparameters={
            "epochs": EPOCHS,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "image_size": [224, 224, 3],
            "pretrained": True,
            "device": "cuda",
            "train_images": 5200,
            "val_images": N_VAL,
        },
        notes=("SYNTHETIC DEMO - these numbers were generated to illustrate "
               "the results.json format. They are not from a real training "
               "run."),
    )

    simulated_epochs(report)

    y_true, y_pred, confidences, files = simulated_predictions()
    report.set_predictions(y_true, y_pred, confidences=confidences,
                           file_paths=files)

    report.print_summary()
    paths = report.save(output_dir="results")

    print()
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
